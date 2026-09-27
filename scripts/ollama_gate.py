#!/usr/bin/env python3
"""
Pre-flight admission gate for ollama models.

Reads a model's real architecture from the running ollama daemon (REST
/api/show), plans it against a declared memory budget, and admits,
degrades, or refuses BEFORE the model is pulled into memory. Designed to
be chained before an ollama invocation:

    ./scripts/ollama_gate.py gemma3:4b --budget-gb 12 --context 4096 \
        && ollama run gemma3:4b

Exit codes:
    0  ADMITTED (or DEGRADED only with --allow-degrade)
    2  REFUSED, or DEGRADED without --allow-degrade (a gate cannot apply
       a fitsproof degradation to an external engine, so a degradation
       would be a false green light for the chained command)
    1  gate itself failed (daemon unreachable, unreadable metadata)

Fail-closed: if the model metadata cannot be read, the gate refuses to
admit. It never guesses a model shape.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.error
import urllib.request

from fitsproof.contract.admit import admit
from fitsproof.contract.plan import plan
from fitsproof.contract.probe import probe
from fitsproof.engine.model import ModelConfig

OLLAMA_HOST = "http://127.0.0.1:11434"


def show_model(name: str, host: str = OLLAMA_HOST) -> dict:
    """POST /api/show and return the decoded response."""
    req = urllib.request.Request(
        f"{host}/api/show",
        data=json.dumps({"model": name}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def map_quant(gguf_quant: str) -> str:
    """Map a GGUF quantisation label (Q4_K_M, F16, ...) to a fitsproof quant.

    Q2/Q3/Q4 -> int4_sym, Q5/Q6/Q8 -> int8_sym, F16/F32/BF16 -> none.
    Unknown labels raise: the gate must not silently pick a wrong class.
    """
    q = gguf_quant.upper()
    if q.startswith(("Q2", "Q3", "Q4")):
        return "int4_sym"
    if q.startswith(("Q5", "Q6", "Q8")):
        return "int8_sym"
    if q.startswith(("F16", "F32", "BF16", "IQ")):
        return "none"
    raise ValueError(f"unmapped GGUF quantisation: {gguf_quant!r}")


def _read_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _skip_value(f, type_id: int):
    if type_id in (0, 1, 7):
        return f.read(1)
    if type_id in (2, 3):
        return f.read(2)
    if type_id in (4, 5, 6):
        return f.read(4)
    if type_id == 8:
        return _read_string(f)
    if type_id == 9:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        return [_skip_value(f, elem_type) for _ in range(count)]
    if type_id in (10, 11, 12):
        return f.read(8)
    raise ValueError(f"unknown GGUF metadata type {type_id}")


def gguf_vocab_size(blob_path: str) -> int:
    """Walk a GGUF file's metadata and return the length of tokenizer.ggml.tokens.

    The vocabulary size sets the embedding-matrix cost; guessing it would
    silently mis-size the plan, so an unreadable file raises.
    """
    with open(blob_path, "rb") as f:
        magic, version, _tensors, kv_count = struct.unpack("<IIQQ", f.read(24))
        if magic != 0x46554747:
            raise ValueError(f"{blob_path}: not a GGUF file (magic {magic:#x})")
        for _ in range(kv_count):
            key = _read_string(f)
            (type_id,) = struct.unpack("<I", f.read(4))
            if key == "tokenizer.ggml.tokens":
                if type_id != 9:
                    raise ValueError("tokenizer.ggml.tokens is not an array")
                (elem_type,) = struct.unpack("<I", f.read(4))
                (count,) = struct.unpack("<Q", f.read(8))
                if elem_type != 8:
                    raise ValueError("tokenizer.ggml.tokens is not a string array")
                return count
            _skip_value(f, type_id)
    raise ValueError(f"tokenizer.ggml.tokens not found (GGUF v{version})")


def blob_path(modelfile: str) -> str:
    """Extract the FROM blob path from an ollama modelfile."""
    for line in modelfile.splitlines():
        if line.strip().upper().startswith("FROM "):
            return line.strip()[5:].strip()
    raise ValueError("no FROM line in modelfile")


def main() -> int:
    parser = argparse.ArgumentParser(description="fitsproof admission gate for ollama models")
    parser.add_argument("model", help="ollama model name (e.g. gemma3:4b)")
    parser.add_argument(
        "--budget-gb", type=float, required=True, help="declared memory budget in GB"
    )
    parser.add_argument("--context", type=int, default=4096, help="context length to plan for")
    parser.add_argument("--host", default=OLLAMA_HOST, help="ollama daemon base URL")
    parser.add_argument(
        "--allow-degrade",
        action="store_true",
        help="chain anyway on a DEGRADED verdict (you must wire the degradation yourself)",
    )
    args = parser.parse_args()

    try:
        show = show_model(args.model, args.host)
    except (urllib.error.URLError, OSError) as exc:
        print(f"GATE ERROR: ollama daemon unreachable at {args.host}: {exc}", file=sys.stderr)
        return 1

    details = show.get("details", {})
    family = details.get("family", "")
    quant_label = details.get("quantization_level", "")
    info = show.get("model_info", {})

    try:
        quant = map_quant(quant_label)
        blob = blob_path(show.get("modelfile", ""))
        vocab = gguf_vocab_size(blob)
        cfg = ModelConfig(
            vocab_size=vocab,
            hidden_size=int(info[f"{family}.embedding_length"]),
            num_layers=int(info[f"{family}.block_count"]),
            num_heads=int(info[f"{family}.attention.head_count"]),
            num_kv_heads=int(info[f"{family}.attention.head_count_kv"]),
            intermediate_size=int(info[f"{family}.feed_forward_length"]),
            max_seq_len=min(int(info[f"{family}.context_length"]), args.context),
            dtype="float32",
        )
        cfg.validate()
    except (KeyError, ValueError) as exc:
        print(f"GATE REFUSED: cannot read model metadata for {args.model}: {exc}", file=sys.stderr)
        return 2

    machine = probe()
    budget = int(args.budget_gb * 1e9)
    p = plan(cfg, machine, args.context, budget, quant)
    record = admit(p)

    print(f"model:   {args.model} ({details.get('parameter_size', '?')}, {quant_label} -> {quant})")
    print(
        f"shape:   {cfg.num_layers}L x {cfg.hidden_size}h, heads {cfg.num_heads}"
        f"/{cfg.num_kv_heads}, vocab {cfg.vocab_size}, ctx {args.context}"
    )
    print(f"budget:  {args.budget_gb:g} GB")
    print(record.message)
    if record.applied_degradation is not None:
        sys.stdout.flush()
        print(f"DEGRADATION RECORD: {record.applied_degradation.description}", file=sys.stderr)

    if p.verdict.value == "does_not_fit":
        return 2
    if p.verdict.value == "fits_with_degradation" and not args.allow_degrade:
        sys.stdout.flush()
        print(
            "GATE: config only fits after a declared degradation, and this gate cannot "
            "apply it to an external engine. Not chaining. Re-run with --allow-degrade "
            "only if you will wire the degradation yourself.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
