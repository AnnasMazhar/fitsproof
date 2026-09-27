"""
fitsproof.contract.cost — Analytical cost model for LLM inference.

The roofline model (Williams et al. 2009) characterises compute-bound vs
memory-bandwidth-bound performance. For LLM decoding:

  decode is memory-bandwidth-bound:
    tok/s = effective_bandwidth / bytes_per_token_of_weights

  prefill is compute-bound (at sufficient batch size):
    TTFT ≈ 2 * n_params * seq_len / peak_FLOPS

KV cache size per token (per layer):
    kv_bytes_per_token_per_layer = 2 * n_kv_heads * head_dim * bytes_per_element

Peak VRAM / RAM usage:
    weight_bytes + kv_cache_bytes + activations_overhead

Sources:
  - Williams et al. 2009 (Roofline model), https://dl.acm.org/doi/10.1145/1498765.1498785
  - Kaplan et al. 2020 (Scaling Laws), https://arxiv.org/abs/2001.08361
  - Sheng et al. 2023 (FlexGen), https://arxiv.org/abs/2303.06865
  - Ainslie et al. 2023 (GQA), https://arxiv.org/abs/2305.13245

All formulas are documented in docs/IMPLEMENTATION-NOTES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from fitsproof.contract.probe import MachineProfile
from fitsproof.engine.model import ModelConfig


BYTES_PER_DTYPE: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "int8": 1,
    "int4": 1,  # 0.5 bytes actual, rounded up to 1 for numpy storage
    "int4_packed": 1,  # per packed byte covers 2 int4 values
}

BITS_PER_QUANT: dict[str, float] = {
    "float32": 32.0,
    "float16": 16.0,
    "int8_sym": 8.0,
    "int8_asym": 8.0,
    "int4_sym": 4.0,
    "int4_asym": 4.0,
    "none": 32.0,
}


@dataclass
class CostEstimate:
    """
    Analytical cost model output for a given (model, quant, context, machine) configuration.

    All byte counts are in bytes. tok_s is tokens/second.
    Fields ending _ci are 95% confidence intervals (lower, upper) added by calibrate.
    """

    weight_bytes: int          # model weight storage
    kv_cache_bytes: int        # KV cache for full context window
    activation_bytes: int      # peak activation buffer (one batch, one layer)
    total_peak_bytes: int      # weight + kv_cache + activation (conservative sum)
    predicted_tok_s: float     # decode throughput (tokens/second)
    predicted_ttft_s: float    # time to first token (seconds), prefill
    arithmetic_intensity: float  # FLOPS/byte for decode step


def weight_bytes(cfg: ModelConfig, quant: str = "none") -> int:
    """
    Compute model weight storage in bytes for the given quantisation.

    Counts all weight matrices (embeddings + per-layer attention + FFN + norms).
    Norm weights are float32 regardless of quant (standard practice).

    Formula: sum over each weight matrix of (elements * bits/8).

    Fault detected: forgetting the embedding table doubles the error for
    models with large vocabularies; tested with a known-config reference.
    """
    bits = BITS_PER_QUANT.get(quant, 32.0)
    bytes_per_element = bits / 8.0

    d = cfg.hidden_size
    h = cfg.num_heads
    kv_h = cfg.num_kv_heads
    hd = cfg.head_dim
    ff = cfg.intermediate_size
    V = cfg.vocab_size
    L = cfg.num_layers

    # Embedding table (float32)
    embed_bytes = V * d * 4

    # Per-layer attention projections (Q, K, V, O)
    attn_bytes_per_layer = (
        h * hd * d  # Q: (n_heads*head_dim, hidden)
        + kv_h * hd * d  # K
        + kv_h * hd * d  # V
        + d * h * hd     # O
    ) * bytes_per_element

    # Per-layer FFN (gate + up + down)
    ffn_bytes_per_layer = (ff * d + ff * d + d * ff) * bytes_per_element

    # Per-layer norms (float32, small)
    norm_bytes_per_layer = 2 * d * 4  # attn_norm + ffn_norm

    # Final norm + unembed
    final_bytes = d * 4 + V * d * 4  # float32

    total = int(
        embed_bytes
        + L * (attn_bytes_per_layer + ffn_bytes_per_layer + norm_bytes_per_layer)
        + final_bytes
    )
    return total


def kv_cache_bytes(cfg: ModelConfig, context_len: int, quant: str = "none") -> int:
    """
    KV cache memory for *context_len* tokens across all layers.

    Formula (from GQA paper, Ainslie et al. 2023):
        2 * n_layers * n_kv_heads * context_len * head_dim * bytes_per_element

    The factor 2 is for K and V tensors.

    Fault detected: omitting the factor 2 halves the estimate and causes
    budget violations to go undetected; tested with known-config reference.
    """
    bits = BITS_PER_QUANT.get(quant, 32.0)
    bytes_per_element = bits / 8.0
    return int(
        2 * cfg.num_layers * cfg.num_kv_heads * context_len * cfg.head_dim * bytes_per_element
    )


def activation_bytes(cfg: ModelConfig) -> int:
    """
    Peak activation buffer for one token, one layer.

    Conservative estimate: two full hidden-size buffers (input + output)
    plus the intermediate FFN buffer.

    This is a heuristic overhead; exact value depends on implementation.
    """
    return int((2 * cfg.hidden_size + cfg.intermediate_size) * 4)  # float32


def decode_tok_s(
    cfg: ModelConfig,
    machine: MachineProfile,
    quant: str = "none",
    bandwidth_utilisation: float = 0.6,
) -> float:
    """
    Predict decode throughput (tokens/second) using the roofline model.

    Decode is memory-bandwidth-bound:
        tok/s = effective_bandwidth / bytes_per_token_streamed

    bytes_per_token_streamed ≈ weight_bytes (we stream all weights per token
    in the limit of large context where the KV cache contribution is comparable).

    bandwidth_utilisation: empirical fraction of peak bandwidth achievable
    (0.6 is a conservative default; calibrate.py fits this from measurements).

    Source: Sheng et al. 2023 (FlexGen), Sec 3.1 — decode throughput is
    proportional to effective memory bandwidth divided by model size.

    Fault detected: using raw peak bandwidth (utilisation=1.0) over-predicts
    by ~40-67% vs measured; calibrate.py tests this on held-out configs.
    """
    w_bytes = weight_bytes(cfg, quant)
    effective_bw = machine.memory_bandwidth_bps * bandwidth_utilisation
    if w_bytes <= 0 or effective_bw <= 0:
        return 0.0
    return effective_bw / w_bytes


def prefill_ttft_s(
    cfg: ModelConfig,
    machine: MachineProfile,
    seq_len: int,
    quant: str = "none",
) -> float:
    """
    Predict time-to-first-token (TTFT) for a prompt of *seq_len* tokens.

    Prefill is compute-bound at large batch (single sequence on CPU is
    effectively batch=1, often memory-bandwidth-bound too, but we use the
    compute bound as an upper bound estimate).

    FLOPS for prefill ≈ 2 * n_params * seq_len  (forward pass, 2 FLOPS per multiply-add).

    Source: Kaplan et al. 2020 (Scaling Laws), Appendix D.

    Fault detected: forgetting the factor 2 halves the FLOPS estimate and
    underpredicts TTFT; tested with a known model size.
    """
    # Approximate n_params from weight tensor counts
    d = cfg.hidden_size
    h = cfg.num_heads
    kv_h = cfg.num_kv_heads
    hd = cfg.head_dim
    ff = cfg.intermediate_size
    V = cfg.vocab_size
    L = cfg.num_layers

    params = (
        V * d  # embed
        + L * (h * hd * d + kv_h * hd * d + kv_h * hd * d + d * h * hd)  # attn
        + L * (ff * d + ff * d + d * ff)  # ffn
        + V * d  # unembed
    )
    flops = 2.0 * params * seq_len
    if machine.gemm_throughput_flops <= 0:
        return float("inf")
    return flops / machine.gemm_throughput_flops


def arithmetic_intensity(cfg: ModelConfig, quant: str = "none") -> float:
    """
    Arithmetic intensity for one decode step: FLOPS per byte read.

    = (2 * n_params) / weight_bytes

    For float32, this is 2 * n_params / (4 * n_params) = 0.5 FLOP/byte.
    For int8, it doubles to ~1 FLOP/byte.

    This determines whether the workload is roofline-compute-bound or
    memory-bandwidth-bound. CPU peak: ~1-4 FLOP/byte at fp32 DRAM speed.
    Decode is always memory-bandwidth-bound in practice.
    """
    d = cfg.hidden_size
    h = cfg.num_heads
    kv_h = cfg.num_kv_heads
    hd = cfg.head_dim
    ff = cfg.intermediate_size
    V = cfg.vocab_size
    L = cfg.num_layers
    params = (
        V * d
        + L * (h * hd * d + kv_h * hd * d + kv_h * hd * d + d * h * hd)
        + L * (ff * d + ff * d + d * ff)
        + V * d
    )
    w = weight_bytes(cfg, quant)
    if w <= 0:
        return 0.0
    return (2.0 * params) / w


def estimate(
    cfg: ModelConfig,
    machine: MachineProfile,
    context_len: int,
    quant: str = "none",
    bandwidth_utilisation: float = 0.6,
) -> CostEstimate:
    """
    Full cost estimate for a given (model, machine, context, quant) configuration.

    Returns a CostEstimate with all memory and throughput fields populated.
    """
    w = weight_bytes(cfg, quant)
    kv = kv_cache_bytes(cfg, context_len, quant)
    act = activation_bytes(cfg)
    total = w + kv + act
    tok_s = decode_tok_s(cfg, machine, quant, bandwidth_utilisation)
    ttft = prefill_ttft_s(cfg, machine, context_len, quant)
    ai = arithmetic_intensity(cfg, quant)

    return CostEstimate(
        weight_bytes=w,
        kv_cache_bytes=kv,
        activation_bytes=act,
        total_peak_bytes=total,
        predicted_tok_s=tok_s,
        predicted_ttft_s=ttft,
        arithmetic_intensity=ai,
    )
