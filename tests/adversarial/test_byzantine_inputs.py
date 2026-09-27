"""
Adversarial / byzantine input tests for the fitsproof enforcement surface.

Sources: [1] Roofline Model (cost model inputs), [7] GPTQ int8 symmetric
quantisation (quantisation error bounds), [13] GGML int4 k-quants (int4
packing and nibble sign conventions).

Each test names the fault it detects:

  test_quantize_rejects_nan_weights:
      Fault: NaN weights silently quantising to integer garbage (fail closed).
  test_quantize_rejects_inf_weights:
      Fault: inf weights silently quantising to integer garbage (fail closed).
  test_int8_asym_constant_row_no_int32_overflow:
      Fault: the range==0 epsilon guard makes zp = round(-c/eps) overflow
      int32 for large constant rows, corrupting the whole row.
  test_int4_asym_constant_row_no_int32_overflow:
      Same overflow fault on the int4 asymmetric path.
  test_int4_asym_known_answer:
      KAT: row [0, 4] must round-trip exactly (hand-derived below);
      a wrong zero-point or a sign-extending nibble unpack fails this.
  test_plan_rejects_unknown_quant:
      Fault: cost.estimate falls back to fp32 for unknown quant names
      (.get(quant, 32.0)), silently planning a typo as float32.
  test_plan_rejects_non_positive_or_non_finite_budget:
      Fault: a 0/negative/NaN budget reaching the verdict comparison.
  test_plan_rejects_bad_context_len:
      Fault: context_len < 1 or non-int producing nonsense KV-cache sizes.
  test_plan_survives_huge_context:
      Fault: integer overflow / crash on a 1e9-token context request.
  test_parse_budget_rejects_hostile_strings:
      Fault: injection strings, empty/whitespace input turning a
      refusal into an int() traceback or worse, an eval.
  test_parse_budget_rejects_non_finite_numbers:
      Fault: NaN/inf/negative numeric budgets reaching int() (OverflowError).
  test_guard_hostile_budget_never_allocates:
      Fault: a malformed budget reaching the wrapped callable (the guard
      must fail closed before the caller allocates).
  test_cli_hostile_args_fail_closed:
      Fault: CLI traceback on hostile argv (must print ERROR, exit 2).
  test_server_refuses_request_over_budget:
      Fault: HTTP server completing a request whose predicted peak exceeds
      the declared budget instead of refusing loudly (503 + record).
  test_server_completion_carries_admission_record:
      Fault: HTTP responses omitting the plan/admission record (M2).
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from urllib.error import HTTPError

import numpy as np
import pytest

from fitsproof.client import DoesNotFit, _parse_budget, guard
from fitsproof.contract.plan import plan
from fitsproof.contract.probe import MachineProfile
from fitsproof.engine.model import REFERENCE_CONFIG
from fitsproof.engine.quant import dequantize, quantize


def _machine() -> MachineProfile:
    """Deterministic synthetic machine profile (no wall clock, no probe)."""
    return MachineProfile(
        hostname="byzantine-test",
        platform_str="linux",
        measured_at=1_700_000_000.0,
        memory_bandwidth_bps=20e9,
        gemm_throughput_flops=100e9,
        memory_bytes=32 * 1024**3,
        gpu_memory_bytes=0,
        cpu_count=8,
    )


# ---------------------------------------------------------------------------
# Quantisation: fail closed on corrupted weights, no silent overflow
# ---------------------------------------------------------------------------


def test_quantize_rejects_nan_weights() -> None:
    w = np.array([[1.0, np.nan], [0.5, 0.25]], dtype=np.float32)
    for mode in ("int8_sym", "int8_asym", "int4_sym", "int4_asym"):
        with pytest.raises(ValueError, match="NaN or inf"):
            quantize(w, mode)


def test_quantize_rejects_inf_weights() -> None:
    w = np.array([[1.0, np.inf], [0.5, -np.inf]], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or inf"):
        quantize(w, "int4_asym")


def test_int8_asym_constant_row_no_int32_overflow() -> None:
    # Constant row c = 1e6: range == 0. An epsilon guard (range -> 1e-8)
    # computes zp = round(-1e6 / 3.9e-11) ~ -2.5e16 -> int32 overflow and a
    # corrupted row (~1e6 off). The safe guard reconstructs within float32 eps.
    w = np.full((4, 8), 1e6, dtype=np.float32)
    back = dequantize(quantize(w, "int8_asym"))
    assert np.allclose(back, w, rtol=1e-5, atol=1.0), float(np.abs(back - w).max())


def test_int4_asym_constant_row_no_int32_overflow() -> None:
    w = np.full((4, 8), 1e6, dtype=np.float32)
    back = dequantize(quantize(w, "int4_asym"))
    assert np.allclose(back, w, rtol=1e-5, atol=1.0), float(np.abs(back - w).max())


def test_int4_asym_known_answer() -> None:
    # Hand-derived KAT (source [7] zero-point formulation, 4-bit range [0,15]):
    # row = [0.0, 4.0]; range = 4; s = 4/15; zp = round(-0/s) = 0;
    # q = round(w/s) = [0, round(15)] = [0, 15]; dequant = (q - zp) * s = [0, 4].
    # A sign-extending nibble unpack maps q=15 -> -1 and fails this.
    w = np.array([[0.0, 4.0]], dtype=np.float32)
    qw = quantize(w, "int4_asym")
    assert qw.zero_points is not None
    assert int(qw.zero_points[0]) == 0
    back = dequantize(qw)
    assert np.allclose(back, w, atol=1e-5), back


# ---------------------------------------------------------------------------
# plan(): fail closed on hostile parameters
# ---------------------------------------------------------------------------


def test_plan_rejects_unknown_quant() -> None:
    for bad in ("int2", "INT4_SYM", "int4\x00sym", "", "fp8"):
        with pytest.raises(ValueError, match="unknown quant"):
            plan(REFERENCE_CONFIG, _machine(), context_len=64, budget_bytes=10**10, quant=bad)


@pytest.mark.parametrize("budget", [0, -1, -(10**9), float("nan"), float("inf"), float("-inf")])
def test_plan_rejects_non_positive_or_non_finite_budget(budget: float) -> None:
    with pytest.raises(ValueError, match="budget_bytes"):
        plan(REFERENCE_CONFIG, _machine(), context_len=64, budget_bytes=budget)


@pytest.mark.parametrize("ctx", [0, -5, 1.5, True, "64", None])
def test_plan_rejects_bad_context_len(ctx: object) -> None:
    with pytest.raises(ValueError, match="context_len"):
        plan(REFERENCE_CONFIG, _machine(), context_len=ctx, budget_bytes=10**10)  # type: ignore[arg-type]


def test_plan_survives_huge_context() -> None:
    p = plan(REFERENCE_CONFIG, _machine(), context_len=10**9, budget_bytes=4 * 1024**3)
    assert p.predicted_peak_bytes > 0
    assert p.verdict.value in ("fits", "fits_with_degradation", "does_not_fit")


# ---------------------------------------------------------------------------
# Budget parser: hostile strings and numbers
# ---------------------------------------------------------------------------

HOSTILE_BUDGETS = [
    "(1+1)GB",  # would succeed under eval()
    "4GB; rm -rf /",
    "",
    "   ",
    "NaN",
    "nanGiB",
    "inf",
    "InfinityMB",
    "-4GiB",
    "__import__('os')",
]


@pytest.mark.parametrize("raw", HOSTILE_BUDGETS)
def test_parse_budget_rejects_hostile_strings(raw: str) -> None:
    with pytest.raises(ValueError):
        _parse_budget(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4GiB", 4 * 1024**3),
        ("512MiB", 512 * 1024**2),
        ("4096", 4096),
        (8 * 1024**3, 8 * 1024**3),
        # Python's float() accepts unicode decimal digits, so a fullwidth
        # "４GiB" parses to the same correct value — documented, not a fault.
        ("４GiB", 4 * 1024**3),
    ],
)
def test_parse_budget_control_accepts_valid(raw: str | int, expected: int) -> None:
    # Control: the rejection tests above are only meaningful if valid input
    # still parses to the right value (non-vacuous).
    assert _parse_budget(raw) == expected


@pytest.mark.parametrize("num", [float("nan"), float("inf"), float("-inf"), 0, -1])
def test_parse_budget_rejects_non_finite_numbers(num: float) -> None:
    with pytest.raises(ValueError):
        _parse_budget(num)


def test_guard_hostile_budget_never_allocates() -> None:
    loaded: list[str] = []

    @guard(budget="not-a-budget")
    def load_model() -> int:
        loaded.append("allocated")
        return 1

    with pytest.raises(ValueError):
        load_model()
    assert loaded == []


def test_guard_refusal_raises_does_not_fit() -> None:
    # Control for the refusal path: a parseable-but-impossible budget must
    # raise DoesNotFit (not ValueError), still before any allocation.
    loaded: list[str] = []

    @guard(budget="1MiB")
    def load_model() -> int:
        loaded.append("allocated")
        return 1

    with pytest.raises(DoesNotFit):
        load_model()
    assert loaded == []


# ---------------------------------------------------------------------------
# CLI: hostile argv must fail closed (no traceback)
# ---------------------------------------------------------------------------

CLI_HOSTILE = [
    ("admit", "--budget-gb", "nan"),
    ("admit", "--budget-gb", "-1"),
    ("admit", "--budget-gb", "inf"),
    ("admit", "--context", "-5"),
    ("admit", "--quant", "int2"),
    ("plan", "--quant", "fp8", "--budget-gb", "4"),
]


@pytest.mark.parametrize("argv", CLI_HOSTILE)
def test_cli_hostile_args_fail_closed(argv: tuple[str, ...]) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "fitsproof.cli", *argv],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0, f"hostile argv {argv} exited 0: {proc.stdout}"
    assert "Traceback" not in proc.stderr, f"hostile argv {argv} crashed:\n{proc.stderr}"
    assert "ERROR:" in proc.stderr, f"hostile argv {argv} gave no error message: {proc.stderr!r}"


# ---------------------------------------------------------------------------
# HTTP server: must refuse loudly, not complete, over-budget requests
# ---------------------------------------------------------------------------


def _start_test_server(budget_bytes: int):
    from fitsproof.engine.model import get_reference_bundle
    from fitsproof.engine.server import start_server
    from fitsproof.engine.transformer import Transformer

    cfg, weights = get_reference_bundle()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = start_server(
        Transformer(cfg, weights),
        cfg,
        host="127.0.0.1",
        port=port,
        block=False,
        budget_bytes=budget_bytes,
    )
    return srv, port


def _post_completion(port: int) -> tuple[int, dict]:
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_refuses_request_over_budget() -> None:
    srv, port = _start_test_server(budget_bytes=1024)  # 1 KiB: cannot fit
    try:
        status, body = _post_completion(port)
        assert status == 503, f"expected 503 refusal, got {status}: {body}"
        assert body["error"]["type"] == "fitsproof_refused"
        assert body["fitsproof"]["admission"] == "refused"
        assert body["fitsproof"]["budget_bytes"] == 1024
        assert "budget" in body["fitsproof"]["message"]
    finally:
        srv.shutdown()


def test_server_completion_carries_admission_record() -> None:
    srv, port = _start_test_server(budget_bytes=4 * 1024**3)
    try:
        status, body = None, None
        last_err: Exception | None = None
        for _ in range(30):
            try:
                status, body = _post_completion(port)
                break
            except (HTTPError, OSError) as e:
                last_err = e
                time.sleep(0.2)
        assert body is not None, f"no completion: {last_err}"
        assert status == 200, f"expected 200, got {status}: {body}"
        assert body["fitsproof"]["admission"] in ("admitted", "degraded")
        assert body["fitsproof"]["budget_bytes"] == 4 * 1024**3
        assert body["fitsproof"]["message"].startswith(("ADMITTED", "DEGRADED"))
        assert body["fitsproof"]["predicted_peak_bytes"] > 0
    finally:
        srv.shutdown()
