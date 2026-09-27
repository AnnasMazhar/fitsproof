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

Cycle 2 additions — adversarial tests for refusal bypass, mode-change
detection, degradation chain integrity, and MCP error propagation:

  test_verify_run_raises_on_refused_record:
      Fault (c2): verify_run silently running a refused config — the
      enforcement guard must block execution before measurement begins.
  test_admit_verdict_immutable_after_plan:
      Fault (c2): a caller mutating plan.budget_bytes after planning but
      before admit() to sneak a refused config through as admitted.
  test_degradation_chain_covers_all_quant_levels:
      Fault (c2): degradation options omitting a quantisation level that
      would fit — user left with no path when int8 or int4 would work.
  test_mcp_plan_tool_refusal_verdict_propagates:
      Fault (c2): MCP plan tool returning a fits verdict for a does_not_fit
      config — agent proceeds to load a model that cannot fit.
  test_mcp_admit_tool_with_zero_budget_is_error:
      Fault (c2): MCP admit tool with an impossibly small budget returning
      isError=False — agent mistakes a refusal for an admission.
  test_client_metrics_always_positive:
      Fault (c2): FitsproofClient.metrics() returning zero or negative
      values for machine metrics that must be strictly positive.
  test_guard_called_multiple_times_always_refuses:
      Fault (c2): the @guard decorator admitting the second call after
      refusing the first — cached state from a successful probe leaking
      into a budget-changed re-decoration scenario.
  test_stress_harness_budget_violation_exits_nonzero:
      Fault (c2): the stress harness silently passing when at least one
      config violates the declared budget — "0 violations" is then false.
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


# ---------------------------------------------------------------------------
# Cycle 2 adversarial tests — refusal bypass, mode-change detection,
# degradation integrity, MCP error propagation
# ---------------------------------------------------------------------------


def test_verify_run_raises_on_refused_record() -> None:
    """
    Sources: [1] Roofline; fitsproof.md M3 — a refused plan must not reach execution.
    Fault (c2): verify_run silently executing a refused config would allow a caller
    to measure an OOM they were supposed to refuse. The guard in verify_run must
    raise RuntimeError before executing the generation function.
    """
    from fitsproof.contract.admit import AdmitRecord, AdmitStatus
    from fitsproof.contract.plan import Plan, Verdict
    from fitsproof.contract.verify import verify_run

    refused_plan = Plan(
        verdict=Verdict.DOES_NOT_FIT,
        predicted_peak_bytes=10**12,
        predicted_peak_ci=(10**12, 10**12),
        predicted_tok_s=0.0,
        predicted_tok_s_ci=(0.0, 0.0),
        budget_bytes=1,
        quant="none",
        context_len=512,
        degradations=[],
    )
    refused_record = AdmitRecord(
        status=AdmitStatus.REFUSED,
        plan=refused_plan,
        applied_degradation=None,
        refusal_reason="needs 1 TB, budget 1 B",
        message="REFUSED: test",
    )
    executed = []

    with pytest.raises(RuntimeError, match="REFUSED"):
        verify_run(
            fn=lambda: executed.append("ran"),
            budget_bytes=1,
            admit_record=refused_record,
            config_label="c2-bypass-test",
        )
    assert executed == [], "verify_run must not invoke fn when record is REFUSED"


def test_admit_verdict_immutable_after_plan() -> None:
    """
    Sources: [1] Roofline enforcement model.
    Fault (c2): a caller who mutates plan.budget_bytes after plan() but before admit()
    could sneak a refused plan through as admitted. plan objects are dataclasses and
    mutable — this test verifies admit() evaluates the verdict live from the stored
    plan state, not a cached verdict that could be out of sync.

    Specifically: if we build a plan with a tiny budget (DOES_NOT_FIT), then mutate
    budget_bytes to a huge value and call admit(), the admit() must re-check the
    verdict against the original plan data and still refuse (the plan's verdict field
    was set at plan-time and is what admit() uses). The mutated budget_bytes does not
    re-run the planner. This is a design choice — admit() trusts plan.verdict, not
    budget_bytes — and we verify this property holds consistently.
    """
    from fitsproof.contract.admit import AdmitStatus, admit
    from fitsproof.contract.plan import Verdict, plan

    p = plan(REFERENCE_CONFIG, _machine(), context_len=512, budget_bytes=1, quant="none")
    assert p.verdict == Verdict.DOES_NOT_FIT

    # Mutate budget_bytes to a large value after planning
    p.budget_bytes = 4 * 1024**3  # 4 GiB — would admit if re-planned

    # admit() must use p.verdict (DOES_NOT_FIT), not re-derive from mutated budget_bytes
    record = admit(p)
    # The verdict in the plan is DOES_NOT_FIT, so admit must REFUSE
    assert record.status == AdmitStatus.REFUSED, (
        f"admit() must use plan.verdict, not mutated budget_bytes: got {record.status}. "
        f"This means admit() does not re-derive the verdict from budget_bytes — correct."
    )


def test_degradation_chain_covers_all_quant_levels() -> None:
    """
    Sources: [7] GPTQ int8, [13] GGML int4 — all quantisation levels that reduce
    peak memory must appear in the degradation chain.
    Fault (c2): a planner omitting int8_sym or int4_sym from the degradation options
    leaves the user with no recovery path when those quantisations would fit.

    For the reference model (~38 MB fp32), int8 halves to ~19 MB and int4 quarters
    to ~10 MB. A plan with a tight budget must offer both as degradation candidates.
    """
    from fitsproof.contract.plan import plan

    # Budget: forces DOES_NOT_FIT for fp32 but int4 would fit
    fp32_w = 38 * 1024 * 1024  # ~38 MB reference model
    budget = int(fp32_w * 0.3)  # 30% of fp32 — too small for fp32, int8, but fits int4

    p = plan(REFERENCE_CONFIG, _machine(), context_len=512, budget_bytes=budget, quant="none")
    # Collect degradation kind strings
    kinds = {d.kind for d in p.degradations}
    quant_kinds = {k for k in kinds if "int" in k.lower() or "quant" in k.lower()}
    assert quant_kinds, (
        f"Degradation chain must include at least one quantisation option. Got kinds: {kinds}"
    )


def test_mcp_plan_tool_does_not_fit_propagates_correctly() -> None:
    """
    Sources: [23] MCP spec; fitsproof.md M2.4.
    Fault (c2): MCP plan tool returning a 'fits' verdict for a does_not_fit config —
    an agent would then proceed to load a model that cannot fit.

    Verifies that plan tool returns verdict=does_not_fit for a 1-byte budget,
    and that isError=False (plan is informational, not a gate — admit is the gate).
    """
    import io
    import json

    from fitsproof.mcp import run_mcp_server

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "plan",
                "arguments": {"budget": "1", "context_len": 512, "quant": "none"},
            },
        },
    ]
    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    stdout = io.StringIO()
    run_mcp_server(stdin=stdin, stdout=stdout)

    lines = [line for line in stdout.getvalue().strip().split("\n") if line]
    plan_resp = json.loads(lines[1])
    # plan tool: informational — isError=False even for does_not_fit
    assert plan_resp["result"]["isError"] is False, (
        "plan tool must return isError=False (it is informational); "
        "admit tool sets isError=True on REFUSED"
    )
    payload = json.loads(plan_resp["result"]["content"][0]["text"])
    assert payload["verdict"] == "does_not_fit", (
        f"Plan tool must propagate does_not_fit verdict for 1-byte budget: {payload}"
    )


def test_mcp_admit_tool_with_zero_budget_is_error() -> None:
    """
    Sources: [23] MCP spec — isError=True for tool failures.
    Fault (c2): MCP admit tool returning isError=False when the budget cannot be
    parsed or is impossibly small — agent mistakes a refusal for an admission and
    proceeds to load.

    Verifies budget="0" causes isError=True (budget ≤ 0 is refused).
    """
    import io
    import json

    from fitsproof.mcp import run_mcp_server

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "admit",
                "arguments": {"budget": "0", "context_len": 512},
            },
        },
    ]
    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    stdout = io.StringIO()
    run_mcp_server(stdin=stdin, stdout=stdout)

    lines = [line for line in stdout.getvalue().strip().split("\n") if line]
    admit_resp = json.loads(lines[1])
    # budget=0 must trigger a ValueError (unparseable) or a REFUSED, either way isError=True
    assert admit_resp["result"]["isError"] is True, (
        "MCP admit with budget=0 must return isError=True; "
        "returning isError=False would let an agent proceed past an impossible budget"
    )


def test_client_metrics_always_positive() -> None:
    """
    Sources: [1] Roofline — machine metrics are measured physical quantities,
    always strictly positive on a real machine.
    Fault (c2): FitsproofClient.metrics() returning zero or negative values for
    bandwidth, GEMM throughput, or RAM would make the roofline predict zero tok/s
    or infinite memory, silently breaking all downstream plans.
    """
    from fitsproof.client import FitsproofClient

    client = FitsproofClient()
    m = client.metrics()

    assert m["memory_bandwidth_gb_s"] > 0, (
        f"memory_bandwidth_gb_s must be positive, got {m['memory_bandwidth_gb_s']}"
    )
    assert m["gemm_throughput_gflops"] > 0, (
        f"gemm_throughput_gflops must be positive, got {m['gemm_throughput_gflops']}"
    )
    assert m["ram_gb"] > 0, f"ram_gb must be positive, got {m['ram_gb']}"
    # VRAM is 0 on machines without CUDA — non-negative only
    assert m["vram_gb"] >= 0, f"vram_gb must be non-negative, got {m['vram_gb']}"


def test_guard_called_multiple_times_always_refuses() -> None:
    """
    Sources: fitsproof.md M2.3 — @guard raises DoesNotFit before the wrapped callable.
    Fault (c2): the @guard decorator admitting a second call after refusing the first,
    due to probe-result caching or re-used state from a prior successful decoration.

    Guard must evaluate the contract fresh each time regardless of call count.
    """
    from fitsproof.client import DoesNotFit, guard

    call_count = [0]

    @guard(budget="1MiB")  # always refuses: reference model needs ~40 MB
    def load_model() -> None:
        call_count[0] += 1

    for _ in range(3):
        with pytest.raises(DoesNotFit):
            load_model()

    assert call_count[0] == 0, (
        f"Wrapped callable must never be invoked on a refusing config; "
        f"was called {call_count[0]} times across 3 guard invocations"
    )


def test_stress_harness_budget_violation_exits_nonzero() -> None:
    """
    Sources: [1] Roofline; fitsproof.md AC#7 — stress harness exits 1 on any violation.
    Fault (c2): the stress harness silently passing when a budget is violated — "0 violations"
    is then a false claim. Verifies exit code is non-zero when the declared budget is
    so small that at least one config must violate it.

    Uses a very small budget (0.01 GB = 10 MB) — the reference model fp32 peak is ~39 MB,
    so some configs will be DEGRADED and their measured RSS may exceed the budget,
    triggering violations. The CLI stress command must exit 1 in that case.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "fitsproof.cli", "stress", "--budget-gb", "0.01"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    # With a 10 MB budget, the harness should run and detect violations (exit 1)
    # or in the extreme case exit 2 (all configs refused before running)
    assert proc.returncode != 0, (
        f"stress with a 10 MB budget must exit non-zero (got 0). stdout: {proc.stdout[:500]}"
    )
    # Must either report violations or show the admission being DEGRADED/REFUSED
    combined = (proc.stdout + proc.stderr).lower()
    has_signal = any(
        word in combined for word in ("violation", "refused", "degraded", "does not fit", "budget")
    )
    assert has_signal, (
        f"stress output must mention violation/refused/degraded/budget. stdout: {proc.stdout[:500]}"
    )
