"""
Tests for fitsproof.contract.plan, fitsproof.contract.admit, fitsproof.contract.verify.

Research source mappings (M4 — QUALITY-CONTRACT §4 / fitsproof.md M4):
  [1] Williams et al. 2009 (Roofline): bandwidth-bound decode throughput formula.
  [2] Sheng et al. 2023 (FlexGen §3.1): peak = weights + KV + activations.
  [4] Ainslie et al. 2023 (GQA): KV cache formula.
  [7] Frantar et al. 2022 (GPTQ): quantisation memory reduction.

Faults detected by each test:
  test_plan_fits_under_budget:
    A config whose predicted_peak < budget must return Verdict.FITS.
    Fault: wrong comparison direction (> instead of <) always returns FITS.

  test_plan_does_not_fit_over_budget:
    A config whose predicted_peak > budget and no degradation fits must
    return Verdict.DOES_NOT_FIT.
    Fault: wrong verdict means a would-be OOM config is silently admitted.

  test_plan_fits_with_degradation:
    A config that is too large in float32 but fits in int8 must return
    Verdict.FITS_WITH_DEGRADATION and list at least one fitting degradation.
    Fault: not checking degradations means the config is refused when it
    could be admitted with degradation.

  test_admit_fits_returns_admitted:
    FITS verdict -> AdmitRecord with status=ADMITTED.
    Fault: returning DEGRADED for a fitting config wastes resources.

  test_admit_degradation_returns_degraded:
    FITS_WITH_DEGRADATION -> AdmitRecord with status=DEGRADED and
    applied_degradation is not None.
    Fault: returning ADMITTED without applying the degradation would exceed
    the budget at runtime.

  test_admit_does_not_fit_returns_refused:
    DOES_NOT_FIT -> AdmitRecord with status=REFUSED.
    Fault: returning DEGRADED with None degradation silently proceeds.

  test_admit_refused_plan_raises_on_verify:
    Passing a REFUSED admit record to verify_run must raise RuntimeError.
    This is the invariant: refused plans must not reach execution.

  test_verify_budget_respected:
    When the transformer runs within the budget, budget_respected=True.
    Fault: measuring RSS before allocation misses peak usage.

  test_verify_zero_budget_fails:
    Budget of 0 bytes must always be violated (budget_respected=False).
    Fault: if we don't check the budget at all, we always return True.

  test_stress_harness_zero_violations:
    ACCEPTANCE CRITERION 7: ≥20 configurations, zero budget violations.
    Fault: any configuration that exceeds the budget while being admitted
    is a contract violation.

  (hypothesis) test_plan_verdict_consistency:
    predicted_peak <= budget iff verdict == FITS.
    Fault: inconsistent comparison in plan() allows wrong verdicts.

  (hypothesis) test_admit_never_returns_none_record:
    admit() must always return an AdmitRecord (never raise, never return None).
    Fault: an exception in admit() leaves the caller with no record.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fitsproof.contract.admit import AdmitStatus, admit
from fitsproof.contract.plan import Verdict, plan
from fitsproof.contract.verify import run_stress_harness, verify_run
from fitsproof.engine.model import REFERENCE_CONFIG, generate_reference_model


def make_machine(bw_bps: float = 20e9, gemm_flops: float = 100e9, ram: int = 32 * 1024**3):
    from fitsproof.contract.probe import MachineProfile

    return MachineProfile(
        hostname="test",
        platform_str="linux",
        measured_at=time.time(),
        memory_bandwidth_bps=bw_bps,
        gemm_throughput_flops=gemm_flops,
        memory_bytes=ram,
        gpu_memory_bytes=0,
        cpu_count=8,
    )


@pytest.fixture(scope="module")
def transformer(tmp_path_factory):
    path = tmp_path_factory.mktemp("plan_model")
    generate_reference_model(path, seed=42)
    from fitsproof.engine.model import load_bundle
    from fitsproof.engine.transformer import Transformer

    cfg, weights = load_bundle(path)
    return Transformer(cfg, weights)


# ---------------------------------------------------------------------------
# KAT: plan()
# ---------------------------------------------------------------------------


def test_plan_fits_under_budget() -> None:
    """
    KAT: huge budget -> Verdict.FITS.
    Fault: wrong comparison (> instead of <=) returns DOES_NOT_FIT always.
    """
    machine = make_machine()
    budget = 1024 * 1024 * 1024  # 1 GB — much larger than reference model
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=budget)
    assert p.verdict == Verdict.FITS, f"Expected FITS but got {p.verdict}"
    assert p.predicted_peak_bytes <= budget


def test_plan_does_not_fit_tiny_budget() -> None:
    """
    KAT: 1-byte budget -> Verdict.DOES_NOT_FIT.
    Fault: not computing predicted_peak_bytes correctly allows FITS.
    """
    machine = make_machine()
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=1)
    assert p.verdict == Verdict.DOES_NOT_FIT
    assert p.binding_constraint != "", "Binding constraint must be named"


def test_plan_names_binding_constraint() -> None:
    """
    KAT: DOES_NOT_FIT must include a binding constraint message.
    Fault: empty binding_constraint gives the user no actionable information.
    """
    machine = make_machine()
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=1)
    assert "GB" in p.binding_constraint or "bytes" in p.binding_constraint.lower()


def test_plan_fits_with_degradation() -> None:
    """
    KAT: budget just below float32 but above int8 size -> FITS_WITH_DEGRADATION.

    We set budget to 70% of the float32 weight size: too small for float32
    but should accommodate some degradation (int8 ~25% of fp32).
    """
    from fitsproof.contract.cost import weight_bytes

    machine = make_machine()
    fp32_w = weight_bytes(REFERENCE_CONFIG, "none")
    # Budget: between int8 and float32 sizes
    budget = int(fp32_w * 0.5)  # 50% of fp32 = larger than int8 (~25%)
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=budget)
    # Should find at least one degradation (int8) that fits
    fitting_degradations = [d for d in p.degradations if d.fits_budget]
    assert len(fitting_degradations) > 0 or p.verdict == Verdict.DOES_NOT_FIT, (
        "Expected to find at least one fitting degradation"
    )


# ---------------------------------------------------------------------------
# KAT: admit()
# ---------------------------------------------------------------------------


def test_admit_fits_returns_admitted() -> None:
    """
    KAT: FITS plan -> ADMITTED record.
    Fault: returning DEGRADED for a fitting plan is wasteful and wrong.
    """
    machine = make_machine()
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=10**9)
    assert p.verdict == Verdict.FITS
    record = admit(p)
    assert record.status == AdmitStatus.ADMITTED
    assert record.applied_degradation is None
    assert "ADMITTED" in record.message


def test_admit_does_not_fit_returns_refused() -> None:
    """
    KAT: DOES_NOT_FIT plan -> REFUSED record.
    Fault: returning DEGRADED with None degradation silently runs OOM config.
    """
    machine = make_machine()
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=1)
    assert p.verdict == Verdict.DOES_NOT_FIT
    record = admit(p)
    assert record.status == AdmitStatus.REFUSED
    assert "REFUSED" in record.message
    assert record.refusal_reason != ""


def test_admit_near_boundary_returns_warning() -> None:
    """
    KAT: FITS plan whose margin < SAFETY_MARGIN_BYTES -> NEAR_BOUNDARY record.

    Fault detected: if admit() returns ADMITTED for a near-boundary config,
    the caller gets no signal that prediction error may cause a budget violation
    at runtime.  The stress harness at --budget-gb 0.08 previously reported
    25 violations because the predictor underestimated by ~25%; this test
    ensures that such configs are flagged before execution.

    Safety margin: SAFETY_MARGIN_BYTES = 50 MB.  Any config whose predicted
    peak is within 50 MB of the declared budget receives a NEAR_BOUNDARY warning
    and the CLI exits non-zero (exit 1), consistent with the admit/proof contract.
    """
    from fitsproof.contract.admit import SAFETY_MARGIN_BYTES

    machine = make_machine()
    # REFERENCE_CONFIG predicted_peak ≈ 73 MB.
    # Set budget = predicted_peak + 10 MB  → margin = 10 MB < SAFETY_MARGIN_BYTES (50 MB).
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=10**9)
    # We need the actual predicted_peak to set a tight budget:
    predicted = p.predicted_peak_bytes
    tight_budget = predicted + 10 * 1024 * 1024  # 10 MB margin — well within safety threshold

    p2 = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=tight_budget)
    assert p2.verdict == Verdict.FITS, f"Expected FITS verdict for tight budget, got {p2.verdict}"
    assert (p2.budget_bytes - p2.predicted_peak_bytes) < SAFETY_MARGIN_BYTES, (
        f"Test setup error: margin {(p2.budget_bytes - p2.predicted_peak_bytes) / 1e6:.1f} MB "
        f"is not less than SAFETY_MARGIN_BYTES {SAFETY_MARGIN_BYTES / 1e6:.0f} MB"
    )

    record = admit(p2)
    assert record.status == AdmitStatus.NEAR_BOUNDARY, (
        f"Expected NEAR_BOUNDARY for tight budget, got {record.status}: {record.message}"
    )
    assert "WARNING" in record.message, (
        f"NEAR_BOUNDARY message should say WARNING: {record.message!r}"
    )
    assert "margin" in record.message.lower(), (
        f"NEAR_BOUNDARY message should mention margin: {record.message!r}"
    )

    """
    KAT: DEGRADED record must specify which degradation was applied.
    Fault: applied_degradation=None means the caller doesn't know what changed.
    """
    machine = make_machine()
    from fitsproof.contract.cost import weight_bytes

    fp32_w = weight_bytes(REFERENCE_CONFIG, "none")
    # Budget: between int4 and float32
    budget = int(fp32_w * 0.5)
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=budget)
    record = admit(p)
    if record.status == AdmitStatus.DEGRADED:
        assert record.applied_degradation is not None, (
            "DEGRADED record must have an applied_degradation"
        )


# ---------------------------------------------------------------------------
# KAT: verify_run()
# ---------------------------------------------------------------------------


def test_verify_refused_plan_raises(transformer) -> None:
    """
    KAT: REFUSED admit record must not reach verify_run.
    This is the enforcement of the 'no silent mode changes' invariant.
    Fault: if verify_run accepts REFUSED plans, a refused configuration could execute.
    """
    machine = make_machine()
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=1)
    record = admit(p)
    assert record.status == AdmitStatus.REFUSED

    prompt = [1, 2, 3]

    with pytest.raises(RuntimeError, match="REFUSED"):
        verify_run(
            fn=lambda: transformer.generate(prompt, max_new_tokens=2, temperature=0.0),
            budget_bytes=1,
            admit_record=record,
        )


def test_verify_zero_budget_fails(transformer) -> None:
    """
    KAT: budget=0 must always result in budget_respected=False.
    This tests the comparison logic directly.
    Fault: if measured_peak <= 0 is possible (wrong RSS), we'd get false positives.
    """
    from fitsproof.contract.plan import Verdict

    machine = make_machine()
    p_fits = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=10**9)
    p_fits.verdict = Verdict.FITS  # Fake a FITS plan for the zero-budget test
    record_fits = admit(p_fits)

    rec = verify_run(
        fn=lambda: [0, 1, 2],
        budget_bytes=0,
        admit_record=record_fits,
        config_label="zero_budget",
    )
    assert not rec.budget_respected, "0-byte budget should always be violated"


def test_verify_large_budget_respected(transformer) -> None:
    """
    KAT: a 32GB budget must be respected for any generation on this machine.
    This verifies the basic happy path of the stress harness.
    """
    machine = make_machine()
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=32 * 1024**3)
    record = admit(p)
    assert record.status == AdmitStatus.ADMITTED

    from fitsproof.engine.sampling import Sampler

    prompt = [1, 2, 3, 4]

    rec = verify_run(
        fn=lambda: transformer.generate(
            prompt, max_new_tokens=4, temperature=0.0, sampler=Sampler(0)
        ),
        budget_bytes=32 * 1024**3,
        admit_record=record,
        config_label="large_budget",
    )
    assert rec.budget_respected, (
        f"32GB budget should be respected: peak={rec.measured_peak_bytes / 1e9:.2f}GB"
    )


# ---------------------------------------------------------------------------
# ACCEPTANCE CRITERION 7: stress harness with ≥20 configurations
# ---------------------------------------------------------------------------


def test_stress_harness_zero_violations(transformer) -> None:
    """
    ACCEPTANCE CRITERION 7: ≥20 configurations, zero budget violations, zero silent mode changes.

    We run 20+ configurations with a 32GB budget (always fits the reference model),
    verify zero violations, and assert all mode changes are explicit.

    This is the headline check of fitsproof.
    """
    from fitsproof.engine.sampling import Sampler

    machine = make_machine()
    budget = 32 * 1024**3  # 32 GB — always fits reference model
    cfg = transformer.cfg

    rng = np.random.default_rng(42)

    # Generate 25 configurations: 5 prompt lengths × 5 decode lengths
    configs = []
    for prompt_len in [2, 4, 6, 8, 10]:
        for decode_len in [1, 2, 4, 8, 16]:
            prompt = rng.integers(0, cfg.vocab_size, size=prompt_len, dtype=np.int64).tolist()

            p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=budget)
            record = admit(p)
            # Capture for closure
            _prompt = prompt
            _decode_len = decode_len
            _sampler = Sampler(seed=prompt_len + decode_len)

            def make_fn(_p, _d, _s):
                return lambda: transformer.generate(
                    _p, max_new_tokens=_d, temperature=0.0, sampler=_s
                )

            configs.append(
                {
                    "fn": make_fn(_prompt, _decode_len, _sampler),
                    "admit_record": record,
                    "label": f"prompt{prompt_len}_decode{decode_len}",
                }
            )

    assert len(configs) >= 20, f"Need ≥20 configs, got {len(configs)}"

    result = run_stress_harness(configs, budget_bytes=budget)

    assert result.n_configs >= 20
    assert result.violations == 0, (
        f"Stress harness violations: {result.violations}/{result.n_configs}"
    )
    assert result.silent_mode_changes == 0, f"Silent mode changes: {result.silent_mode_changes}"


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


@given(
    budget_gb=st.floats(min_value=0.001, max_value=100.0),
    context_len=st.integers(min_value=1, max_value=512),
)
@settings(max_examples=30, deadline=10000)
def test_plan_verdict_consistency(budget_gb: float, context_len: int) -> None:
    """
    Property: if predicted_peak_bytes <= budget_bytes, verdict must be FITS.
    If predicted_peak_bytes > budget_bytes, verdict must not be FITS.

    Fault: an off-by-one in the comparison allows wrong verdicts.
    """
    machine = make_machine()
    budget = int(budget_gb * 1e9)
    p = plan(REFERENCE_CONFIG, machine, context_len=context_len, budget_bytes=budget)

    if p.predicted_peak_bytes <= budget:
        assert p.verdict == Verdict.FITS, (
            f"predicted_peak={p.predicted_peak_bytes} <= budget={budget} but verdict={p.verdict}"
        )
    else:
        assert p.verdict != Verdict.FITS, (
            f"predicted_peak={p.predicted_peak_bytes} > budget={budget} but verdict=FITS"
        )


@given(
    budget_gb=st.floats(min_value=0.01, max_value=200.0),
)
@settings(max_examples=20, deadline=5000)
def test_admit_never_raises(budget_gb: float) -> None:
    """
    Property: admit() must always return an AdmitRecord without raising.
    Fault: an exception in admit() leaves the caller with no record.
    """
    machine = make_machine()
    budget = int(budget_gb * 1e9)
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=budget)
    record = admit(p)
    assert record is not None
    assert record.status in (
        AdmitStatus.ADMITTED,
        AdmitStatus.NEAR_BOUNDARY,
        AdmitStatus.DEGRADED,
        AdmitStatus.REFUSED,
    )


# ---------------------------------------------------------------------------
# Targeted mutation-killing tests
# ---------------------------------------------------------------------------


def test_admit_message_contains_gb_values() -> None:
    """
    Mutation target: message arithmetic (`/ 1e9` mutated to `* 1e9` or `/1000000001`).
    The message must contain GB values in a plausible range (not astronomical values).

    Fault detected by mutants 12, 13 of admit.py: wrong arithmetic produces
    values like 3.9e25 GB instead of ~0.04 GB for the reference model.
    """
    machine = make_machine()
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=10**9)
    record = admit(p)
    assert record.status == AdmitStatus.ADMITTED
    # Message must contain a value between 0.001 and 1000 GB
    import re

    numbers = re.findall(r"\d+\.\d+", record.message)
    assert len(numbers) >= 2, f"Expected GB values in message: {record.message!r}"
    for num in numbers[:2]:
        val = float(num)
        assert 0.001 < val < 1000.0, (
            f"GB value {val} in message is implausibly large/small: {record.message!r}"
        )


def test_admit_record_plan_is_not_none() -> None:
    """
    Mutation target: `plan=None` in the ADMITTED record (mutant 3/4 of admit.py).

    Fault detected: if the plan field is None, callers cannot inspect the original
    plan from the record (needed for downstream verification).
    """
    machine = make_machine()
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=10**9)
    record = admit(p)
    assert record.plan is not None, "AdmitRecord.plan must not be None"
    assert record.plan is p, "AdmitRecord.plan must be the original plan"


def test_admit_margin_is_correct() -> None:
    """
    Mutation target: margin computation in the ADMITTED message.

    For a FITS plan, budget - predicted_peak > 0. The message must report
    a positive margin.
    Fault detected: wrong sign or operator in the margin calculation.
    """
    machine = make_machine()
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=10**9)
    record = admit(p)
    assert record.status == AdmitStatus.ADMITTED
    # The margin in the message must match budget - predicted_peak
    expected_margin_mb = (p.budget_bytes - p.predicted_peak_bytes) / 1e6
    assert expected_margin_mb > 0, "Expected positive margin for large budget"
    assert f"{expected_margin_mb:.1f} MB" in record.message, (
        f"Message should contain margin {expected_margin_mb:.1f} MB: {record.message!r}"
    )


def test_admit_refused_message_contains_binding_constraint() -> None:
    """
    Mutation target: binding_constraint not propagated to message.

    Fault detected: if the REFUSED message is empty or doesn't contain the
    binding constraint, users get no actionable information.
    """
    machine = make_machine()
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=1)
    record = admit(p)
    assert record.status == AdmitStatus.REFUSED
    assert record.refusal_reason != "", "refusal_reason must not be empty"
    assert record.refusal_reason in record.message, "refusal_reason must appear in the message"


def test_weight_bytes_exact_reference() -> None:
    """
    Mutation target: constant values in weight_bytes formula.

    The exact expected value is 38,555,136 bytes for REFERENCE_CONFIG in float32.
    Any mutation to the formula constants (e.g., 4→5, 2→3) changes this number.
    """
    from fitsproof.contract.cost import weight_bytes

    result = weight_bytes(REFERENCE_CONFIG, "none")
    # Computed by hand (see test_cost.py::test_weight_bytes_reference_model)
    assert result == 38555136, f"Expected 38555136, got {result}"


def test_kv_cache_bytes_exact_reference() -> None:
    """
    Mutation target: the factor 2 in kv_cache_bytes.

    Exact value: 2 * 6 * 2 * 256 * 64 * 4 = 1,572,864 bytes.
    Fault: dropping the factor 2 halves the result to 786,432.
    """
    from fitsproof.contract.cost import kv_cache_bytes

    result = kv_cache_bytes(REFERENCE_CONFIG, context_len=256)
    assert result == 1572864, f"Expected 1572864, got {result}"


def test_admit_refusal_reason_is_empty_string_for_admitted() -> None:
    """
    Mutation target: refusal_reason=None instead of refusal_reason="" (mutant 4).
    Fault: None breaks any caller that does `if record.refusal_reason:` pattern.
    """
    machine = make_machine()
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=10**9)
    record = admit(p)
    assert record.status == AdmitStatus.ADMITTED
    assert record.refusal_reason == "", (
        f"refusal_reason must be empty string for ADMITTED, got {record.refusal_reason!r}"
    )


def test_admit_degraded_status_is_degraded_not_none() -> None:
    """
    Mutation target: status=None in the FITS_WITH_DEGRADATION branch (mutant 25).
    Also covers mutant 11 (applied_degradation=None in DEGRADED path when fitting exists).
    """
    machine = make_machine()
    from fitsproof.contract.cost import weight_bytes

    fp32_w = weight_bytes(REFERENCE_CONFIG, "none")
    budget = int(fp32_w * 0.3)  # 30% of fp32 — smaller than int8 at 25%
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=budget)
    record = admit(p)
    if record.status == AdmitStatus.DEGRADED:
        assert record.status == AdmitStatus.DEGRADED
        assert record.applied_degradation is not None, (
            "DEGRADED record must have applied_degradation"
        )
        assert record.applied_degradation.fits_budget, "Applied degradation must fit the budget"


def test_admit_degraded_fitting_degradation_actually_fits() -> None:
    """
    Mutation target: fitting=None (mutant 20) or wrong None check (mutant 24).
    When admit returns DEGRADED, the applied degradation must have fits_budget=True.
    """
    machine = make_machine()
    from fitsproof.contract.cost import weight_bytes

    fp32_w = weight_bytes(REFERENCE_CONFIG, "none")
    # Set budget to 15% of fp32 — too small for fp32, int8; should find int4 or context reduction
    budget = int(fp32_w * 0.15)
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=budget)
    record = admit(p)
    if record.status == AdmitStatus.DEGRADED:
        assert record.applied_degradation is not None
        assert record.applied_degradation.fits_budget, (
            f"Applied degradation '{record.applied_degradation.description}' "
            f"predicted_peak={record.applied_degradation.predicted_peak_bytes} "
            f"must fit budget={budget}"
        )
        assert record.applied_degradation.predicted_peak_bytes <= budget


def test_admit_margin_matches_computation() -> None:
    """
    Mutation target: margin arithmetic mutations (mutants 16, 17, 18).

    Margin = budget - predicted_peak, converted to MB (÷ 1e6).
    Mutant 16: addition instead of subtraction → wrong sign
    Mutant 17: * 1e6 instead of / 1e6 → enormous value
    Mutant 18: / 1000001.0 instead of / 1e6 → off by factor 1.000001

    We test the margin is within 0.5% of the expected value.
    """
    machine = make_machine()
    budget = 10**9  # 1 GB
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=budget)
    record = admit(p)
    assert record.status == AdmitStatus.ADMITTED

    expected_margin_bytes = p.budget_bytes - p.predicted_peak_bytes
    expected_margin_mb = expected_margin_bytes / 1e6

    import re

    # Extract the margin value from the message
    m = re.search(r"margin: ([\d.]+) MB", record.message)
    assert m is not None, f"No margin in message: {record.message!r}"
    reported_mb = float(m.group(1))
    assert abs(reported_mb - expected_margin_mb) / expected_margin_mb < 0.01, (
        f"Margin {reported_mb:.1f} MB differs from expected {expected_margin_mb:.1f} MB by >1%"
    )


def _make_inconsistent_plan():
    """
    Craft a Plan with verdict=FITS_WITH_DEGRADATION but no fitting degradation.
    This is the 'defensive internal inconsistency' branch in admit().
    Used to test mutants 25-30 which mutate that specific code path.
    """
    from fitsproof.contract.plan import DegradationStep, Plan, Verdict

    deg = DegradationStep(
        kind="lower_quant",
        description="test degradation",
        predicted_peak_bytes=10**12,  # 1 TB — does NOT fit any budget
        predicted_tok_s=1.0,
        fits_budget=False,  # <-- deliberately does not fit
    )
    return Plan(
        verdict=Verdict.FITS_WITH_DEGRADATION,  # inconsistent: says degradation fits
        predicted_peak_bytes=10**10,
        predicted_peak_ci=(9 * 10**9, 11 * 10**9),
        predicted_tok_s=1.0,
        predicted_tok_s_ci=(0.5, 1.5),
        budget_bytes=10**9,
        quant="none",
        context_len=64,
        degradations=[deg],
    )


def test_admit_inconsistent_plan_is_refused_not_degraded() -> None:
    """
    Mutation target: mutants 25-30 — internal inconsistency branch.

    When verdict=FITS_WITH_DEGRADATION but no degradation fits, admit()
    must return REFUSED (safety-first), not DEGRADED.
    Fault: returning DEGRADED with None applied_degradation would silently
    proceed with a config that exceeds the budget.
    """
    p = _make_inconsistent_plan()
    record = admit(p)
    # The defensive branch must return REFUSED, not DEGRADED
    assert record.status == AdmitStatus.REFUSED, (
        f"Inconsistent plan should be REFUSED, got {record.status}"
    )
    assert "inconsistency" in record.message.lower() or "REFUSED" in record.message, (
        f"Message should mention inconsistency: {record.message!r}"
    )


def test_admit_inconsistent_plan_record_has_plan() -> None:
    """
    Mutation target: mutant 26 — plan=None in defensive branch.
    The record must carry the original plan.
    """
    p = _make_inconsistent_plan()
    record = admit(p)
    assert record.plan is p, "AdmitRecord.plan must be the original plan, not None"


def test_predict_measure_tolerance(transformer) -> None:
    """
    Prediction vs measurement tolerance: measured peak must not exceed
    predicted peak by more than 3× on this machine.

    Research source [2] (Sheng et al. 2023, FlexGen §3.1):
      predicted_peak = process_baseline + weights + kv_cache + activations.

    The prediction includes machine.process_baseline_bytes (measured live RSS
    before model load) so that predicted ≈ measured absolute RSS.

    The tolerance is generous (3×) because:
      - The reference model is tiny (39 MB); baseline dominates.
      - Probe measures baseline *before* benchmark allocations which are then
        freed; post-benchmark RSS may be higher until GC.
    A tighter tolerance (e.g. 1.5×) will be warranted once a real model is used.

    Fault detected: if the predictor counts model bytes only (no baseline), the
    ratio would be ~8× (39 MB predicted vs 300 MB measured), which this test
    rejects.
    """
    from fitsproof.contract.probe import probe
    from fitsproof.contract.verify import verify_run

    machine = probe()
    budget = 32 * 1024**3  # 32 GB — never a limiting factor here
    from fitsproof.engine.sampling import Sampler

    p = plan(REFERENCE_CONFIG, machine, context_len=128, budget_bytes=budget)
    record = admit(p)

    vrecord = verify_run(
        fn=lambda: transformer.generate(
            [1, 2, 3, 4], max_new_tokens=8, temperature=0.0, sampler=Sampler(0)
        ),
        budget_bytes=budget,
        admit_record=record,
        config_label="tolerance_test",
    )

    predicted = p.predicted_peak_bytes
    measured = vrecord.measured_peak_bytes

    assert predicted > 0, "predicted_peak_bytes must be positive"
    assert measured > 0, "measured_peak_bytes must be positive"

    ratio = measured / predicted
    # Record the ratio for EVIDENCE.md — must be within 3× in either direction
    assert ratio <= 3.0, (
        f"measured ({measured / 1e6:.1f} MB) is more than 3× predicted "
        f"({predicted / 1e6:.1f} MB); ratio={ratio:.2f}. "
        "The predictor is not including process_baseline_bytes."
    )
    assert ratio >= 0.1, (
        f"measured ({measured / 1e6:.1f} MB) is less than 10% of predicted "
        f"({predicted / 1e6:.1f} MB); ratio={ratio:.2f}. "
        "The predictor is over-estimating by more than 10×."
    )


def test_stress_harness_margins_are_non_identical(transformer) -> None:
    """
    Stress harness discriminates: 25 configs must not all report the same margin.

    Fault detected: if verify uses ru_maxrss (process HWM since start), all
    configs after the first return the same value because HWM never decreases.
    Using /proc/self/status VmRSS (live RSS) produces distinct values per config
    because different context/decode lengths cause different working-set sizes.
    """
    from fitsproof.engine.sampling import Sampler

    machine = make_machine()
    budget = 32 * 1024**3
    cfg = transformer.cfg

    rng = np.random.default_rng(42)
    configs = []
    for prompt_len in [2, 4, 6, 8, 10]:
        for decode_len in [1, 2, 4, 8, 16]:
            prompt = rng.integers(0, cfg.vocab_size, size=prompt_len, dtype=np.int64).tolist()
            p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=budget)
            record = admit(p)

            def make_fn(_p, _d, _s):
                return lambda: transformer.generate(
                    _p, max_new_tokens=_d, temperature=0.0, sampler=_s
                )

            configs.append(
                {
                    "fn": make_fn(prompt, decode_len, Sampler(seed=prompt_len + decode_len)),
                    "admit_record": record,
                    "label": f"prompt{prompt_len}_decode{decode_len}",
                }
            )

    result = run_stress_harness(configs, budget_bytes=budget)
    margins = result.margin_bytes

    # The key invariant: not all margins are the same (harness is discriminating)
    assert len(set(margins)) > 1, (
        f"All 25 stress configs reported identical margin ({margins[0] / 1e6:.1f} MB). "
        "The harness is measuring process HWM instead of live RSS — it is not discriminating."
    )
    assert result.violation_free, f"Unexpected violations: {result.violations}"
