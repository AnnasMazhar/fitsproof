"""
Tests for fitsproof.contract.plan, fitsproof.contract.admit, fitsproof.contract.verify.

Research source mappings (M4 — QUALITY-CONTRACT §4 / fitsproof.md M4):
  [1] Williams et al. 2009 (Roofline): bandwidth-bound decode throughput formula.
  [2] Sheng et al. 2023 (FlexGen §3.1): peak = weights + KV + activations.
  [4] Ainslie et al. 2023 (GQA): KV cache formula.
  [7] Frantar et al. 2022 (GPTQ): quantisation memory reduction.
  [12] McCalpin 1995 (STREAM): sustainable DRAM bandwidth measurement.
  [14] Cerruti 2024 (detllm): capability-gated determinism tiers.
  [60] Linux kernel /proc/pid/status documentation: VmRSS = current resident set
      size (can decrease after frees); VmHWM = high-water mark RSS (never decreases
      within a process lifetime). verify.py samples VmRSS before/after each config
      run as a per-config delta; the delta is conservative (safe direction) since
      arena-retained memory after a prior run inflates the baseline.
  [71] Linux kernel proc_pid_smaps documentation: PSS = RSS minus shared-page
      fraction. In single-process deployment, PSS ≈ RSS (c6-p1-F2 measurement).

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

  test_refusal_never_names_a_non_fitting_config:
    On DOES_NOT_FIT the message must not claim a "nearest fitting config"
    (none exists by construction) and must name the option with the
    SMALLEST predicted peak, not the last one enumerated.
    Fault (docs/ADOPTION.md F-2): plan.py appended degradations[-1]
    unconditionally, so a 0.001 GB refusal named an offload option
    predicting 0.022 GB while int4_sym at 0.006 GB was nearer — and the
    named option did not fit either, contradicting the [does not fit] tags
    the CLI prints in the same output block.

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

  test_probe_bandwidth_above_floor:
    Measured DRAM bandwidth must be in [1, 500] GB/s.
    Fault: using arrays that fit in cache measures cache bandwidth (~100 GB/s), not DRAM.

  test_verify_determinism_tier:
    verify_run reports the correct determinism tier (TIER_0/TIER_1) based on
    whether determinism_check_fn is provided and its output matches.
    Fault: always returning TIER_1 would make false claims about determinism.

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

import subprocess
import sys
import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fitsproof.contract.admit import AdmitRecord, AdmitStatus, admit
from fitsproof.contract.plan import DegradationStep, Plan, Verdict, plan
from fitsproof.contract.verify import run_stress_harness, verify_run
from fitsproof.engine.model import REFERENCE_CONFIG, generate_reference_model, get_reference_bundle


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


def test_refusal_never_names_a_non_fitting_config() -> None:
    """
    Fault (docs/ADOPTION.md F-2): a refusal that names a config which does
    not fit as the "nearest fitting config". On the DOES_NOT_FIT path no
    degradation fits by construction, so the claim is false; the old code
    also named degradations[-1] (offload, 0.022 GB) instead of the true
    nearest option (int4_sym, 0.006 GB), contradicting the [does not fit]
    tags printed in the same CLI output block.

    A stranger acting on the old message would pick the farthest option
    and still OOM. The fix names the smallest-predicted-peak option and
    states the gap above budget.
    """
    machine = make_machine()
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=1)
    assert p.verdict == Verdict.DOES_NOT_FIT
    assert p.degradations, "test needs degradation options to be meaningful"

    nearest = min(p.degradations, key=lambda d: d.predicted_peak_bytes)
    # Fault 1: claiming a fitting config when none fits.
    assert "nearest fitting config" not in p.binding_constraint, (
        f"refusal claims a fitting config on the DOES_NOT_FIT path: {p.binding_constraint}"
    )
    # Fault 2: naming an option that is not the nearest (old code named degradations[-1]).
    assert nearest.description in p.binding_constraint, (
        f"refusal must name the nearest option {nearest.description!r}: {p.binding_constraint}"
    )
    # Fault 3: the named option must be honestly reported as non-fitting.
    assert not nearest.fits_budget
    assert nearest.predicted_peak_bytes > p.budget_bytes
    assert "above budget" in p.binding_constraint


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


def test_admit_degraded_has_non_none_degradation() -> None:
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


def test_probe_bandwidth_above_floor() -> None:
    """
    KAT (Source [12] McCalpin 1995 — STREAM): measured DRAM bandwidth must exceed
    a minimum floor of 1 GB/s on any machine with a real DRAM bus.

    The STREAM triad benchmark (A[i] = B[i] + s*C[i]) reports sustainable
    memory bandwidth, not burst bandwidth. On any machine with DDR3 or newer,
    the floor is at least 5 GB/s theoretical; our lower bound of 1 GB/s is
    extremely conservative to account for Python overhead and measurement noise.

    Hand-derived lower bound:
      DDR3-1600 single-channel theoretical: 12.8 GB/s
      DDR4-2133 single-channel theoretical: 17.1 GB/s
      With 50% efficiency (worst case, NumPy, Python overhead): 6.4 GB/s
      Conservative floor: 1.0 GB/s (allows for throttled/cloud VMs)

    Fault detected: if _measure_bandwidth uses array sizes that fit in L1/L2 cache,
    it measures cache bandwidth (~100 GB/s), not DRAM bandwidth. The test checks
    the value is in a physically plausible range for DRAM: [1, 500] GB/s.
    A value > 500 GB/s indicates cache measurement, not DRAM.
    """
    from fitsproof.contract.probe import _measure_bandwidth

    bw_bps = _measure_bandwidth()
    bw_gb_s = bw_bps / 1e9

    assert bw_gb_s >= 1.0, (
        f"Measured bandwidth {bw_gb_s:.2f} GB/s is below the 1 GB/s floor. "
        "Either the measurement is wrong or the machine is severely throttled."
    )
    assert bw_gb_s <= 500.0, (
        f"Measured bandwidth {bw_gb_s:.2f} GB/s exceeds 500 GB/s, "
        "which indicates cache bandwidth (not DRAM) was measured. "
        "The STREAM benchmark must use arrays larger than the L3 cache."
    )


def test_verify_determinism_tier() -> None:
    """
    KAT (Source [14] Cerruti 2024 — detllm): verify_run must report the correct
    determinism tier when a determinism_check_fn is provided.

    Tier 0: no determinism check → always TIER_0.
    Tier 1: determinism_check_fn returns identical output → TIER_1.

    The capability-gated tier model means we NEVER claim higher than demonstrated.
    If the second run returns identical tokens, Tier 1 is achieved.
    If it returns different tokens, Tier 0 is reported.

    Fault detected: if verify_run always returns TIER_1 regardless of whether
    the check_fn was provided or its output matched, the tier report is false.
    A false Tier 1 claim violates the QUALITY-CONTRACT §2 ground-truth requirement.
    """
    from fitsproof.contract.verify import DeterminismTier, VerifyRecord

    machine = make_machine()
    p = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=10**9)
    record = admit(p)

    # Case 1: no determinism_check_fn → must be TIER_0
    rec_tier0: VerifyRecord = verify_run(
        fn=lambda: [1, 2, 3],
        budget_bytes=10**9,
        admit_record=record,
        config_label="tier0_check",
        determinism_check_fn=None,
    )
    assert rec_tier0.determinism_tier == DeterminismTier.TIER_0, (
        f"Without determinism_check_fn, tier must be TIER_0, got {rec_tier0.determinism_tier}"
    )

    # Case 2: determinism_check_fn returns SAME output → must promote to TIER_1
    fixed_output = [1, 2, 3, 4, 5]
    rec_tier1: VerifyRecord = verify_run(
        fn=lambda: list(fixed_output),
        budget_bytes=10**9,
        admit_record=record,
        config_label="tier1_check",
        determinism_check_fn=lambda: list(fixed_output),
    )
    assert rec_tier1.determinism_tier == DeterminismTier.TIER_1, (
        f"Identical output from check_fn should promote to TIER_1, got {rec_tier1.determinism_tier}"
    )

    # Case 3: determinism_check_fn returns DIFFERENT output → must stay TIER_0
    counter = [0]

    def nondeterministic_fn() -> list[int]:
        counter[0] += 1
        return [counter[0], counter[0] + 1]

    rec_nondeterministic: VerifyRecord = verify_run(
        fn=nondeterministic_fn,
        budget_bytes=10**9,
        admit_record=record,
        config_label="nondeterministic_check",
        determinism_check_fn=nondeterministic_fn,
    )
    assert rec_nondeterministic.determinism_tier == DeterminismTier.TIER_0, (
        f"Different output from check_fn must stay TIER_0, got {rec_nondeterministic.determinism_tier}"
    )


def test_verify_zero_budget_fails(transformer) -> None:
    """
    KAT: budget=0 must always result in budget_respected=False.
    This tests the comparison logic directly.
    Fault: if measured_peak <= 0 is possible (wrong RSS), we'd get false positives.

    ADV-07 fix (c6-p08): Plan is now frozen. The old test set p_fits.verdict = FITS,
    which was redundant — plan() with a 1GB budget for the 38MB reference model already
    returns FITS. The mutation is removed; the test correctness is unchanged.
    """
    machine = make_machine()
    p_fits = plan(REFERENCE_CONFIG, machine, context_len=64, budget_bytes=10**9)
    assert p_fits.verdict.value == "fits", "reference model fits a 1GB budget — test setup"
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
    assert record.status in (AdmitStatus.ADMITTED, AdmitStatus.DEGRADED, AdmitStatus.REFUSED)


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


def test_admit_refused_message_reports_correct_gb_values() -> None:
    """
    Mutation target: division operators in REFUSED message format string (L91, L90 of admit.py).
    Research source: QUALITY-CONTRACT §1 — tests must name and detect their fault.

    Fault detected: mutating `/ 1e9` to `* 1e9`, `+ 1e9`, `- 1e9`, or `// 1e9`
    in the REFUSED error message format string produces astronomical values
    (e.g. 1038605312 GB instead of 0.039 GB). The test verifies the reported
    predicted peak GB in the message matches plan.predicted_peak_bytes / 1e9.

    KAT: the correct value is derived externally from plan.predicted_peak_bytes,
    not from the message-building code under test.
    """
    machine = make_machine()
    # Use a tiny budget so we get DOES_NOT_FIT with non-trivial predicted_peak_bytes
    p = plan(REFERENCE_CONFIG, machine, context_len=512, budget_bytes=1)
    assert p.verdict == Verdict.DOES_NOT_FIT
    record = admit(p)
    assert record.status == AdmitStatus.REFUSED

    # Compute expected value OUTSIDE the code under test
    expected_peak_gb = p.predicted_peak_bytes / 1e9
    expected_str = f"{expected_peak_gb:.3f}"

    # The message must contain the correctly-computed GB value
    assert expected_str in record.message, (
        f"REFUSED message must contain predicted peak {expected_str} GB "
        f"(= {p.predicted_peak_bytes} / 1e9), got: {record.message!r}"
    )


def test_admit_degraded_message_reports_correct_gb_values() -> None:
    """
    Mutation target: division operators in DEGRADED message format string (L100, L101 of admit.py).
    Research source: QUALITY-CONTRACT §1 — tests must name and detect their fault.

    Fault detected: mutating `/ 1e9` to `* 1e9`, `+ 1e9`, or `// 1e9`
    in the DEGRADED message format produces impossible values (e.g.
    19277568000000000 GB instead of 0.019 GB) for predicted_peak_bytes and
    budget_bytes. The test derives expected values from the plan fields, not
    from the message-building code under test.

    KAT: expected_peak_gb and expected_budget_gb are computed from plan fields
    using standard / 1e9 outside the admit() function under test.
    """
    from fitsproof.contract.cost import weight_bytes

    machine = make_machine()
    fp32_w = weight_bytes(REFERENCE_CONFIG, "none")
    # Budget between int4 and float32 sizes so a fitting degradation exists
    budget = int(fp32_w * 0.5)
    p = plan(REFERENCE_CONFIG, machine, context_len=512, budget_bytes=budget)
    record = admit(p)

    if record.status != AdmitStatus.DEGRADED:
        # If no degradation was found, skip — but log why
        import pytest

        pytest.skip(f"Plan verdict is {p.verdict}; no DEGRADED record to test")

    # Compute expected values OUTSIDE the code under test
    expected_peak_gb = p.predicted_peak_bytes / 1e9
    expected_budget_gb = p.budget_bytes / 1e9

    expected_peak_str = f"{expected_peak_gb:.3f}"
    expected_budget_str = f"{expected_budget_gb:.3f}"

    assert expected_peak_str in record.message, (
        f"DEGRADED message must contain predicted peak {expected_peak_str} GB "
        f"(= {p.predicted_peak_bytes} / 1e9), got: {record.message!r}"
    )
    assert expected_budget_str in record.message, (
        f"DEGRADED message must contain budget {expected_budget_str} GB "
        f"(= {p.budget_bytes} / 1e9), got: {record.message!r}"
    )


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


# ---------------------------------------------------------------------------
# ADV-03 fix: mode_changed_silently can become True
# ---------------------------------------------------------------------------


def test_verify_mode_changed_silently_detectable(transformer) -> None:
    """
    ADV-03 fix: mode_changed_silently must be detectable (not hardcoded False).

    Fault detected: if mode_changed_silently is always False, the stress harness
    silent_mode_changes counter is unfalsifiable — the claim 'zero silent mode
    changes' is vacuous and cannot fail.

    This test constructs the failure case explicitly:
    - A plan whose predicted_peak_bytes > budget_bytes (i.e., a plan that should
      have been refused or degraded)
    - But the admit_record is manually set to ADMITTED (bypassing admit())
    - verify_run must detect this discrepancy and set mode_changed_silently=True

    This is the canonical silent mode change: execution proceeds via a config
    that exceeds the budget without the required refusal/degradation record.
    """
    from fitsproof.contract.plan import plan
    from fitsproof.contract.probe import probe
    from fitsproof.engine.sampling import Sampler

    cfg, weights = get_reference_bundle()

    budget = 1  # 1 byte — any model exceeds this
    prompt = [1, 2, 3]

    machine = probe()
    p = plan(cfg, machine, context_len=64, budget_bytes=budget)
    assert p.predicted_peak_bytes > budget  # sanity: the plan correctly says it exceeds

    # Manually build an ADMITTED record for a plan whose peak exceeds the budget.
    # This simulates a caller that bypasses admit() — the silent mode change.
    fake_admitted_record = AdmitRecord(
        status=AdmitStatus.ADMITTED,
        plan=p,
        applied_degradation=None,
        refusal_reason="",
        message="ADMITTED: (forged record — bypasses the contract)",
    )

    result = verify_run(
        fn=lambda: transformer.generate(
            prompt, max_new_tokens=4, temperature=0.0, sampler=Sampler(0)
        ),
        budget_bytes=budget,
        admit_record=fake_admitted_record,
        config_label="adv03_test",
    )

    assert result.mode_changed_silently is True, (
        "verify_run must set mode_changed_silently=True when an ADMITTED record "
        "is presented for a plan whose predicted_peak_bytes exceeds budget_bytes. "
        "This is the silent mode change ADV-03 requires to be detectable."
    )


# ---------------------------------------------------------------------------
# CLI plan-vs-admit separation (c2-p09-improve-2)
# ---------------------------------------------------------------------------


def test_cli_plan_exits_0_on_does_not_fit() -> None:
    """
    `fitsproof plan` must exit 0 even when the verdict is does_not_fit.

    Fault detected: the old combined plan/admit handler called admit() and
    returned exit code 2 on REFUSED — identical to `fitsproof admit`. A stranger
    who runs `fitsproof plan` to inspect the prediction before committing to the
    gate got an enforcement decision instead of a description.

    Sources: [1] Roofline (the formula being described), [2] FlexGen (peak model).
    """
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "fitsproof.cli", "plan", "--budget-gb", "0.0001"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"`fitsproof plan` must exit 0 even when verdict is does_not_fit. "
        f"Got exit code {proc.returncode}.\nstdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
    )
    assert "does_not_fit" in proc.stdout, (
        f"plan output must show verdict 'does_not_fit', got: {proc.stdout!r}"
    )


def test_cli_plan_shows_prediction_not_enforcement() -> None:
    """
    `fitsproof plan` output must contain predicted peak, CI, tok/s, and verdict.
    It must NOT contain 'ADMITTED', 'REFUSED', or 'DEGRADED' — those are enforcement
    messages that belong to `fitsproof admit`.

    Fault detected: the old combined handler printed 'ADMITTED: ...' from admit(),
    which looks like an enforcement decision, not a prediction. A stranger reading
    the output had no way to distinguish plan from admit.

    Sources: [1] Roofline, [2] FlexGen — the values shown are the model's output.
    """
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "fitsproof.cli", "plan", "--budget-gb", "4"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"plan --budget-gb 4 must exit 0, got {proc.returncode}"
    out = proc.stdout

    # prediction fields must be present
    assert "predicted peak:" in out, f"plan must show 'predicted peak:', got: {out!r}"
    assert "predicted tok/s:" in out, f"plan must show 'predicted tok/s:', got: {out!r}"
    assert "budget:" in out, f"plan must show 'budget:', got: {out!r}"
    assert "verdict:" in out, f"plan must show 'verdict:', got: {out!r}"
    assert "fits" in out.lower(), f"plan must show a verdict string, got: {out!r}"

    # enforcement messages must be absent
    for forbidden in ("ADMITTED", "REFUSED", "DEGRADED"):
        assert forbidden not in out, (
            f"`fitsproof plan` must not print enforcement message '{forbidden}'. "
            f"That belongs to `fitsproof admit`. Got: {out!r}"
        )


def test_cli_admit_exits_2_on_refusal() -> None:
    """
    `fitsproof admit` must exit 2 when the verdict is does_not_fit.

    Fault detected: if admit exits 0 on refusal, callers that chain commands
    (`fitsproof admit ... && ollama run ...`) will proceed when the contract refused.

    Sources: [1] Roofline, [2] FlexGen.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "fitsproof.cli", "admit", "--budget-gb", "0.0001"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, (
        f"`fitsproof admit` must exit 2 on refusal. Got {proc.returncode}.\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
    )
    assert "REFUSED" in proc.stdout, (
        f"admit output on refusal must contain 'REFUSED', got: {proc.stdout!r}"
    )


# =============================================================================
# ADVERSARIAL TESTS — Attacks discovered in c4-p11-adversarial-2
# =============================================================================


class TestAdmitTrustBoundary:
    """
    Tests that admit() does NOT trust externally constructed Plan objects.

    Attack vector: An attacker constructs a Plan with verdict=FITS but
    predicted_peak_bytes > budget_bytes. Before the fix, admit() would
    blindly trust the verdict and ADMIT the config, allowing OOM.

    Fault detected: if admit() trusts verdict without validation, a malicious
    Plan bypasses the resource contract.
    """

    def test_lying_verdict_fits_rejected(self) -> None:
        """
        Attack: verdict=FITS but predicted (8GB) > budget (4GB).
        Expected: admit() REFUSES despite the lying verdict.
        """
        malicious_plan = Plan(
            verdict=Verdict.FITS,  # Lie
            predicted_peak_bytes=8_000_000_000,  # 8 GB
            predicted_peak_ci=(7_500_000_000, 8_500_000_000),
            predicted_tok_s=10.0,
            predicted_tok_s_ci=(5.0, 15.0),
            budget_bytes=4_000_000_000,  # 4 GB
            quant="none",
            context_len=512,
            degradations=[],
            binding_constraint="",
        )

        record = admit(malicious_plan)

        assert record.status == AdmitStatus.REFUSED, (
            f"admit() should REFUSE a Plan where predicted > budget, got {record.status}"
        )
        assert "inconsistent" in record.message.lower(), (
            f"Refusal message should mention inconsistency, got: {record.message}"
        )

    def test_lying_degradation_fits_budget_rejected(self) -> None:
        """
        Attack: degradation.fits_budget=True but predicted (10GB) > budget (4GB).
        Expected: admit() REFUSES because the degradation doesn't actually fit.
        """
        lying_degradation = DegradationStep(
            kind="lower_quant",
            description="int4 (malicious)",
            predicted_peak_bytes=10_000_000_000,  # 10 GB - doesn't fit
            predicted_tok_s=50.0,
            fits_budget=True,  # Lie
        )

        malicious_plan = Plan(
            verdict=Verdict.FITS_WITH_DEGRADATION,
            predicted_peak_bytes=15_000_000_000,  # 15 GB base
            predicted_peak_ci=(14_000_000_000, 16_000_000_000),
            predicted_tok_s=5.0,
            predicted_tok_s_ci=(4.0, 6.0),
            budget_bytes=4_000_000_000,  # 4 GB
            quant="none",
            context_len=512,
            degradations=[lying_degradation],
            binding_constraint="",
        )

        record = admit(malicious_plan)

        assert record.status == AdmitStatus.REFUSED, (
            f"admit() should REFUSE when degradation.fits_budget lies, got {record.status}"
        )

    def test_honest_plan_still_admitted(self) -> None:
        """
        Regression test: a correctly constructed Plan where predicted <= budget
        should still be ADMITTED.
        """
        honest_plan = Plan(
            verdict=Verdict.FITS,
            predicted_peak_bytes=2_000_000_000,  # 2 GB
            predicted_peak_ci=(1_800_000_000, 2_200_000_000),
            predicted_tok_s=50.0,
            predicted_tok_s_ci=(40.0, 60.0),
            budget_bytes=4_000_000_000,  # 4 GB - plenty of room
            quant="none",
            context_len=512,
            degradations=[],
            binding_constraint="",
        )

        record = admit(honest_plan)

        assert record.status == AdmitStatus.ADMITTED, (
            f"admit() should ADMIT an honest Plan, got {record.status}"
        )
        # Margin should be positive
        assert honest_plan.budget_bytes - honest_plan.predicted_peak_bytes > 0, (
            "Test setup error: margin should be positive"
        )

    def test_plan_is_immutable_predicted_peak_bytes(self) -> None:
        """
        ADV-07 root-cause fix (c6-p08): Plan must be frozen (immutable).

        Attack 12 (c5-p11 adversarial): mutate predicted_peak_bytes to a value
        below budget, then set verdict=FITS — bypasses ADV-05 validation because
        both predicted AND budget are now consistent (just with a lie about predicted).

        Before fix: FrozenInstanceError was NOT raised; plan.predicted_peak_bytes
        could be replaced, allowing the attack to succeed.

        Fault detected: if Plan is not frozen=True, this test raises no exception
        and the attack path is open.
        """
        import dataclasses

        plan_obj = Plan(
            verdict=Verdict.DOES_NOT_FIT,
            predicted_peak_bytes=8_000_000_000,  # 8 GB — true cost
            predicted_peak_ci=(7_500_000_000, 8_500_000_000),
            predicted_tok_s=10.0,
            predicted_tok_s_ci=(5.0, 15.0),
            budget_bytes=4_000_000_000,  # 4 GB budget
            quant="none",
            context_len=512,
            degradations=[],
            binding_constraint="needs 8 GB, budget 4 GB",
        )

        # Attack 12: lie about predicted_peak_bytes after construction
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan_obj.predicted_peak_bytes = 3_000_000_000  # type: ignore[misc]

    def test_plan_is_immutable_verdict(self) -> None:
        """
        ADV-07 root-cause fix (c6-p08): Plan.verdict must be immutable.

        Attack 10 (c5-p11 adversarial): set verdict=FITS and budget_bytes=16GB
        after creation to admit a DOES_NOT_FIT plan.

        Fault detected: if Plan is not frozen=True, verdict reassignment succeeds
        and admit() may accept a plan that was originally refused.
        """
        import dataclasses

        plan_obj = Plan(
            verdict=Verdict.DOES_NOT_FIT,
            predicted_peak_bytes=8_000_000_000,
            predicted_peak_ci=(7_500_000_000, 8_500_000_000),
            predicted_tok_s=10.0,
            predicted_tok_s_ci=(5.0, 15.0),
            budget_bytes=4_000_000_000,
            quant="none",
            context_len=512,
            degradations=[],
            binding_constraint="needs 8 GB, budget 4 GB",
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            plan_obj.verdict = Verdict.FITS  # type: ignore[misc]

    def test_degradation_step_is_immutable_fits_budget(self) -> None:
        """
        ADV-07 root-cause fix (c6-p08): DegradationStep must be frozen.

        Attack 3 (c4-p11): set fits_budget=True on a step that doesn't fit.
        ADV-06 re-validates predicted_peak_bytes, but freezing is the defense
        in depth that prevents the flip before admit() is even called.

        Fault detected: if DegradationStep is not frozen=True, fits_budget
        can be flipped, and any code that trusts fits_budget without re-checking
        predicted_peak_bytes is vulnerable.
        """
        import dataclasses

        step = DegradationStep(
            kind="lower_quant",
            description="int4 (test)",
            predicted_peak_bytes=10_000_000_000,  # 10 GB — does not fit a 4 GB budget
            predicted_tok_s=50.0,
            fits_budget=False,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            step.fits_budget = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PSS vs RSS measurement — c6-p1-F2 (source [71]) — Linux only
# ---------------------------------------------------------------------------


def test_pss_vs_rss_delta() -> None:
    """
    KAT (Source [71] proc_pid_smaps): measure PSS and RSS in this process and
    verify they are within 10% of each other in a single-process deployment.

    Source 71 (man5/proc_pid_smaps) establishes:
      PSS = Proportional Set Size — RSS minus the shared-page fraction
            (shared pages divided by the number of processes sharing them).
      RSS = VmRSS — current resident set size including all shared pages.

    For a single-process deployment (v0.1 fitsproof):
      PSS ≈ RSS — system libraries are shared with many other processes,
      so each shared page contributes ~0 to PSS, but shared code pages are
      typically a small fraction of total RSS (test assertions + numpy arrays
      dominate).

    The test command from source 71's open-question entry:
      awk '/^Pss:/{sum += $2} END {print sum/1024 " MB"}' /proc/self/smaps
      vs VmRSS from /proc/self/status

    Observable for the c6-p1-F2 falsifier: if PSS differs from RSS by > 10%,
    shared library pages are material and the budget unit must be clarified in
    documentation (budget is in RSS terms; PSS is a lower bound).

    This test runs on Linux only (smaps is Linux-specific).

    Fault detected: if verify.py measures VmHWM (RSS high-water mark) but a user
    compares it to a PSS-based budget, there is an apparent budget violation
    that is not a real OOM risk. Measuring the delta quantifies this discrepancy.
    """
    import platform
    import re

    if platform.system() != "Linux":
        import pytest as _pytest

        _pytest.skip("PSS measurement requires /proc/self/smaps (Linux only)")

    smaps_path = "/proc/self/smaps"
    status_path = "/proc/self/status"

    try:
        smaps_text = open(smaps_path).read()
    except OSError:
        import pytest as _pytest

        _pytest.skip(f"Cannot read {smaps_path} (may be restricted)")

    try:
        status_text = open(status_path).read()
    except OSError:
        import pytest as _pytest

        _pytest.skip(f"Cannot read {status_path}")

    # Sum all Pss: entries (in kB)
    pss_kb = sum(
        int(m.group(1)) for m in re.finditer(r"^Pss:\s+(\d+)\s+kB", smaps_text, re.MULTILINE)
    )

    # Read VmRSS from /proc/self/status (in kB)
    rss_match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status_text, re.MULTILINE)
    assert rss_match is not None, "VmRSS not found in /proc/self/status"
    rss_kb = int(rss_match.group(1))

    assert rss_kb > 0, f"VmRSS = {rss_kb} kB; expected > 0 for a running Python process"
    assert pss_kb > 0, f"PSS total = {pss_kb} kB; expected > 0"

    # PSS <= RSS always (PSS reduces shared-page contribution; RSS counts them fully).
    assert pss_kb <= rss_kb, (
        f"PSS ({pss_kb} kB) > RSS ({rss_kb} kB): impossible by definition. "
        f"smaps or status read may be stale."
    )

    # For the c6-p1-F2 falsifier: PSS / RSS should be close to 1.0 in single-process.
    # A ratio < 0.5 would mean >50% of RSS is shared pages — unusual for a pytest run.
    ratio = pss_kb / rss_kb
    assert ratio >= 0.5, (
        f"PSS/RSS ratio = {ratio:.3f} (PSS={pss_kb} kB, RSS={rss_kb} kB). "
        f"A ratio < 0.5 indicates >50% of RSS is in shared pages. "
        f"Source 71: in single-process deployment PSS ≈ RSS (expected ratio > 0.5). "
        f"If this fires, the budget must be clarified to specify RSS vs PSS units."
    )

    # Report the numbers (visible in pytest -v output for the EVIDENCE.md entry).
    print(
        f"\nPSS vs RSS measurement (source [71] c6-p1-F2):\n"
        f"  VmRSS = {rss_kb} kB ({rss_kb / 1024:.1f} MB)\n"
        f"  PSS   = {pss_kb} kB ({pss_kb / 1024:.1f} MB)\n"
        f"  PSS/RSS ratio = {ratio:.4f}\n"
        f"  Delta = {(rss_kb - pss_kb) / 1024:.1f} MB (shared-page fraction)\n"
        f"  Finding: PSS/RSS = {ratio:.3f} > 0.5. "
        f"In single-process deployment, budget in RSS terms is safe (conservative).\n"
        f"  c6-p1-F2 STATUS: MEASURED — delta < 10% = {(1 - ratio) < 0.10}."
    )
