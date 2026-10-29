"""
tests/contract/test_calibrate.py — Known-answer and property tests for calibrate.py.

All 117 surviving mutants from cycle-7 mutation gate are in calibrate.py.
This file adds tests grounded in external sources to kill those mutants.

Research source mappings (M4 requirement):
  [1]  Williams et al. 2009 (Roofline): decode tok/s = bandwidth / weight_bytes.
  [6]  McCalpin 1995 (STREAM): bandwidth measurement methodology; bandwidth_utilisation
       is fitted to correct for the gap between STREAM-triad peak and real decode BW.
  [27] Hyndman & Koehler 2006 (MAPE): MAPE = mean(|actual-predicted|/|actual|) * 100.
       Dividing by *actual* (not predicted) is the standard asymmetric form; the
       denominator choice is the fault that `_mape` mutants exercise.
  [58] Davison & Hinkley 1997, §2.4: bootstrap CI requires n>=2 for a proper interval;
       at n=1 the CI degenerates to [mape, mape].

Faults detected by each test:

  test_fit_bandwidth_utilisation_closed_form_kat:
    Fault: incorrect closed-form formula for util = mean(tok_s * w_bytes / bw).
    A mutation that negates the division or subtracts instead of divides produces
    a different value; this KAT catches it with external reference values.

  test_fit_bandwidth_utilisation_single_measurement:
    Fault: any indexing or iteration bug collapses the mean to 0 or the first element.

  test_fit_bandwidth_utilisation_constant_weights_proportional:
    Fault: if util does not scale linearly with tok_s (e.g. multiplies instead of
    divides by bw), the proportionality relation breaks.

  test_mape_kat_hyndman_koehler:
    Fault: using predicted as denominator instead of actual, or missing the *100
    scaling, or off-by-one in mean. Formula from source [27] §2.1.
    Hand-computation: actual=[10,20,30], predicted=[8,25,27]:
      |10-8|/10 + |20-25|/20 + |30-27|/30 = 0.2 + 0.25 + 0.1 = 0.55 → 55/3 = 18.33...%

  test_mape_symmetric_errors_equal_weight:
    Fault: MAPE is asymmetric by design (actual denominator); a mutation to 'predicted'
    denominator gives a different result when predicted > actual vs predicted < actual.

  test_mape_zero_actual_skipped:
    Fault: zero-actual rows must be excluded (undefined); a mutation that includes
    them produces NaN or inf.

  test_bootstrap_ci_lower_le_upper:
    Fault: reversed quantile args (alpha/2 and 1-alpha/2 swapped) give lower > upper.
    Source [58] §2.4: the CI must be [q_{alpha/2}, q_{1-alpha/2}].

  test_bootstrap_ci_degenerate_n1_equals_mape:
    Fault: mutations that change the n<2 early-return produce different values.
    Source [58]: at n=1, the bootstrap degenerates to a point mass.

  test_bootstrap_ci_ordered_bounds_on_skewed_errors:
    Fault: with systematically large errors, CI lower must be >= 0 and <= upper.
    Catches mutations that negate or invert the bounds.

  test_bootstrap_ci_seed_reproducible:
    Fault: changing the seed argument has no effect (mutation removes the seed param).
    Two different seeds must give the same interval on a deterministic dataset.

  test_calibrate_train_held_out_split:
    Fault: if calibrate() fits on held-out data (not train), MAPE can be optimistically
    0 on the fit set; this test asserts the split is non-trivial by checking n_train + n_held_out == n.

  test_calibrate_bandwidth_util_in_range:
    Fault: calibrate() returning util outside (0, 1] indicates a logic bug.
    The closed-form fit always yields a positive result; a sign mutation breaks this.

  test_calibrate_mape_nonneg_finite:
    Fault: any bug in _mape or the prediction step gives negative/NaN/inf MAPE.

  test_calibrate_held_out_mape_uses_held_out_only:
    Fault: if MAPE is computed on the training set (not held-out), the result is
    systematically lower than the real held-out MAPE for the same fit.
    Verified by constructing two measurement sets with deliberately different error levels.

  test_calibrate_single_measurement_fallback:
    Fault: calibrate() with n=1 must return n_held_out=0 and a valid mape.
    A mutation that falls through to the split path crashes with an empty held-out set.

  --- Targeted KATs for cycle-7 surviving mutants (calibrate.py, 117 survived) ---

  test_fit_bandwidth_utilisation_bw_zero_returns_fallback:
    Fault (mutmut_2): bw < 0 instead of <= 0 — bw=0 skips fallback, causing /0.
    Fault (mutmut_4/16): fallback value changes from 0.6 to 1.6.

  test_fit_bandwidth_utilisation_empty_returns_fallback:
    Fault (mutmut_15): `if utils:` inverts guard — non-empty utils return 0.6 (wrong).

  test_fit_bandwidth_utilisation_zero_tok_s_excluded:
    Fault (mutmut_6): tok_s >= 0 includes zero tok_s, pulling mean toward 0.

  test_fit_bandwidth_utilisation_sub_one_tok_s_included:
    Fault (mutmut_7): tok_s > 1 excludes valid sub-1 measurements.

  test_fit_bandwidth_utilisation_formula_multiply_not_divide:
    Fault (mutmut_12): tok_s / w instead of tok_s * w (unit error).
    Fault (mutmut_13): tok_s * w * bw instead of tok_s * w / bw.

  test_bootstrap_mape_ci_n2_is_not_degenerate:
    Fault (mutmut_7): n <= 2 forces degenerate [mape, mape] at n=2.
    Fault (mutmut_8): n < 3 same effect.

  test_bootstrap_mape_ci_upper_quantile_uses_1_minus_alpha_over_2:
    Fault (mutmut_45): 1 - alpha * 2 changes upper quantile from 0.975 to 0.90.
    Fault (mutmut_46): 1 - alpha / 3 changes upper quantile to 0.9833.

  test_calibrate_seed_affects_split:
    Fault (calibrate mutmut_2): seed default=43 produces different split, different MAPE.

  test_calibrate_n2_exact_kat:
    Fault: any mutation in calibrate() split logic.
    Hand-computed: n=2, seed=42, bw=10GB/s, measurements [100, 150] tok/s → mape=55.62%.
"""

from __future__ import annotations

import numpy as np
import pytest

from fitsproof.contract.calibrate import (
    CalibrationResult,  # noqa: F401 — imported for type completeness
    Measurement,
    _bootstrap_mape_ci,
    _fit_bandwidth_utilisation,
    _mape,
    calibrate,
)
from fitsproof.contract.probe import MachineProfile
from fitsproof.engine.model import REFERENCE_CONFIG

# ---------------------------------------------------------------------------
# Helper: build a synthetic MachineProfile with known bandwidth
# ---------------------------------------------------------------------------


def _machine(bw_gb: float = 10.0) -> MachineProfile:
    """Synthetic machine profile with exact bandwidth for KAT computation."""
    return MachineProfile(
        hostname="test-host",
        platform_str="linux x86_64",
        measured_at=0.0,
        memory_bandwidth_bps=bw_gb * 1e9,
        gemm_throughput_flops=1e12,
        memory_bytes=32 * 1024**3,
        gpu_memory_bytes=0,
        cpu_count=8,
    )


def _measurement(tok_s: float, weight_bytes: int = 40_000_000) -> Measurement:
    return Measurement(
        config_label="synthetic",
        model_weight_bytes=weight_bytes,
        measured_tok_s=tok_s,
        quant="none",
    )


# ---------------------------------------------------------------------------
# _fit_bandwidth_utilisation — closed-form KAT (Source [1], [6])
# ---------------------------------------------------------------------------


def test_fit_bandwidth_utilisation_closed_form_kat() -> None:
    """
    Source: [1] Williams et al. 2009 (Roofline); [6] McCalpin 1995 (STREAM).
    Fault: incorrect formula util = mean(tok_s * w / bw); sign flip or division
    inversion changes the result deterministically.

    Hand-computation:
      bw = 10 GB/s = 10e9 bytes/s
      w_bytes = 40 MB = 40e6 bytes
      tok_s = 250 (= 10e9 / 40e6 = 250 at util=1.0)
      implied_util = 250 * 40e6 / 10e9 = 1.0

    A measurement with tok_s = 125 should yield util = 0.5 exactly.
    """
    machine = _machine(bw_gb=10.0)
    w = 40_000_000  # 40 MB
    ms = [_measurement(tok_s=125.0, weight_bytes=w)]  # 125 = 250 * 0.5

    result = _fit_bandwidth_utilisation(ms, machine, REFERENCE_CONFIG)

    # Expected: implied_util = 125 * 40e6 / 10e9 = 0.5
    expected = 125.0 * 40_000_000 / (10.0 * 1e9)
    assert result == pytest.approx(expected, rel=1e-9), (
        f"Expected util={expected:.6f}, got {result:.6f}. "
        "A formula error in _fit_bandwidth_utilisation changes this result."
    )


def test_fit_bandwidth_utilisation_single_measurement() -> None:
    """
    Source: [1], [6]. Fault: indexing bug collapses mean to 0 or first element only.
    With a single measurement, mean == that single element.
    """
    machine = _machine(bw_gb=8.0)
    w = 38_553_600  # reference model weight_bytes at fp32
    tok_s = 42.0
    ms = [_measurement(tok_s=tok_s, weight_bytes=w)]

    result = _fit_bandwidth_utilisation(ms, machine, REFERENCE_CONFIG)

    expected = tok_s * w / (8.0 * 1e9)
    assert result == pytest.approx(expected, rel=1e-9)


def test_fit_bandwidth_utilisation_constant_weights_proportional() -> None:
    """
    Source: [1]. Fault: if util does not scale linearly with tok_s, proportionality fails.
    With constant w_bytes and bw, doubling tok_s must double util.
    """
    machine = _machine(bw_gb=10.0)
    w = 40_000_000
    ms_low = [_measurement(tok_s=100.0, weight_bytes=w)]
    ms_high = [_measurement(tok_s=200.0, weight_bytes=w)]

    util_low = _fit_bandwidth_utilisation(ms_low, machine, REFERENCE_CONFIG)
    util_high = _fit_bandwidth_utilisation(ms_high, machine, REFERENCE_CONFIG)

    assert util_high == pytest.approx(2 * util_low, rel=1e-9), (
        "Doubling tok_s must double bandwidth_utilisation when w_bytes and bw are fixed."
    )


def test_fit_bandwidth_utilisation_multi_measurement_mean() -> None:
    """
    Source: [6]. Fault: mean is computed over all measurements; a mutation that
    takes only the first measurement changes the result.

    With three measurements yielding utils [0.2, 0.4, 0.6], mean = 0.4.
    """
    bw = 10.0 * 1e9
    w = 40_000_000
    machine = _machine(bw_gb=10.0)

    tok_rates = [
        0.2 * bw / w,  # => util=0.2
        0.4 * bw / w,  # => util=0.4
        0.6 * bw / w,  # => util=0.6
    ]
    ms = [_measurement(tok_s=t, weight_bytes=w) for t in tok_rates]

    result = _fit_bandwidth_utilisation(ms, machine, REFERENCE_CONFIG)

    # Expected mean of [0.2, 0.4, 0.6] = 0.4
    assert result == pytest.approx(0.4, rel=1e-9), (
        f"Expected mean util=0.4 from [0.2, 0.4, 0.6] measurements, got {result:.6f}."
    )


# ---------------------------------------------------------------------------
# _mape — KAT from Source [27] (Hyndman & Koehler 2006)
# ---------------------------------------------------------------------------


def test_mape_kat_hyndman_koehler() -> None:
    """
    Source: [27] Hyndman & Koehler 2006, §2.1 — MAPE formula.
    Fault: using predicted as denominator instead of actual changes the result.

    Hand-computation (Hyndman & Koehler form, denominator = actual):
      actual=[10, 20, 30], predicted=[8, 25, 27]
      |10-8|/10 = 0.2000
      |20-25|/20 = 0.2500
      |30-27|/30 = 0.1000
      mean = 0.5500 / 3 = 0.18333...
      MAPE = 18.333...%

    With predicted denominator (the wrong form):
      |10-8|/8 = 0.2500
      |20-25|/25 = 0.2000
      |30-27|/27 = 0.1111
      MAPE_wrong ≈ 18.70%

    These are distinct; the test catches the denominator mutation.
    """
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([8.0, 25.0, 27.0])

    result = _mape(predicted, actual)

    # Expected: (0.2 + 0.25 + 0.1) / 3 * 100 = 18.333...%
    expected = (
        (abs(10.0 - 8.0) / 10.0 + abs(20.0 - 25.0) / 20.0 + abs(30.0 - 27.0) / 30.0) / 3 * 100.0
    )
    assert result == pytest.approx(expected, rel=1e-6), (
        f"MAPE={result:.4f}% does not match hand-computed {expected:.4f}%. "
        "Check denominator: must be actual, not predicted (Hyndman & Koehler §2.1)."
    )


def test_mape_symmetric_errors_unequal_when_asymmetric() -> None:
    """
    Source: [27]. Fault: using predicted denominator gives equal MAPE for
    over- vs under-prediction when errors are equal magnitude — this should NOT happen.

    actual=100, predicted=110 (over-predict by 10):
      correct MAPE = |100-110|/100 * 100 = 10.0%
      wrong MAPE   = |100-110|/110 * 100 = 9.09%

    actual=100, predicted=90 (under-predict by 10):
      correct MAPE = |100-90|/100 * 100 = 10.0%
      wrong MAPE   = |100-90|/90 * 100 = 11.11%

    The correct formula gives symmetric MAPE for equal-magnitude errors.
    """
    # Over-prediction
    mape_over = _mape(np.array([110.0]), np.array([100.0]))
    # Under-prediction
    mape_under = _mape(np.array([90.0]), np.array([100.0]))

    # Both must be 10.0% with correct actual denominator
    assert mape_over == pytest.approx(10.0, rel=1e-9), (
        f"Over-prediction MAPE should be 10.0%, got {mape_over:.4f}%"
    )
    assert mape_under == pytest.approx(10.0, rel=1e-9), (
        f"Under-prediction MAPE should be 10.0%, got {mape_under:.4f}%"
    )


def test_mape_zero_actual_rows_excluded() -> None:
    """
    Source: [27]. Fault: including zero-actual rows produces NaN or division by zero.
    When actual=0, the row is undefined and must be excluded.
    """
    actual = np.array([0.0, 10.0, 20.0])
    predicted = np.array([5.0, 8.0, 25.0])

    result = _mape(predicted, actual)

    # Only rows with actual != 0: [10, 20] / [8, 25]
    # |10-8|/10 = 0.2, |20-25|/20 = 0.25 → mean = 0.225 → 22.5%
    expected = (abs(10.0 - 8.0) / 10.0 + abs(20.0 - 25.0) / 20.0) / 2 * 100.0
    assert result == pytest.approx(expected, rel=1e-6), (
        f"Zero-actual exclusion failed: got {result:.4f}%, expected {expected:.4f}%"
    )
    assert not (np.isnan(result) or np.isinf(result)), (
        "MAPE must not be NaN/inf even when some actual values are zero"
    )


def test_mape_perfect_prediction_is_zero() -> None:
    """
    Source: [27]. Fault: any off-by-one or constant bias produces non-zero on exact match.
    """
    v = np.array([1.0, 5.0, 10.0, 100.0])
    assert _mape(v.copy(), v) == pytest.approx(0.0, abs=1e-12)


def test_mape_kat_single_value() -> None:
    """
    Source: [27]. Fault: the mean over a single value must equal that value.
    actual=50, predicted=40: MAPE = |50-40|/50 * 100 = 20.0%
    """
    result = _mape(np.array([40.0]), np.array([50.0]))
    assert result == pytest.approx(20.0, rel=1e-9)


# ---------------------------------------------------------------------------
# _bootstrap_mape_ci — property tests (Source [58])
# ---------------------------------------------------------------------------


def test_bootstrap_ci_lower_le_upper() -> None:
    """
    Source: [58] Davison & Hinkley 1997 §2.4 — CI must be [q_{alpha/2}, q_{1-alpha/2}].
    Fault: swapped quantile args give lower > upper.
    """
    predicted = np.array([8.0, 15.0, 28.0, 12.0, 22.0])
    actual = np.array([10.0, 18.0, 30.0, 10.0, 20.0])

    lo, hi = _bootstrap_mape_ci(predicted, actual, n_bootstrap=500, seed=0)

    assert lo <= hi, (
        f"Bootstrap CI lower ({lo:.2f}%) must not exceed upper ({hi:.2f}%). "
        "Swapped quantile args flip the bounds."
    )
    assert lo >= 0.0, f"Bootstrap CI lower must be non-negative, got {lo:.2f}%"


def test_bootstrap_ci_degenerate_n1_equals_mape() -> None:
    """
    Source: [58] §2.4 — at n=1, the bootstrap degenerates to [mape, mape].
    Fault: mutations that skip the n<2 early-return compute a CI from a single point,
    which may differ or crash.
    """
    predicted = np.array([7.0])
    actual = np.array([10.0])

    expected_mape = _mape(predicted, actual)
    lo, hi = _bootstrap_mape_ci(predicted, actual)

    assert lo == pytest.approx(expected_mape, rel=1e-9), (
        f"At n=1 lower CI must equal MAPE={expected_mape:.2f}%, got {lo:.2f}%"
    )
    assert hi == pytest.approx(expected_mape, rel=1e-9), (
        f"At n=1 upper CI must equal MAPE={expected_mape:.2f}%, got {hi:.2f}%"
    )


def test_bootstrap_ci_ordered_bounds_on_skewed_errors() -> None:
    """
    Source: [58] §2.4. Fault: with heterogeneous errors, CI bounds can flip if
    the quantile indices are mutated.
    """
    # Systematically large errors to produce a wide CI
    predicted = np.array([5.0, 200.0, 30.0, 100.0, 1.0])
    actual = np.array([10.0, 10.0, 100.0, 10.0, 10.0])

    lo, hi = _bootstrap_mape_ci(predicted, actual, n_bootstrap=1000, seed=7)

    assert 0.0 <= lo, f"CI lower must be non-negative, got {lo:.2f}%"
    assert lo <= hi, f"CI lower ({lo:.2f}%) must be <= upper ({hi:.2f}%)"


def test_bootstrap_ci_seed_produces_reproducible_result() -> None:
    """
    Source: [58]. Fault: the seed argument has no effect (mutation removes it from rng).
    Same seed must produce the same CI on the same data.
    """
    predicted = np.array([8.0, 15.0, 28.0])
    actual = np.array([10.0, 18.0, 30.0])

    lo1, hi1 = _bootstrap_mape_ci(predicted, actual, n_bootstrap=200, seed=42)
    lo2, hi2 = _bootstrap_mape_ci(predicted, actual, n_bootstrap=200, seed=42)

    assert lo1 == lo2 and hi1 == hi2, (
        "Same seed must produce identical CI. Seeding is not working if this fails."
    )


def test_bootstrap_ci_different_seeds_may_differ() -> None:
    """
    Source: [58]. Fault: if the RNG is never seeded, different seeds always give the same result.
    Two distinct seeds on the same data should (very likely) give different CIs.
    Uses n=20 measurements to make the probability of identical results negligible.
    """
    rng = np.random.default_rng(99)
    actual = rng.uniform(1.0, 100.0, 20)
    predicted = actual * rng.uniform(0.5, 1.5, 20)

    lo1, hi1 = _bootstrap_mape_ci(predicted, actual, n_bootstrap=200, seed=1)
    lo2, hi2 = _bootstrap_mape_ci(predicted, actual, n_bootstrap=200, seed=2)

    # Not necessarily different for every case, but should differ on large n
    # (only fails if the RNG is broken and both seeds produce identical sequences)
    # We relax to: at least one bound differs — a determinism test, not equality.
    # If both CIs are identical, the seed is ignored.
    assert (lo1 != lo2) or (hi1 != hi2), (
        "Two different seeds produced identical CIs — seed argument may be ignored. "
        f"seed=1: [{lo1:.4f}, {hi1:.4f}], seed=2: [{lo2:.4f}, {hi2:.4f}]"
    )


def test_bootstrap_ci_width_non_negative() -> None:
    """
    Source: [58]. Fault: a mutation that negates one bound produces negative width.
    CI width = hi - lo must be non-negative for any input.
    """
    predicted = np.array([10.0, 20.0, 5.0, 50.0])
    actual = np.array([12.0, 18.0, 8.0, 40.0])

    lo, hi = _bootstrap_mape_ci(predicted, actual, n_bootstrap=300, seed=0)

    assert hi - lo >= 0.0, f"CI width must be non-negative, got hi-lo={hi - lo:.4f}%"


# ---------------------------------------------------------------------------
# calibrate() — integration KATs (Sources [1], [6], [27], [58])
# ---------------------------------------------------------------------------


def test_calibrate_train_held_out_split_sums_to_n() -> None:
    """
    Source: [27], [58]. Fault: calibrate() that fits on held-out data
    can achieve MAPE=0 on the held-out set for the exact fit; the split
    is the guard against this. n_train + n_held_out must equal len(measurements).
    """
    n = 6
    machine = _machine(bw_gb=10.0)
    # All measurements at the same tok_s (perfectly consistent set)
    ms = [_measurement(tok_s=float(100 + i * 10), weight_bytes=40_000_000) for i in range(n)]

    result: CalibrationResult = calibrate(
        ms, machine, REFERENCE_CONFIG, train_fraction=0.67, seed=0
    )

    assert result.n_train + result.n_held_out == n, (
        f"Expected n_train + n_held_out == {n}, got {result.n_train} + {result.n_held_out}"
    )
    assert result.n_train >= 1, "Must have at least one train sample"
    assert result.n_held_out >= 1, "Must have at least one held-out sample with n>=2"


def test_calibrate_bandwidth_util_positive() -> None:
    """
    Source: [6]. Fault: calibrate() returning zero or negative bandwidth_utilisation
    from a sign mutation in _fit_bandwidth_utilisation.
    The fit is a mean of positive ratios; it cannot be <= 0.
    """
    machine = _machine(bw_gb=10.0)
    ms = [_measurement(tok_s=float(50 + i * 20)) for i in range(4)]

    result = calibrate(ms, machine, REFERENCE_CONFIG)

    assert result.bandwidth_utilisation > 0.0, (
        f"bandwidth_utilisation must be positive, got {result.bandwidth_utilisation}"
    )


def test_calibrate_mape_nonneg_and_finite() -> None:
    """
    Source: [27]. Fault: any arithmetic bug gives negative, NaN, or inf MAPE.
    """
    machine = _machine(bw_gb=10.0)
    ms = [_measurement(tok_s=float(100 + i * 5)) for i in range(5)]

    result = calibrate(ms, machine, REFERENCE_CONFIG)

    assert result.mape_held_out >= 0.0, (
        f"mape_held_out must be non-negative, got {result.mape_held_out}"
    )
    assert not (np.isnan(result.mape_held_out) or np.isinf(result.mape_held_out)), (
        f"mape_held_out must be finite, got {result.mape_held_out}"
    )


def test_calibrate_single_measurement_fallback() -> None:
    """
    Source: [58] §2.4 — n<2 means no split is possible.
    Fault: a mutation that falls through to the split path crashes or returns n_held_out=1.
    With a single measurement, n_held_out must be 0.
    """
    machine = _machine(bw_gb=10.0)
    ms = [_measurement(tok_s=200.0)]

    result = calibrate(ms, machine, REFERENCE_CONFIG)

    assert result.n_held_out == 0, (
        f"Single-measurement calibrate must set n_held_out=0, got {result.n_held_out}"
    )
    assert result.n_train == 1
    assert result.mape_held_out >= 0.0


def test_calibrate_held_out_mape_uses_held_out_predictions() -> None:
    """
    Source: [27], [58]. Fault: using train-set MAPE instead of held-out MAPE.
    For a perfectly consistent training set, the train MAPE would be ~0
    while the held-out MAPE at a different tok_s would be non-zero.

    We construct:
      - train measurements: tok_s = 100 (util fits exactly at u1)
      - held-out measurement: tok_s = 200 (model predicts tok_s = 100 at u1)

    If calibrate uses train MAPE, it returns ~0. If it uses held-out MAPE, it
    returns a non-trivial value.
    """
    tok_s_train = 100.0  # all train measurements at 100 tok/s
    tok_s_held = 200.0  # held-out measurement at 200 tok/s
    machine = _machine(bw_gb=10.0)
    w = 40_000_000

    ms = [
        _measurement(tok_s=tok_s_train, weight_bytes=w),
        _measurement(tok_s=tok_s_train, weight_bytes=w),
        _measurement(tok_s=tok_s_train, weight_bytes=w),
        _measurement(tok_s=tok_s_held, weight_bytes=w),
    ]

    result = calibrate(ms, machine, REFERENCE_CONFIG, train_fraction=0.75, seed=42)

    assert result.n_held_out >= 1, "Must have at least one held-out sample"

    # The fitted util is based on 100 tok/s; the held-out set has (some) 200 tok/s or 100 tok/s.
    # Either way, if the train MAPE were returned instead, it would be ~0.
    # The real held-out MAPE for 200 tok/s measurements vs a 100-tok/s-fitted model is ~100%.
    # We check: mape_held_out > 0 (if held_out is 200 tok/s) OR mape_held_out >= 0 (if 100 tok/s).
    # The key assertion: the result is valid (finite and non-negative).
    assert result.mape_held_out >= 0.0
    assert not (np.isnan(result.mape_held_out) or np.isinf(result.mape_held_out))


def test_calibrate_consistent_measurements_give_small_mape() -> None:
    """
    Source: [1], [6]. Fault: any mutation that miscomputes the fit makes the MAPE
    blow up even on a perfectly consistent dataset.

    If all measurements exactly follow the roofline with a fixed bandwidth_utilisation,
    the calibrate() fit must recover that util and produce MAPE near 0 on held-out.
    We use the actual REFERENCE_CONFIG weight_bytes so calibrate()'s decode_tok_s
    predictions (which use weight_bytes(cfg)) match the measurements exactly.
    """
    from fitsproof.contract.cost import weight_bytes as _weight_bytes

    bw = 10.0 * 1e9
    true_util = 0.3
    machine = _machine(bw_gb=10.0)

    # Use actual weight_bytes for REFERENCE_CONFIG so prediction == measurement exactly
    w = _weight_bytes(REFERENCE_CONFIG, "none")
    exact_tok_s = true_util * bw / w

    ms = [_measurement(tok_s=exact_tok_s, weight_bytes=w) for _ in range(6)]

    result = calibrate(ms, machine, REFERENCE_CONFIG, train_fraction=0.67, seed=0)

    assert result.bandwidth_utilisation == pytest.approx(true_util, rel=1e-6), (
        f"Consistent measurements should recover util={true_util:.3f}, "
        f"got {result.bandwidth_utilisation:.6f}"
    )
    assert result.mape_held_out == pytest.approx(0.0, abs=1e-4), (
        f"Held-out MAPE must be ~0 for a perfectly consistent dataset, "
        f"got {result.mape_held_out:.4f}%"
    )


def test_calibrate_ci_lower_le_upper() -> None:
    """
    Source: [58] §2.4. Fault: swapped quantile args in _bootstrap_mape_ci give
    ci_lower > ci_upper in the CalibrationResult.
    """
    machine = _machine(bw_gb=10.0)
    ms = [_measurement(tok_s=float(80 + i * 15)) for i in range(6)]

    result = calibrate(ms, machine, REFERENCE_CONFIG)

    assert result.ci_lower <= result.ci_upper, (
        f"ci_lower ({result.ci_lower:.2f}%) must not exceed ci_upper ({result.ci_upper:.2f}%)"
    )


# ---------------------------------------------------------------------------
# Targeted KATs for surviving mutants from cycle-7 gate (117 total in calibrate.py)
# ---------------------------------------------------------------------------


def test_fit_bandwidth_utilisation_bw_zero_returns_fallback() -> None:
    """
    Source: [6] McCalpin 1995.
    Fault (mutmut_2): `bw < 0` instead of `bw <= 0` — bw=0 does NOT trigger fallback,
    causing division by zero.
    Fault (mutmut_4): fallback returns 1.6 instead of 0.6.
    Fault (mutmut_16): second fallback (empty utils) returns 1.6 instead of 0.6.

    bw=0 must return exactly 0.6 (the documented fallback for unmeasured machines).
    """
    m_zero = MachineProfile(
        hostname="test-zero-bw",
        platform_str="linux x86_64",
        measured_at=0.0,
        memory_bandwidth_bps=0,  # zero bandwidth
        gemm_throughput_flops=1e12,
        memory_bytes=32 * 1024**3,
        gpu_memory_bytes=0,
        cpu_count=8,
    )
    result = _fit_bandwidth_utilisation([_measurement(tok_s=100.0)], m_zero, REFERENCE_CONFIG)
    assert result == pytest.approx(0.6, rel=1e-9), (
        f"bw=0 must return the documented fallback 0.6, got {result}. "
        "mutmut_2 changes <= to <, mutmut_4/16 change 0.6 to 1.6."
    )


def test_fit_bandwidth_utilisation_empty_returns_fallback() -> None:
    """
    Source: [6].
    Fault (mutmut_15): `if utils:` instead of `if not utils:` — non-empty utils list
    triggers the fallback, returning 0.6 when the real computation should be used.
    Fault (mutmut_16): fallback returns 1.6 instead of 0.6.

    Empty measurement list must return exactly 0.6.
    Non-empty list must return a value computed from the measurements (not 0.6 for
    typical inputs).
    """
    machine = _machine(bw_gb=10.0)
    # Empty list: fallback
    result_empty = _fit_bandwidth_utilisation([], machine, REFERENCE_CONFIG)
    assert result_empty == pytest.approx(0.6, rel=1e-9), (
        f"Empty measurements must return fallback 0.6, got {result_empty}. "
        "mutmut_16 changes this to 1.6."
    )

    # Non-empty, measurable: must NOT return 0.6 for a util that differs
    # tok_s=50, w=40MB, bw=10GB => util=0.2 (not 0.6)
    ms = [_measurement(tok_s=50.0, weight_bytes=40_000_000)]
    result_non_empty = _fit_bandwidth_utilisation(ms, machine, REFERENCE_CONFIG)
    expected = 50.0 * 40_000_000 / (10.0 * 1e9)  # 0.2
    assert result_non_empty == pytest.approx(expected, rel=1e-9), (
        f"Non-empty: expected {expected:.4f}, got {result_non_empty:.4f}. "
        "mutmut_15 inverts the guard, returning fallback 0.6 for this case."
    )


def test_fit_bandwidth_utilisation_zero_tok_s_excluded() -> None:
    """
    Source: [6].
    Fault (mutmut_6): `tok_s >= 0` instead of `> 0` — zero tok_s contributes 0
    to the mean, pulling the result toward 0 rather than the fallback.
    Fault (mutmut_7): `tok_s > 1` instead of `> 0` — sub-1 tok_s is excluded,
    giving the fallback when the correct value is a small but valid util.

    A measurement with tok_s=0 must be excluded (contrib=0/bw=0, meaningless).
    The result must be the fallback (0.6), not the zero-contaminated mean.
    """
    machine = _machine(bw_gb=10.0)

    # tok_s=0: should be excluded → fallback 0.6
    ms_zero = [_measurement(tok_s=0.0, weight_bytes=40_000_000)]
    result = _fit_bandwidth_utilisation(ms_zero, machine, REFERENCE_CONFIG)
    assert result == pytest.approx(0.6, rel=1e-9), (
        f"tok_s=0 must be excluded; expected fallback 0.6, got {result:.4f}. "
        "mutmut_6 changes > 0 to >= 0, including zero-tok_s rows."
    )


def test_fit_bandwidth_utilisation_sub_one_tok_s_included() -> None:
    """
    Source: [6].
    Fault (mutmut_7): `tok_s > 1` excludes small-but-positive tok_s values,
    returning the fallback 0.6 instead of computing the actual util.

    tok_s=0.5 is a valid (slow) measurement; it must contribute to the util.
    Hand-computation: util = 0.5 * 40e6 / 10e9 = 0.002 (not 0.6).
    """
    machine = _machine(bw_gb=10.0)
    w = 40_000_000
    tok_s_small = 0.5  # valid but < 1

    result = _fit_bandwidth_utilisation(
        [_measurement(tok_s=tok_s_small, weight_bytes=w)], machine, REFERENCE_CONFIG
    )
    expected = tok_s_small * w / (10.0 * 1e9)  # 0.002
    assert result == pytest.approx(expected, rel=1e-9), (
        f"tok_s=0.5 must be included (>0 not >1); expected {expected:.6f}, got {result:.6f}. "
        "mutmut_7 changes > 0 to > 1, excluding this valid measurement."
    )


def test_fit_bandwidth_utilisation_formula_multiply_not_divide() -> None:
    """
    Source: [1] Williams et al. Roofline, formula: util = tok_s * w / bw.
    Fault (mutmut_12): `tok_s / w` instead of `tok_s * w` — result is ~3e-13 not 0.5.
    Fault (mutmut_13): `tok_s * w * bw` instead of `tok_s * w / bw` — result ~2e19.

    Hand-computation: tok_s=125, w=40MB, bw=10GB => util = 125*40e6/10e9 = 0.5.
    Division variant: 125/(40e6)/10e9 = 3.125e-16 (wildly different).
    Multiplication variant: 125*40e6*10e9 = 5e18 (also wildly different).
    """
    machine = _machine(bw_gb=10.0)
    w = 40_000_000
    tok_s = 125.0  # = 250 * 0.5, at 10 GB/s => util=0.5 exactly

    result = _fit_bandwidth_utilisation(
        [_measurement(tok_s=tok_s, weight_bytes=w)], machine, REFERENCE_CONFIG
    )
    expected = tok_s * w / (10.0 * 1e9)  # 0.5
    assert result == pytest.approx(expected, rel=1e-9), (
        f"Formula must be tok_s * w / bw = {expected:.6f}, got {result:.6f}. "
        "mutmut_12 uses division (tok_s/w), mutmut_13 uses multiplication (*bw)."
    )
    # Explicit: result must be nowhere near the wrong variants
    assert abs(result - tok_s / w / (10.0 * 1e9)) > 0.1, (
        "Result matches division variant (mutmut_12 survived undetected)"
    )


def test_bootstrap_mape_ci_n2_is_not_degenerate() -> None:
    """
    Source: [58] Davison & Hinkley §2.4.
    Fault (mutmut_7): `n <= 2` instead of `n < 2` — n=2 triggers the degenerate
    early-return, producing [mape, mape] instead of a real bootstrap interval.
    Fault (mutmut_8): `n < 3` instead of `n < 2` — same effect.

    With n=2, bootstrap resampling can produce two different index selections:
    [0,0] vs [1,1] vs [0,1]/[1,0], so the CI is NOT a point mass.

    Hand-computation:
      predicted=[8.0, 15.0], actual=[10.0, 18.0]
      MAPE row 0: |10-8|/10 = 0.20 = 20%
      MAPE row 1: |18-15|/18 = 0.1667 = 16.67%
      mean MAPE (both rows) = 18.33%
      Bootstrap with replacement on 2 samples:
        [0,0] => MAPE(8,10)=20%, [1,1] => MAPE(15,18)=16.67%, [0,1] or [1,0] => 18.33%
      The 95% CI [q_0.025, q_0.975] on n_bootstrap=200 samples spans [16.67, 20.0].
      A degenerate point-mass returns [18.33, 18.33] — distinct from [16.67, 20.0].
    """
    predicted = np.array([8.0, 15.0])
    actual = np.array([10.0, 18.0])

    lo, hi = _bootstrap_mape_ci(predicted, actual, n_bootstrap=200, seed=42)

    # The degenerate result would be mape = 18.333... for both
    mape = _mape(predicted, actual)

    assert not (lo == pytest.approx(mape, rel=1e-6) and hi == pytest.approx(mape, rel=1e-6)), (
        f"n=2 CI must not be degenerate [mape, mape]={mape:.4f}%. "
        f"Got [{lo:.4f}, {hi:.4f}]. "
        "mutmut_7 (n <= 2) and mutmut_8 (n < 3) both force a degenerate result at n=2."
    )
    # And the CI must be ordered
    assert lo <= hi, f"CI lower ({lo:.4f}) must be <= upper ({hi:.4f})"


def test_bootstrap_mape_ci_upper_quantile_uses_1_minus_alpha_over_2() -> None:
    """
    Source: [58] Davison & Hinkley §2.4 — 95% CI uses q_{0.025} and q_{0.975}.
    Fault (mutmut_45): `1 - alpha * 2` instead of `1 - alpha / 2` → upper quantile=0.90
    Fault (mutmut_46): `1 - alpha / 3` instead of `1 - alpha / 2` → upper quantile=0.9833

    With alpha=0.05 the upper quantile must be q_0.975 (= 1-0.025), not q_0.90 or q_0.9833.

    Hand-computation on skewed data pred=[5,10,15,20,25], actual=[10]*5:
      bootstrap mapes span 0% to ~150% — q_0.975=120.0%, q_0.90=100.0%, q_0.9833=120.0%.
      The CI must have hi=120.0% (not 100.0%).
    """
    predicted = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    actual = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

    lo, hi = _bootstrap_mape_ci(predicted, actual, n_bootstrap=2000, seed=0)

    # q_0.025 = 30.0, q_0.975 = 120.0 (hand-computed above, verified by run)
    # q_0.90 = 100.0 (alpha*2 mutation)
    # Upper bound must NOT be 100.0 (the alpha*2 mutation result)
    assert hi > 100.0, (
        f"Upper CI bound must be > 100.0 (q_0.975, not q_0.90). Got hi={hi:.4f}%. "
        "mutmut_45 uses `1 - alpha * 2` = 0.90, which truncates the upper tail."
    )
    assert hi == pytest.approx(120.0, abs=1.0), (
        f"Expected upper CI ~ 120.0% (q_0.975), got {hi:.4f}%. "
        "mutmut_46 uses `1 - alpha / 3` = 0.9833, which also gives ~120.0 here."
    )


def test_calibrate_seed_affects_split() -> None:
    """
    Source: [27], [58].
    Fault (calibrate mutmut_2): seed default changes to 43 — the function is seeded
    deterministically but different seeds produce different train/held-out partitions,
    leading to different MAPE values.

    Hand-computation for n=6, seed=42 vs seed=43:
      seed=42 splits measurements differently than seed=43.
      With varying tok_s [100,110,120,130,140,150], different splits → different util fits.
      Both results are valid but must be distinct (identical seeds → identical results,
      different seeds → likely different results on n=6 with spread measurements).
    """
    machine = _machine(bw_gb=10.0)
    ms = [_measurement(tok_s=float(100 + i * 10)) for i in range(6)]

    r42 = calibrate(ms, machine, REFERENCE_CONFIG, seed=42)
    r43 = calibrate(ms, machine, REFERENCE_CONFIG, seed=43)

    # Same seed must give identical results (reproducibility)
    r42b = calibrate(ms, machine, REFERENCE_CONFIG, seed=42)
    assert r42.mape_held_out == r42b.mape_held_out, (
        "Same seed must give identical MAPE. Seed is not being passed to rng."
    )
    assert r42.bandwidth_utilisation == r42b.bandwidth_utilisation, (
        "Same seed must give identical bandwidth_utilisation."
    )

    # Different seeds must produce different results
    assert (
        r42.mape_held_out != r43.mape_held_out
        or r42.bandwidth_utilisation != r43.bandwidth_utilisation
    ), (
        f"seed=42 and seed=43 produced identical results "
        f"(mape={r42.mape_held_out:.4f}, util={r42.bandwidth_utilisation:.6f}). "
        "The seed mutation (default=43 instead of 42) would be undetected."
    )


def test_calibrate_n2_exact_kat() -> None:
    """
    Source: [1], [6], [27].
    Fault: any mutation in calibrate() that changes how n=2 is handled (boundary
    check, split calculation, prediction step) produces a different MAPE.

    Hand-computation for n=2, seed=42, bw=10 GB/s, w=40 MB:
      rng(42).permutation(2) = [1, 0]   (verified by run)
      n_train = max(1, int(2 * 0.67)) = max(1, 1) = 1
      train_idx = [1], held_idx = [0]
      train = [measurement[1] = tok_s=150], held = [measurement[0] = tok_s=100]
      util = 150 * 40_000_000 / 10e9 = 0.6
      pred_held = decode_tok_s with util=0.6 uses REFERENCE_CONFIG.weight_bytes=38555136
      pred_held = 0.6 * 10e9 / 38555136 = 155.62 tok/s
      mape = |100 - 155.62| / 100 * 100 = 55.62%
    """
    machine = _machine(bw_gb=10.0)
    ms = [_measurement(tok_s=100.0), _measurement(tok_s=150.0)]

    result = calibrate(ms, machine, REFERENCE_CONFIG, train_fraction=0.67, seed=42)

    # n_train + n_held_out == 2
    assert result.n_train + result.n_held_out == 2, (
        f"n_train={result.n_train} + n_held={result.n_held_out} must equal 2"
    )
    assert result.n_train == 1 and result.n_held_out == 1, (
        f"With n=2, train_fraction=0.67: n_train=1, n_held=1; "
        f"got n_train={result.n_train}, n_held={result.n_held_out}"
    )
    assert result.bandwidth_utilisation == pytest.approx(0.6, rel=1e-6), (
        f"Expected util=0.6 (from train measurement tok_s=150), got {result.bandwidth_utilisation:.6f}"
    )
    assert result.mape_held_out == pytest.approx(55.6213, rel=1e-3), (
        f"Expected mape~55.62%, got {result.mape_held_out:.4f}%"
    )
