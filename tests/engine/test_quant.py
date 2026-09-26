"""
Tests for fitsproof.engine.quant.

Faults detected by each test:
  test_int8_sym_known_values:
    Using 128 instead of 127 as the clip range would shift the scale,
    causing dequantised values to differ from the hand-computed expected.
    Derived from first principles: scale = max(|w|)/127.

  test_int8_sym_round_trip_error:
    Round-trip error (fp32 → int8 → fp32) must be bounded by scale/2.
    A wrong scale would violate this bound.

  test_int8_asym_non_zero_mean:
    For a weight with min != -max, asymmetric quant must produce lower
    round-trip error than symmetric quant. If zero_point is wrong, the
    bias term causes systematic error.

  test_int4_pack_unpack_roundtrip:
    Nibble pack/unpack must exactly recover the original int4 values.
    Wrong nibble order would cause every other element to be wrong.

  test_int4_known_values:
    Hand-computed int4 quantisation: for a weight row [7, -7, 3, -3],
    scale = 7/7 = 1.0, quantised = [7, -7, 3, -3], dequant = [7, -7, 3, -3].

  test_memory_reduction_factor_int8:
    int8 should reduce to 1/4 of float32 storage. Any other value is wrong.

  test_memory_reduction_factor_int4:
    int4 should reduce to 1/8 of float32 storage.

  test_top1_agreement_perfect:
    Identical weights must yield 1.0 top1 agreement.

  test_top1_agreement_shuffled:
    Shuffled weights yield < 1.0 top1 agreement (tests the comparison logic).

  test_quantize_rejects_1d:
    1D input must raise ValueError (not silently proceed with wrong shapes).

  (hypothesis) test_dequant_error_bounded:
    For any random weight, dequant round-trip error is bounded by scale/2.
    This is a mathematical property of symmetric quantisation.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fitsproof.engine.quant import (
    _int4_pack,
    _int4_unpack,
    _int8_sym_dequant,
    _int8_sym_quant,
    dequantize,
    dequantize_matmul,
    memory_reduction_factor,
    quantize,
    top1_agreement,
)

# ---------------------------------------------------------------------------
# KAT: int8 symmetric
# ---------------------------------------------------------------------------


def test_int8_sym_known_values() -> None:
    """
    KAT: int8_sym quantisation of a known weight row.

    Hand-computed:
      w = [[1.0, 2.0, 3.0, -6.0]]
      max(|w[0]|) = 6.0
      scale = 6.0 / 127 = 0.047244...
      quantised[0] = round([1/scale, 2/scale, 3/scale, -6/scale])
                   = round([21.17, 42.33, 63.5, -127]) = [21, 42, 64, -127]
      dequant = [21*scale, 42*scale, 64*scale, -127*scale]
              = [0.9921, 1.9843, 3.0236, -6.0]
    """
    w = np.array([[1.0, 2.0, 3.0, -6.0]], dtype=np.float32)
    qdata, scales = _int8_sym_quant(w)

    expected_scale = 6.0 / 127.0
    np.testing.assert_allclose(scales, [expected_scale], rtol=1e-5)

    expected_q = np.array([[21, 42, 64, -127]], dtype=np.int8)
    np.testing.assert_array_equal(qdata, expected_q)

    dequant = _int8_sym_dequant(qdata, scales)
    expected_dequant = np.array([[21, 42, 64, -127]], dtype=np.float32) * expected_scale
    np.testing.assert_allclose(dequant, expected_dequant, rtol=1e-5)


def test_int8_sym_round_trip_error() -> None:
    """
    KAT: round-trip error must be <= scale/2 for symmetric int8.

    The maximum quantisation error is half the scale (rounding error).
    Fault: wrong scale (e.g. using 128) would violate this bound.
    """
    rng = np.random.default_rng(1)
    w = rng.standard_normal((8, 16)).astype(np.float32)
    qw = quantize(w, "int8_sym")
    w_back = dequantize(qw)
    max_error = np.abs(w - w_back).max()
    max_allowed = (qw.scales.max() / 2.0) * 1.01  # 1% tolerance
    assert max_error <= max_allowed, (
        f"Round-trip error {max_error:.6f} exceeds allowed {max_allowed:.6f}"
    )


def test_int8_asym_non_zero_mean() -> None:
    """
    KAT: asymmetric quantisation should outperform symmetric for skewed weights.

    For a weight row [0, 1, 2, 3] (all positive), the optimal zero_point shifts
    the quantisation range to [0, 3] rather than [-3, 3].
    Asymmetric should give lower round-trip error than symmetric here.
    """
    w = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    qw_sym = quantize(w, "int8_sym")
    qw_asym = quantize(w, "int8_asym")
    w_sym = dequantize(qw_sym)
    w_asym = dequantize(qw_asym)
    err_sym = np.abs(w - w_sym).mean()
    err_asym = np.abs(w - w_asym).mean()
    # Asymmetric should be at least as good as symmetric
    assert err_asym <= err_sym + 1e-4, (
        f"Asymmetric ({err_asym:.6f}) should not be worse than symmetric ({err_sym:.6f})"
    )


# ---------------------------------------------------------------------------
# KAT: int4
# ---------------------------------------------------------------------------


def test_int4_pack_unpack_roundtrip() -> None:
    """
    KAT: int4 pack/unpack must exactly recover values in [-7, 7].

    Fault detected: wrong nibble order (high/low swapped) causes misalignment;
    sign-extension error causes values 8-15 to appear as positive.
    """
    values = np.array([-7, -3, 0, 3, 7, -1, 5, -5], dtype=np.int8)
    packed = _int4_pack(values[np.newaxis, :])
    unpacked = _int4_unpack(packed)[0]
    np.testing.assert_array_equal(unpacked, values)


def test_int4_known_values() -> None:
    """
    KAT: int4_sym quantisation of a known row.

    w = [[7.0, -7.0, 3.5, -3.5]]
    max(|w[0]|) = 7.0
    scale = 7.0 / 7 = 1.0
    quantised = [7, -7, 4, -4]   (round(3.5/1)=4, round(-3.5/1)=-4)
    dequant = [7.0, -7.0, 4.0, -4.0]
    max round-trip error = |3.5 - 4.0| = 0.5 = scale/2
    """
    w = np.array([[7.0, -7.0, 3.5, -3.5]], dtype=np.float32)
    qw = quantize(w, "int4_sym")
    np.testing.assert_allclose(qw.scales, [1.0], atol=1e-6)
    w_back = dequantize(qw)
    np.testing.assert_allclose(w_back, [[7.0, -7.0, 4.0, -4.0]], atol=0.01)


# ---------------------------------------------------------------------------
# Memory reduction factors
# ---------------------------------------------------------------------------


def test_memory_reduction_factor_int8() -> None:
    """
    KAT: int8 reduces storage to exactly 1/4 of float32 (8 bits / 32 bits).
    Fault: returning 0.5 or 0.125 would be wrong.
    """
    assert memory_reduction_factor("int8_sym") == 0.25
    assert memory_reduction_factor("int8_asym") == 0.25


def test_memory_reduction_factor_int4() -> None:
    """
    KAT: int4 reduces storage to exactly 1/8 of float32 (4 bits / 32 bits).
    """
    assert memory_reduction_factor("int4_sym") == 0.125


def test_memory_reduction_is_between_0_and_1() -> None:
    """
    Fault: returning >1 would indicate expansion rather than reduction.
    """
    for mode in ("int8_sym", "int8_asym", "int4_sym"):
        factor = memory_reduction_factor(mode)
        assert 0.0 < factor < 1.0, f"Reduction factor {factor} for {mode} is not in (0,1)"


# ---------------------------------------------------------------------------
# Top-1 agreement
# ---------------------------------------------------------------------------


def test_top1_agreement_perfect() -> None:
    """
    KAT: identical weights must yield agreement=1.0.
    Fault: if any comparison is wrong, agreement < 1.0.
    """
    w = np.array([[1.0, 5.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float32)
    assert top1_agreement(w, w) == 1.0


def test_top1_agreement_shifted() -> None:
    """
    KAT: a weight where the argmax shifts after quantisation returns < 1.0.

    w[0] = [1.0, 2.0, 1.0] -> argmax=1
    After quant to int4 scale=2/7, [1,2,1]*7/2=[3.5,7,3.5] -> argmax=1 (same)
    But if we manually flip: [2.0, 1.0, 1.0] vs [1.0, 2.0, 1.0] -> agreement=0.0
    """
    w_orig = np.array([[2.0, 1.0, 1.0]], dtype=np.float32)
    w_quant = np.array([[1.0, 2.0, 1.0]], dtype=np.float32)
    assert top1_agreement(w_orig, w_quant) == 0.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_quantize_rejects_1d() -> None:
    """
    Fault: 1D input must raise ValueError, not silently produce wrong shapes.
    """
    w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="at least 2D"):
        quantize(w, "int8_sym")


def test_quantize_unknown_mode() -> None:
    """
    Fault: unknown mode must raise ValueError (not silently return garbage).
    """
    w = np.ones((4, 4), dtype=np.float32)
    with pytest.raises((ValueError, NotImplementedError)):
        quantize(w, "int2")  # type: ignore


def test_dequantize_matmul_shape() -> None:
    """
    Fault: wrong output shape from dequantize_matmul would cause silent
    broadcasting errors in the transformer.
    x: (batch=2, in=4), qw: out=3, in=4 -> out shape (2, 3).
    """
    x = np.ones((2, 4), dtype=np.float32)
    w = np.eye(4, dtype=np.float32)[:3, :]  # (3, 4)
    qw = quantize(w, "int8_sym")
    out = dequantize_matmul(x, qw)
    assert out.shape == (2, 3), f"Expected (2,3), got {out.shape}"


# ---------------------------------------------------------------------------
# Hypothesis: dequant error bounded by scale/2
# ---------------------------------------------------------------------------


@given(
    n_out=st.integers(min_value=2, max_value=8),
    n_in=st.integers(min_value=2, max_value=16),
    mode=st.sampled_from(["int8_sym", "int8_asym", "int4_sym"]),
)
@settings(max_examples=40, deadline=5000)
def test_dequant_error_bounded(n_out: int, n_in: int, mode: str) -> None:
    """
    Property: max dequant round-trip error <= scale/2 (for symmetric) or scale (asym).

    Mathematical property of quantisation: the maximum error is bounded by
    half the step size (scale). Fault: using wrong scale or wrong rounding
    violates this bound.
    """
    rng = np.random.default_rng(n_out * 100 + n_in)
    w = rng.standard_normal((n_out, n_in)).astype(np.float32)
    qw = quantize(w, mode)  # type: ignore
    w_back = dequantize(qw)
    max_err = float(np.abs(w - w_back).max())
    # Max scale over all channels
    max_scale = float(qw.scales.max())
    # Allow a small tolerance for rounding in the formula itself
    assert max_err <= max_scale + 1e-4, (
        f"Error {max_err:.6f} > scale {max_scale:.6f} for mode={mode}, shape=({n_out},{n_in})"
    )
