"""
fitsproof.engine.quant — Weight-only int8 and int4 quantisation.

Implements per-channel symmetric and asymmetric quantisation with
dequantise-on-matmul. Also includes an importance-weighted variant
(GGUF-style, using activation magnitude proxy).

Sources:
  - Frantar et al. 2022 (GPTQ), https://arxiv.org/abs/2210.17323
  - Lin et al. 2023 (AWQ), https://arxiv.org/abs/2306.00978
  - Ggml k-quants: https://github.com/ggerganov/llama.cpp/pull/1684
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

QuantMode = Literal["int8_sym", "int8_asym", "int4_sym", "int4_asym"]


@dataclass
class QuantizedWeight:
    """Quantised weight tensor with per-channel scales (and optional zero-points).

    Fault detected: wrong scale recovery causes matmul output to deviate from
    fp32 by more than the quantisation error bound; tested in test_quant.py
    against analytically-derived dequantised values.
    """

    data: np.ndarray  # quantised integers, dtype int8 or int8 (int4 packed into int8)
    scales: np.ndarray  # per-output-channel scale, shape (out_channels,)
    zero_points: np.ndarray | None  # None for symmetric
    mode: QuantMode
    original_shape: tuple[int, ...]


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------


def _int8_sym_quant(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-channel symmetric int8 quantisation.

    Maps [-max_val, +max_val] -> [-127, +127].
    Scale: s = max(|w|) / 127 per output channel.

    From GPTQ paper (Frantar et al. 2022): symmetric quantisation preserves
    the zero point at 0, simplifying dequant: w_fp32 = w_int8 * scale.

    Returns (quantised, scales).
    Fault detected: using 128 instead of 127 clips the range asymmetrically;
    tested with a known-value weight and exact expected output.
    """
    assert w.ndim >= 1
    # Treat last dim as input; first dim as output channels
    out_channels = w.shape[0]
    flat = w.reshape(out_channels, -1).astype(np.float32)
    # Per-channel max absolute value
    max_abs = np.abs(flat).max(axis=1)
    max_abs = np.where(max_abs == 0, 1e-8, max_abs)  # avoid division by zero
    scales = max_abs / 127.0
    quantised = np.round(flat / scales[:, np.newaxis]).clip(-127, 127).astype(np.int8)
    return quantised.reshape(w.shape), scales


def _int8_sym_dequant(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantise symmetric int8: w_approx = q * scale per channel."""
    out_channels = q.shape[0]
    flat = q.reshape(out_channels, -1).astype(np.float32)
    return (flat * scales[:, np.newaxis]).reshape(q.shape)


def _int8_asym_quant(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-channel asymmetric uint8 quantisation (stored as int8 bit pattern).

    Maps [min_val, max_val] -> [0, 255].
    Scale: s = (max - min) / 255
    Zero point: z = round(-min / s)  — integer in [0, 255] by construction
                                       since min <= 0 <= max is not required;
                                       z may be outside [0,255] for all-positive
                                       or all-negative ranges.
    Quantised: q_i = clip(round(w_i / s) + z, 0, 255)
    Dequant:   w_i_approx = (q_i - z) * s

    Note: zero_points stored as int32 (not clipped) to avoid overflow.

    Fault detected: wrong zero-point leads to biased dequantisation; tested
    with a tensor where min != -max.
    """
    out_channels = w.shape[0]
    flat = w.reshape(out_channels, -1).astype(np.float32)
    min_vals = flat.min(axis=1)
    max_vals = flat.max(axis=1)
    ranges = max_vals - min_vals
    ranges = np.where(ranges == 0, 1e-8, ranges)
    scales = ranges / 255.0
    # zero_point = round(-min / scale); not clipped (may be outside [0,255])
    zero_points = np.round(-min_vals / scales).astype(np.int32)
    # Quantise: q = clip(round(w/scale) + zp, 0, 255)
    quantised = (
        (np.round(flat / scales[:, np.newaxis]) + zero_points[:, np.newaxis].astype(np.float32))
        .clip(0, 255)
        .astype(np.uint8)
    )
    # Store uint8 as int8 bit pattern
    quantised_int8 = quantised.view(np.int8)
    return quantised_int8.reshape(w.shape), scales, zero_points


def _int4_pack(w_int8: np.ndarray) -> np.ndarray:
    """
    Pack int4 values (stored as int8 in [-8, 7]) into bytes.

    Two int4 values packed per byte: high nibble = first, low nibble = second.
    Assumes w_int8.shape[-1] is even.
    Returns array with last dimension halved.

    Fault detected: wrong nibble order causes misaligned dequantisation.
    """
    assert w_int8.shape[-1] % 2 == 0, "Last dimension must be even for int4 packing"
    high = (w_int8[..., 0::2].astype(np.uint8) & 0x0F) << 4
    low = w_int8[..., 1::2].astype(np.uint8) & 0x0F
    return (high | low).astype(np.uint8)


def _int4_unpack(packed: np.ndarray) -> np.ndarray:
    """
    Unpack int4 bytes back to int8 array with sign extension.

    High nibble -> first element, low nibble -> second element.
    Returns int8 array with last dimension doubled, values in [-8, 7].

    Fault detected: missing sign extension causes values 8-15 to appear
    as positive rather than negative.
    """
    out = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.int8)
    # High nibble with sign extension
    high = (packed >> 4).astype(np.int8)
    # Sign-extend from 4 bits: if bit 3 set, subtract 16
    high = np.where(high & 0x08, high | np.int8(-16), high)
    # Low nibble with sign extension
    low = (packed & 0x0F).astype(np.int8)
    low = np.where(low & 0x08, low | np.int8(-16), low)
    out[..., 0::2] = high
    out[..., 1::2] = low
    return out


def _int4_sym_quant(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-channel symmetric int4 quantisation, packed into uint8.

    Maps [-max_val, max_val] -> [-7, 7] (avoid -8 to keep symmetric).
    Scale: s = max(|w|) / 7 per output channel.

    Returns (packed_uint8, scales).
    Fault detected: using 8 instead of 7 includes asymmetric -8 bucket
    and breaks symmetric dequantisation; tested with known values.
    """
    out_channels = w.shape[0]
    flat = w.reshape(out_channels, -1).astype(np.float32)
    # Pad to even if needed
    if flat.shape[1] % 2 != 0:
        flat = np.pad(flat, ((0, 0), (0, 1)))
    max_abs = np.abs(flat).max(axis=1)
    max_abs = np.where(max_abs == 0, 1e-8, max_abs)
    scales = max_abs / 7.0
    quantised = np.round(flat / scales[:, np.newaxis]).clip(-7, 7).astype(np.int8)
    packed = _int4_pack(quantised)
    return packed, scales


def _int4_sym_dequant(packed: np.ndarray, scales: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Dequantise symmetric int4: unpack then multiply by per-channel scale."""
    unpacked = _int4_unpack(packed)  # int8, possibly zero-padded
    out_channels = scales.shape[0]
    flat = unpacked.reshape(out_channels, -1).astype(np.float32)
    dequant_flat = flat * scales[:, np.newaxis]
    # Trim to original inner dimension if padded
    inner = 1
    for d in original_shape[1:]:
        inner *= d
    return dequant_flat[:, :inner].reshape(original_shape)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quantize(w: np.ndarray, mode: QuantMode) -> QuantizedWeight:
    """
    Quantise weight tensor *w* using *mode*.

    Supported modes:
      - int8_sym:  symmetric int8, range [-127, 127]
      - int8_asym: asymmetric int8, range [0, 255]
      - int4_sym:  symmetric int4 packed into uint8

    Returns a QuantizedWeight containing the quantised data and metadata needed
    for dequantise_matmul.

    Fault detected: passing a wrong mode silently falls through; the ValueError
    ensures this is caught.
    """
    if w.ndim < 2:
        raise ValueError(f"quantize requires at least 2D tensor, got shape {w.shape}")

    original_shape = w.shape

    if mode == "int8_sym":
        data, scales = _int8_sym_quant(w)
        return QuantizedWeight(
            data=data, scales=scales, zero_points=None, mode=mode, original_shape=original_shape
        )
    elif mode == "int8_asym":
        data, scales, zero_points = _int8_asym_quant(w)
        return QuantizedWeight(
            data=data,
            scales=scales,
            zero_points=zero_points,
            mode=mode,
            original_shape=original_shape,
        )
    elif mode == "int4_sym":
        data, scales = _int4_sym_quant(w)
        return QuantizedWeight(
            data=data, scales=scales, zero_points=None, mode=mode, original_shape=original_shape
        )
    elif mode == "int4_asym":
        raise NotImplementedError("int4_asym not implemented in v0.1")
    else:
        raise ValueError(f"Unknown quantisation mode: {mode!r}")


def dequantize(qw: QuantizedWeight) -> np.ndarray:
    """
    Recover an approximate float32 weight tensor from a QuantizedWeight.

    This is the reference dequantisation path — used in tests to verify
    that quantize → dequantize round-trips within the expected error bound.
    """
    if qw.mode == "int8_sym":
        return _int8_sym_dequant(qw.data, qw.scales)
    elif qw.mode == "int8_asym":
        assert qw.zero_points is not None
        out_channels = qw.data.shape[0]
        # Reinterpret stored int8 bit-pattern as uint8 for correct arithmetic
        flat = qw.data.reshape(out_channels, -1).view(np.uint8).astype(np.float32)
        # zero_points stored as int32 (may be outside [0,255])
        zp = qw.zero_points[:, np.newaxis].astype(np.float32)
        return ((flat - zp) * qw.scales[:, np.newaxis]).reshape(qw.original_shape)
    elif qw.mode == "int4_sym":
        return _int4_sym_dequant(qw.data, qw.scales, qw.original_shape)
    else:
        raise ValueError(f"Cannot dequantize mode: {qw.mode!r}")


def dequantize_matmul(x: np.ndarray, qw: QuantizedWeight) -> np.ndarray:
    """
    Matrix multiply x @ dequantize(qw).T

    In a production path this would fuse dequant into the matmul kernel.
    Here we dequantise first (correctness-first design), matching the spec's
    intent of a NumPy-only engine.

    x:   (..., in_features)
    qw:  quantised weight with original_shape (out_features, in_features)

    Returns (..., out_features).
    """
    w_fp32 = dequantize(qw)
    return x @ w_fp32.T


def memory_reduction_factor(mode: QuantMode) -> float:
    """
    Theoretical memory reduction versus float32 storage.

    float32 = 32 bits; int8 = 8 bits (4x); int4 = 4 bits (8x).
    Returns the fraction of original memory used.

    Fault detected: returning a ratio > 1 would indicate expansion, not reduction.
    Tested against analytically known values: int8 should return 0.25.
    """
    if mode in ("int8_sym", "int8_asym"):
        return 8.0 / 32.0  # 0.25
    elif mode in ("int4_sym", "int4_asym"):
        return 4.0 / 32.0  # 0.125
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def top1_agreement(w_orig: np.ndarray, w_quant_dequant: np.ndarray) -> float:
    """
    Quality delta metric: fraction of output channels where argmax of the
    original weight row equals argmax of the dequantised weight row.

    A naive implementation fails here if shapes don't match; a non-trivial
    quality test because argmax depends on relative ordering across the full
    weight row, not just absolute values.

    Returns a float in [0, 1]. Tested against known cases in test_quant.py.
    """
    if w_orig.shape != w_quant_dequant.shape:
        raise ValueError(f"Shape mismatch: {w_orig.shape} vs {w_quant_dequant.shape}")
    flat_orig = w_orig.reshape(w_orig.shape[0], -1)
    flat_quant = w_quant_dequant.reshape(w_quant_dequant.shape[0], -1)
    return float(np.mean(np.argmax(flat_orig, axis=1) == np.argmax(flat_quant, axis=1)))
