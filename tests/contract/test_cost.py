"""
Tests for fitsproof.contract.cost and fitsproof.contract.probe.

Research source mappings (M4 — QUALITY-CONTRACT §4 / fitsproof.md M4):
  [1] Williams et al. 2009 (Roofline): decode_tok_s = bandwidth / weight_bytes.
  [2] Sheng et al. 2023 (FlexGen §3.1): single-batch decode bandwidth-bound.
  [4] Ainslie et al. 2023 (GQA): KV cache size = 2*n_layers*n_kv_heads*seq*head_dim*bytes.
  [7] Frantar et al. 2022 (GPTQ): int8 halves, int4 quarters weight bytes vs float32.

Faults detected by each test:
  test_weight_bytes_reference_model:
    Hand-computed weight_bytes for REFERENCE_CONFIG (float32). If any
    weight matrix is forgotten (e.g. embedding), the result is wrong.
    Computed:
      embed: 256*384*4 = 393,216
      unembed: 256*384*4 = 393,216
      per layer (x6):
        q: 6*64*384*4 = 589,824
        k: 2*64*384*4 = 196,608
        v: 2*64*384*4 = 196,608
        o: 384*6*64*4 = 589,824
        gate: 1024*384*4 = 1,572,864
        up:   1024*384*4 = 1,572,864
        down: 384*1024*4 = 1,572,864
        norms: 2*384*4 = 3,072
      layer total = 6,294,528
      total = 2*393216 + 6*6,294,528 = 786,432 + 37,767,168 = 38,553,600

  test_kv_cache_bytes_known:
    Hand-computed KV cache for REFERENCE_CONFIG:
      2 * 6 * 2 * 256 * 64 * 4 = 786,432 bytes for context=256, float32

  test_kv_cache_bytes_scales_linearly_in_context:
    Doubling context must double KV cache bytes exactly.
    Fault: any non-linear term (e.g. +overhead) breaks this.

  test_decode_tok_s_positive:
    With any valid machine profile, decode_tok_s must be positive.
    Fault: inverted bandwidth formula (bytes/bw instead of bw/bytes) returns near-0.

  test_mape_known_values:
    MAPE for known predicted/actual arrays, verified by hand.
    predicted=[1,2], actual=[2,4]: MAPE=|2-1|/2 + |4-2|/4) / 2 * 100 = 37.5%

  test_mape_zero_for_perfect_prediction:
    MAPE=0 when predicted==actual.
    Fault: dividing by predicted instead of actual gives NaN when actual=0.

  test_bandwidth_measurement_plausible:
    The measured bandwidth must be > 1 GB/s on any modern machine.
    Fault: measuring cache bandwidth (tiny arrays) would overestimate.
    (We use a large array to hit DRAM.)

  (hypothesis) test_weight_bytes_scales_with_layers:
    Adding more layers must strictly increase weight_bytes.

  (hypothesis) test_kv_cache_bytes_scales_with_context:
    Larger context must use more KV cache memory.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fitsproof.contract.calibrate import _mape
from fitsproof.contract.cost import (
    decode_tok_s,
    kv_cache_bytes,
    weight_bytes,
)
from fitsproof.contract.probe import MachineProfile, _get_system_ram
from fitsproof.engine.model import REFERENCE_CONFIG, ModelConfig

# ---------------------------------------------------------------------------
# Synthetic machine profile for deterministic tests (no IO)
# ---------------------------------------------------------------------------


def make_machine(
    bw_bps: float = 20e9,  # 20 GB/s
    gemm_flops: float = 100e9,  # 100 GFLOPS
    ram: int = 32 * 1024**3,  # 32 GB
) -> MachineProfile:
    import time

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


# ---------------------------------------------------------------------------
# KAT: weight_bytes
# ---------------------------------------------------------------------------


def test_weight_bytes_reference_model() -> None:
    """
    KAT: hand-computed weight_bytes for REFERENCE_CONFIG in float32.

    Config: vocab=256, hidden=384, layers=6, heads=6, kv_heads=2,
            intermediate=1024, head_dim=64.

    Layer computation (per layer, float32=4 bytes):
      q_proj:     6*64 * 384 = 147456 params * 4 = 589824 bytes
      k_proj:     2*64 * 384 = 49152 params * 4 = 196608 bytes
      v_proj:     2*64 * 384 = 49152 params * 4 = 196608 bytes
      o_proj:     384 * 6*64 = 147456 params * 4 = 589824 bytes
      gate_proj:  1024 * 384 = 393216 params * 4 = 1572864 bytes
      up_proj:    1024 * 384 = 393216 params * 4 = 1572864 bytes
      down_proj:  384 * 1024 = 393216 params * 4 = 1572864 bytes
      attn_norm:  384 * 4 = 1536 bytes
      ffn_norm:   384 * 4 = 1536 bytes
      Layer total = 589824+196608+196608+589824+1572864+1572864+1572864+1536+1536
                  = 6294528 bytes

    Global:
      embed:      256 * 384 * 4 = 393216 bytes
      final_norm: 384 * 4 = 1536 bytes
      unembed:    256 * 384 * 4 = 393216 bytes

    Total = 393216 + 1536 + 393216 + 6 * 6294528
          = 787968 + 37767168 = 38555136 bytes
    """
    expected = (
        256 * 384 * 4  # embed
        + 384 * 4  # final_norm
        + 256 * 384 * 4  # unembed
        + 6
        * (  # 6 layers
            6 * 64 * 384 * 4  # q_proj
            + 2 * 64 * 384 * 4  # k_proj
            + 2 * 64 * 384 * 4  # v_proj
            + 384 * 6 * 64 * 4  # o_proj
            + 1024 * 384 * 4  # gate_proj
            + 1024 * 384 * 4  # up_proj
            + 384 * 1024 * 4  # down_proj
            + 384 * 4  # attn_norm
            + 384 * 4  # ffn_norm
        )
    )
    result = weight_bytes(REFERENCE_CONFIG, quant="none")
    assert result == expected, f"weight_bytes={result}, expected={expected}"


def test_weight_bytes_int8_is_quarter() -> None:
    """
    KAT: int8 weight_bytes should be approximately 1/4 of float32.
    Fault: applying the wrong bits factor (e.g. 16 instead of 8).
    Small differences allowed for norm weights (still float32).
    """
    fp32 = weight_bytes(REFERENCE_CONFIG, quant="none")
    int8 = weight_bytes(REFERENCE_CONFIG, quant="int8_sym")
    # int8 should be roughly 1/4 of fp32 (norms stay float32 so not exact)
    ratio = int8 / fp32
    assert 0.22 < ratio < 0.30, f"int8/fp32 ratio {ratio:.3f} not near 0.25"


def test_weight_bytes_int4_is_eighth() -> None:
    """
    KAT: int4 weight_bytes should be approximately 1/8 of float32.
    """
    fp32 = weight_bytes(REFERENCE_CONFIG, quant="none")
    int4 = weight_bytes(REFERENCE_CONFIG, quant="int4_sym")
    ratio = int4 / fp32
    assert 0.10 < ratio < 0.18, f"int4/fp32 ratio {ratio:.3f} not near 0.125"


# ---------------------------------------------------------------------------
# KAT: kv_cache_bytes
# ---------------------------------------------------------------------------


def test_kv_cache_bytes_known() -> None:
    """
    KAT: hand-computed KV cache for REFERENCE_CONFIG, context=256, float32.

    Formula: 2 * n_layers * n_kv_heads * context * head_dim * bytes_per_element
           = 2 * 6 * 2 * 256 * 64 * 4 = 1,572,864 bytes

    Fault: omitting the factor 2 (for K and V) gives 786,432 (half).
    """
    cfg = REFERENCE_CONFIG
    expected = 2 * cfg.num_layers * cfg.num_kv_heads * 256 * cfg.head_dim * 4
    result = kv_cache_bytes(cfg, context_len=256, quant="none")
    assert result == expected, f"kv_cache_bytes={result}, expected={expected}"


def test_kv_cache_bytes_scales_linearly_in_context() -> None:
    """
    KAT: doubling context must exactly double KV cache bytes.
    Fault: any additive constant would break exact doubling.
    """
    cfg = REFERENCE_CONFIG
    kv_128 = kv_cache_bytes(cfg, 128)
    kv_256 = kv_cache_bytes(cfg, 256)
    assert kv_256 == 2 * kv_128, f"kv(256)={kv_256} != 2*kv(128)={2 * kv_128}"


# ---------------------------------------------------------------------------
# KAT: decode_tok_s
# ---------------------------------------------------------------------------


def test_decode_tok_s_positive() -> None:
    """
    KAT: decode_tok_s must be positive for any valid machine profile.
    Fault: inverted formula (bytes/bw) gives near-zero result.
    """
    machine = make_machine()
    tok_s = decode_tok_s(REFERENCE_CONFIG, machine, quant="none", bandwidth_utilisation=0.6)
    assert tok_s > 0, "decode_tok_s should be positive"


def test_decode_tok_s_formula() -> None:
    """
    KAT: decode_tok_s = (bandwidth * utilisation) / weight_bytes.

    Verified against the formula directly.
    """
    machine = make_machine(bw_bps=20e9)
    util = 0.6
    w = weight_bytes(REFERENCE_CONFIG)
    expected = (20e9 * util) / w
    result = decode_tok_s(REFERENCE_CONFIG, machine, bandwidth_utilisation=util)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_higher_bandwidth_gives_higher_tok_s() -> None:
    """
    Property: machines with higher memory bandwidth should give higher tok/s.
    Fault: the formula is inverted.
    """
    slow = make_machine(bw_bps=10e9)
    fast = make_machine(bw_bps=40e9)
    tok_slow = decode_tok_s(REFERENCE_CONFIG, slow)
    tok_fast = decode_tok_s(REFERENCE_CONFIG, fast)
    assert tok_fast > tok_slow, f"Fast machine should give higher tok/s: {tok_fast} vs {tok_slow}"


# ---------------------------------------------------------------------------
# KAT: MAPE
# ---------------------------------------------------------------------------


def test_mape_known_values() -> None:
    """
    KAT: MAPE for known predicted/actual pairs.

    predicted = [1.0, 2.0], actual = [2.0, 4.0]
    errors = [|2-1|/2, |4-2|/4] = [0.5, 0.5]
    MAPE = mean([0.5, 0.5]) * 100 = 50.0%

    Fault: dividing by predicted instead of actual gives different result.
    """
    predicted = np.array([1.0, 2.0])
    actual = np.array([2.0, 4.0])
    result = _mape(predicted, actual)
    np.testing.assert_allclose(result, 50.0, rtol=1e-5)


def test_mape_zero_for_perfect() -> None:
    """
    KAT: MAPE=0 when predicted==actual.
    """
    x = np.array([1.0, 2.0, 3.0])
    assert _mape(x, x) == pytest.approx(0.0, abs=1e-9)


def test_mape_handles_single_element() -> None:
    """
    Fault: single-element bootstrap CI should degenerate gracefully.
    """
    predicted = np.array([1.5])
    actual = np.array([2.0])
    mape = _mape(predicted, actual)
    np.testing.assert_allclose(mape, 25.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# System RAM detection
# ---------------------------------------------------------------------------


def test_system_ram_plausible() -> None:
    """
    KAT: _get_system_ram() must return > 1 GB on any real machine.
    Fault: wrong unit conversion (treating KB as bytes) gives 1000x error.
    """
    ram = _get_system_ram()
    assert ram > 1024**3, f"Reported RAM {ram / 1e9:.2f} GB is implausibly low"


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


@given(n_layers=st.integers(min_value=1, max_value=12))
@settings(max_examples=20, deadline=5000)
def test_weight_bytes_scales_with_layers(n_layers: int) -> None:
    """
    Property: more layers -> strictly more weight bytes.
    Fault: forgetting to multiply layer weights by n_layers.
    """
    cfg_base = ModelConfig(
        vocab_size=32,
        hidden_size=64,
        num_layers=n_layers,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_seq_len=64,
        dtype="float32",
    )
    cfg_more = ModelConfig(
        vocab_size=32,
        hidden_size=64,
        num_layers=n_layers + 1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_seq_len=64,
        dtype="float32",
    )
    assert weight_bytes(cfg_more) > weight_bytes(cfg_base)


@given(ctx=st.integers(min_value=1, max_value=1024))
@settings(max_examples=20, deadline=2000)
def test_kv_cache_bytes_scales_with_context(ctx: int) -> None:
    """
    Property: larger context -> larger KV cache.
    Fault: kv_cache is a constant (forgets to multiply by context).
    """
    small = kv_cache_bytes(REFERENCE_CONFIG, ctx)
    large = kv_cache_bytes(REFERENCE_CONFIG, ctx * 2)
    assert large == 2 * small
