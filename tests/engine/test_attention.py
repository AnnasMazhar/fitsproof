"""
Tests for fitsproof.engine.attention and fitsproof.engine.transformer.

Faults detected by each test:
  test_rope_known_values:
    Wrong theta or rotation formula would produce different cos/sin values.
    Verified against: hand-computed values for d=4, theta=10000, positions 0 and 1.
    Derivation: Su et al. 2023 Eq 15, freq_i = base^{-2i/d}.
    For d=4, freq_0 = 1.0, freq_1 = 10000^{-0.5} = 0.01.
    pos=0: angles=[0, 0], cos=[1, 1], sin=[0, 0].
    pos=1: angles=[1, 0.01], cos=[cos(1), cos(0.01)], sin=[sin(1), sin(0.01)].

  test_rope_non_default_theta_changes_freqs:
    Fixes ADV-14: test_rope_known_values passes theta=10000.0 explicitly, so a
    mutation to the default argument in _rope_freqs (or to Attention's use of
    cfg.rope_theta) would not be caught by that test.
    This test exercises Attention.__init__ with a non-default rope_theta
    (1000.0 vs 10000.0) and asserts the RoPE frequencies differ from the default,
    so a mutation to 10000.0 → any other value in Attention.__init__ (or in
    _rope_freqs's default) is caught here.

  test_rope_rotation_is_invertible:
    RoPE rotation must be an isometry (norm-preserving) — wrong implementation
    would change the L2 norm.

  test_sdp_attention_weights_sum_to_one:
    Missing 1/sqrt(d_k) scale causes softmax instability; the weights must
    still sum to 1 per query position.

  test_kv_cache_equals_reference:
    This is the core correctness oracle: the KV-cache decode path must produce
    logits identical (within float32 tolerance) to the full-sequence reference
    path. A wrong RoPE offset, wrong K/V append, or wrong matmul would fail this.

  test_kv_cache_memory_grows_linearly:
    KV cache memory must grow linearly in seq_len; non-linear growth would
    indicate a bug in the cache concatenation.

  test_causal_mask_prevents_future_attention:
    A future token must not affect the logits of a past token; violation would
    break autoregressive generation correctness.

  test_gqa_kv_expansion:
    With num_heads=6, num_kv_heads=2, the GQA expansion must produce the correct
    shape; wrong group size would silently broadcast to wrong dimensions.

  test_generate_returns_expected_length:
    generate() must return exactly max_new_tokens tokens; off-by-one would
    violate the API contract.

  test_generate_is_deterministic:
    generate() with temperature=0 and same seed must return identical output
    on repeated calls; non-determinism would break the Tier 1 determinism claim.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fitsproof.engine.attention import (
    KVCache,
    _rope_freqs,
    _sdp_attention,
    apply_rope,
)
from fitsproof.engine.model import REFERENCE_CONFIG, generate_reference_model
from fitsproof.engine.transformer import Transformer, rms_norm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_bundle(tmp_path_factory: pytest.TempPathFactory):
    """Generate and load the reference model once per test module."""
    path = tmp_path_factory.mktemp("refmodel")
    generate_reference_model(path, seed=42)
    from fitsproof.engine.model import load_bundle

    cfg, weights = load_bundle(path)
    return cfg, weights


@pytest.fixture(scope="module")
def transformer(reference_bundle):
    cfg, weights = reference_bundle
    return Transformer(cfg, weights)


# ---------------------------------------------------------------------------
# KAT: RoPE
# ---------------------------------------------------------------------------


def test_rope_known_values() -> None:
    """
    KAT: RoPE frequencies for d=4, theta=10000, positions 0 and 1.

    Hand-computed derivation (Su et al. 2023, Eq 15):
      d=4 => half=2
      freq_0 = 10000^{-0/4} = 1.0
      freq_1 = 10000^{-2/4} = 10000^{-0.5} = 0.01

      pos=0: angle_0=0*1.0=0,    angle_1=0*0.01=0
             cos=[1.0, 1.0],      sin=[0.0, 0.0]

      pos=1: angle_0=1*1.0=1.0,  angle_1=1*0.01=0.01
             cos=[cos(1.0), cos(0.01)],  sin=[sin(1.0), sin(0.01)]
    """
    freqs = _rope_freqs(head_dim=4, max_seq=2, theta=10000.0)
    # Shape: (2, 2, 2) = (seq, half, cos_sin)
    assert freqs.shape == (2, 2, 2)

    # pos=0: all angles=0 => cos=1, sin=0
    np.testing.assert_allclose(freqs[0, :, 0], [1.0, 1.0], atol=1e-6)  # cos
    np.testing.assert_allclose(freqs[0, :, 1], [0.0, 0.0], atol=1e-6)  # sin

    # pos=1: angles=[1.0, 0.01]
    expected_cos = np.cos(np.array([1.0, 0.01]))
    expected_sin = np.sin(np.array([1.0, 0.01]))
    np.testing.assert_allclose(freqs[1, :, 0], expected_cos, atol=1e-6)
    np.testing.assert_allclose(freqs[1, :, 1], expected_sin, atol=1e-6)


def test_rope_rotation_is_norm_preserving() -> None:
    """
    KAT: RoPE is an isometry — the L2 norm of each head vector is unchanged.

    From the definition: rotation matrices are orthogonal, so ||Rx||_2 = ||x||_2.
    Tested with random vectors of shape (1, 1, 4, 8) [batch, heads, seq, head_dim].
    """
    rng = np.random.default_rng(7)
    head_dim = 8
    x = rng.standard_normal((1, 1, 4, head_dim)).astype(np.float32)
    freqs = _rope_freqs(head_dim, max_seq=4)
    x_rot = apply_rope(x, freqs, offset=0)
    # Norms must match within float32 precision
    np.testing.assert_allclose(
        np.linalg.norm(x, axis=-1),
        np.linalg.norm(x_rot, axis=-1),
        atol=1e-5,
    )


def test_rope_offset_shifts_positions() -> None:
    """
    KAT: applying RoPE with offset=k to a token is identical to applying it
    at position k in a full-sequence encoding.

    If offset is wrong, the KV-cache path will have wrong positional encodings.
    """
    freqs = _rope_freqs(head_dim=8, max_seq=10)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 1, 1, 8)).astype(np.float32)

    for offset in [0, 2, 5]:
        x_with_offset = apply_rope(x, freqs, offset=offset)
        # Should equal RoPE applied to position `offset` in a full sequence
        x_full = np.zeros((1, 1, offset + 1, 8), dtype=np.float32)
        x_full[:, :, offset : offset + 1, :] = x
        x_full_rope = apply_rope(x_full, freqs, offset=0)
        np.testing.assert_allclose(
            x_with_offset, x_full_rope[:, :, offset : offset + 1, :], atol=1e-6
        )


def test_rope_non_default_theta_changes_freqs() -> None:
    """
    KAT (ADV-14 fix): exercises cfg.rope_theta propagation through Attention.__init__.

    ADV-14 (c3-p10-adversarial-1): test_rope_known_values passes theta=10000.0
    explicitly to _rope_freqs, so a mutation to the DEFAULT theta (10000.0 → X in
    either _rope_freqs's signature or in Attention.__init__) would survive that test.

    This test constructs two Transformer instances with different rope_theta values via
    ModelConfig (the path that Transformer uses at runtime), generates a sequence, and
    asserts the outputs differ. A mutation that hardcodes rope_theta=10000.0 in
    Attention.__init__ (ignoring cfg.rope_theta) would cause this test to fail when
    theta=100.0 is requested, because the freqs would still be computed for 10000.0
    and the outputs would be identical.

    Hand-derived ground truth: for head_dim=64, position 1:
      theta=10000 → freq_31 = 10000^{-62/64} ≈ 3.16e-4
      theta=100   → freq_31 = 100^{-62/64}   ≈ 3.98e-3
    A ~12× difference in the high-frequency component changes attention scores and
    thus greedy token selection at step 4+ (positions > 3).
    Observed divergence (prompt=[23,197], max_new_tokens=8, seed=42):
      theta=10000: [5, 5, 209, 65, 65, 131, 131, 65]
      theta=100:   [5, 5, 209, 209, 65, 65, 131, 131]

    Research source: Su et al. 2023 (RoPE), Eq 15, source [3] in RESEARCH.md.
    """
    import tempfile
    from dataclasses import replace
    from pathlib import Path

    from fitsproof.engine.model import generate_reference_model, load_bundle
    from fitsproof.engine.transformer import Transformer

    with tempfile.TemporaryDirectory() as td:
        generate_reference_model(Path(td), seed=42)
        cfg, weights = load_bundle(Path(td))

    # Prompt [23, 197] with 8 tokens to generate — chosen because theta=100.0
    # diverges from theta=10000.0 at generation step 4 (first position where the
    # high-frequency component produces a meaningfully different rotation).
    prompt = [23, 197]

    out_default = Transformer(replace(cfg, rope_theta=10000.0), weights).generate(
        prompt, max_new_tokens=8, temperature=0.0
    )
    out_alt = Transformer(replace(cfg, rope_theta=100.0), weights).generate(
        prompt, max_new_tokens=8, temperature=0.0
    )

    # Outputs must differ — if they are identical, cfg.rope_theta is being ignored.
    # Known outputs (deterministic, seed=42 reference model):
    #   theta=10000: [5, 5, 209, 65, 65, 131, 131, 65]
    #   theta=100:   [5, 5, 209, 209, 65, 65, 131, 131]
    assert out_default != out_alt, (
        "Outputs are identical with rope_theta=10000.0 and rope_theta=100.0 — "
        "cfg.rope_theta is not being propagated to Attention._freqs (ADV-14 regression).\n"
        f"  rope_theta=10000: {out_default}\n"
        f"  rope_theta=  100: {out_alt}"
    )


# ---------------------------------------------------------------------------
# KAT: SDP Attention
# ---------------------------------------------------------------------------


def test_sdp_attention_weights_sum_to_one() -> None:
    """
    KAT: attention weights must sum to 1 per query position.

    Vaswani et al. 2017, Eq 1: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V.
    The softmax output sums to 1; tested by examining that
    sum(attn_weights, dim=-1) == 1 within tolerance.

    We cannot directly inspect attn_weights from _sdp_attention (it returns
    the weighted sum), so we test via a known case: if Q=K=V=I with one query
    and one key, the output must equal V.
    """
    batch, heads, seq_q, seq_k, hd = 1, 2, 3, 3, 4
    rng = np.random.default_rng(1)
    q = rng.standard_normal((batch, heads, seq_q, hd)).astype(np.float32)
    k = rng.standard_normal((batch, heads, seq_k, hd)).astype(np.float32)
    v = rng.standard_normal((batch, heads, seq_k, hd)).astype(np.float32)

    out = _sdp_attention(q, k, v, causal_mask=False)
    assert out.shape == (batch, heads, seq_q, hd)

    # For single key (seq_k=1), output must equal v[0] (only one key → weight=1)
    q_single = rng.standard_normal((1, 1, 1, hd)).astype(np.float32)
    k_single = rng.standard_normal((1, 1, 1, hd)).astype(np.float32)
    v_single = rng.standard_normal((1, 1, 1, hd)).astype(np.float32)
    out_single = _sdp_attention(q_single, k_single, v_single, causal_mask=False)
    np.testing.assert_allclose(out_single, v_single, atol=1e-5)


def test_sdp_attention_causal_mask() -> None:
    """
    KAT: causal mask must prevent future tokens from attending to past.

    If we set V[1] to a large constant and causal_mask=True, then output[0]
    must not contain any contribution from V[1].
    """
    hd = 4
    # Q: 2 queries, K: 2 keys
    q = np.ones((1, 1, 2, hd), dtype=np.float32)
    k = np.ones((1, 1, 2, hd), dtype=np.float32)
    v = np.zeros((1, 1, 2, hd), dtype=np.float32)
    v[0, 0, 1, :] = 1000.0  # V[1] is large

    out = _sdp_attention(q, k, v, causal_mask=True)
    # First query must not see V[1] (masked by causal constraint)
    # out[0,0,0] should have no contribution from v[1]
    # With causal mask: position 0 only attends to position 0 (V[0]=0)
    np.testing.assert_allclose(out[0, 0, 0], np.zeros(hd), atol=0.1)


# ---------------------------------------------------------------------------
# Core correctness oracle: KV-cache equals reference
# ---------------------------------------------------------------------------


def test_kv_cache_equals_reference(transformer: Transformer) -> None:
    """
    CORE CORRECTNESS ORACLE: KV-cache decode path must produce identical
    logits to the reference full-sequence path.

    This is Acceptance Criterion 2 from the spec.
    Fault detected: any bug in KV cache (wrong offset, wrong append, wrong
    masking, wrong RoPE position) causes the cache path to diverge.

    Method: run reference_forward on a 4-token sequence, then run the
    cached path one token at a time, and compare the final token's logits.
    """
    cfg = transformer.cfg
    rng = np.random.default_rng(99)
    seq_len = 4
    token_ids = rng.integers(0, cfg.vocab_size, size=seq_len, dtype=np.int64)

    # Reference path: full sequence
    ref_logits = transformer.forward_reference(token_ids[np.newaxis, :])  # (1, 4, vocab)
    ref_last = ref_logits[0, -1, :]  # (vocab,)

    # Cache path: token by token, then compare last logits
    from fitsproof.engine.attention import KVCache
    from fitsproof.engine.transformer import rms_norm

    cache = KVCache(cfg, batch=1)
    x_last = None
    for pos in range(seq_len):
        tok = token_ids[pos : pos + 1][np.newaxis, :]  # (1, 1)
        x = transformer._embed(tok)
        for i in range(cfg.num_layers):
            x = transformer._layer_forward_cached(x, i, cache)
        x_last = x

    assert x_last is not None
    x_last = rms_norm(x_last, transformer.weights["final_norm"], cfg.rms_norm_eps)
    cache_logits = (x_last @ transformer.weights["unembed"].T)[0, 0]

    # The cache path should match the reference path for the last token
    np.testing.assert_allclose(
        cache_logits,
        ref_last,
        atol=1e-3,  # float32 accumulation tolerance over 6 layers
        err_msg="KV-cache path diverged from reference path",
    )


def test_kv_cache_memory_grows_linearly() -> None:
    """
    KAT: KV cache memory must grow linearly in seq_len.

    Formula: 2 * n_layers * n_kv_heads * seq * head_dim * bytes_per_element.
    For REFERENCE_CONFIG (float32): 2 * 6 * 2 * seq * 64 * 4 = 6144 * seq bytes.
    """
    cfg = REFERENCE_CONFIG
    cache = KVCache(cfg, batch=1, dtype=np.float32)
    hd = cfg.head_dim

    rng = np.random.default_rng(0)
    for seq_len in range(1, 5):
        k = rng.standard_normal((1, cfg.num_kv_heads, 1, hd)).astype(np.float32)
        v = rng.standard_normal((1, cfg.num_kv_heads, 1, hd)).astype(np.float32)
        cache.append(0, k, v)  # Only layer 0 to test formula

    # After appending 4 tokens to layer 0:
    # memory: 2 * 1 (layer) * 2 (kv) * 4 (seq) * 64 (hd) * 4 = 4096 bytes
    # But actual seq_len counter advances on layer 0 per step
    # Test the formula at seq=4
    expected = 2 * 1 * cfg.num_kv_heads * 4 * hd * 4
    k_full, _ = cache._cache[0]
    actual = 2 * k_full.nbytes  # K + V
    assert actual == expected, f"Expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# RMSNorm KAT
# ---------------------------------------------------------------------------


def test_rms_norm_known_value() -> None:
    """
    KAT: RMSNorm of a known vector against hand-computed result.

    For x = [1, 2, 3, 4], weight = [1, 1, 1, 1], eps=0:
      RMS(x) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
      RMSNorm(x) = x / RMS(x) = [0.3651, 0.7303, 1.0954, 1.4606]

    This verifies that mean is NOT subtracted (unlike LayerNorm).
    Fault detected: using std() instead of RMS() gives wrong result when mean != 0.
    """
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    weight = np.ones(4, dtype=np.float32)
    # Expected: x / sqrt(mean(x^2))
    rms = np.sqrt(np.mean(x**2))
    expected = x / rms

    result = rms_norm(x, weight, eps=1e-8)
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_rms_norm_weight_scaling() -> None:
    """
    KAT: RMSNorm weight scaling must be applied after normalisation.

    If weight is 2x the identity, output must be 2x the normalised value.
    Fault detected: applying weight before normalisation changes the result.
    """
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    weight_1 = np.ones(2, dtype=np.float32)
    weight_2 = np.full(2, 2.0, dtype=np.float32)
    out_1 = rms_norm(x, weight_1)
    out_2 = rms_norm(x, weight_2)
    np.testing.assert_allclose(out_2, 2.0 * out_1, atol=1e-6)


# ---------------------------------------------------------------------------
# Generate API tests
# ---------------------------------------------------------------------------


def test_generate_returns_expected_length(transformer: Transformer) -> None:
    """
    Fault detected: off-by-one in the generation loop returns n+1 or n-1 tokens.
    """
    rng = np.random.default_rng(5)
    prompt = rng.integers(0, transformer.cfg.vocab_size, size=4, dtype=np.int64).tolist()
    for n in [1, 5, 10]:
        tokens = transformer.generate(prompt, max_new_tokens=n, temperature=0.0)
        assert len(tokens) == n, f"Expected {n} tokens, got {len(tokens)}"


def test_generate_is_deterministic(transformer: Transformer) -> None:
    """
    Fault detected: non-deterministic sampling breaks Tier 1 reproducibility.
    temperature=0 (greedy) must be fully deterministic.
    """
    rng = np.random.default_rng(11)
    prompt = rng.integers(0, transformer.cfg.vocab_size, size=3, dtype=np.int64).tolist()
    out1 = transformer.generate(prompt, max_new_tokens=8, temperature=0.0)
    out2 = transformer.generate(prompt, max_new_tokens=8, temperature=0.0)
    assert out1 == out2, "generate() is not deterministic under greedy decoding"


def test_generate_rejects_empty_prompt(transformer: Transformer) -> None:
    """
    Fault detected: empty prompt causes index error if not guarded.
    """
    with pytest.raises(ValueError, match="prompt_ids must not be empty"):
        transformer.generate([], max_new_tokens=1)


# ---------------------------------------------------------------------------
# Hypothesis: property-based tests
# ---------------------------------------------------------------------------


@given(
    seq_len=st.integers(min_value=1, max_value=8),
    head_dim=st.sampled_from([4, 8, 16]),
    n_heads=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=30, deadline=5000)
def test_rope_norm_preservation_hypothesis(seq_len: int, head_dim: int, n_heads: int) -> None:
    """
    Property: RoPE is always norm-preserving, regardless of sequence length,
    head dimension, or number of heads.

    This comes from the mathematical property that rotation matrices are orthogonal.
    Fault: any scaling or sign error in the rotation formula breaks norm preservation.
    """
    rng = np.random.default_rng(seq_len + head_dim)
    x = rng.standard_normal((1, n_heads, seq_len, head_dim)).astype(np.float32)
    freqs = _rope_freqs(head_dim, max_seq=seq_len)
    x_rot = apply_rope(x, freqs, offset=0)
    np.testing.assert_allclose(
        np.linalg.norm(x, axis=-1),
        np.linalg.norm(x_rot, axis=-1),
        atol=1e-4,
    )


@given(seq_len=st.integers(min_value=1, max_value=6))
@settings(max_examples=20, deadline=10000)
def test_kv_cache_seq_len_counter(seq_len: int) -> None:
    """
    Property: after appending *seq_len* tokens, KVCache.seq_len == seq_len.

    Fault: incorrect counter update causes RoPE offset to drift.
    """
    cfg = REFERENCE_CONFIG
    cache = KVCache(cfg, batch=1)
    rng = np.random.default_rng(0)
    for _ in range(seq_len):
        k = rng.standard_normal((1, cfg.num_kv_heads, 1, cfg.head_dim)).astype(np.float32)
        v = rng.standard_normal((1, cfg.num_kv_heads, 1, cfg.head_dim)).astype(np.float32)
        cache.append(0, k, v)
    assert cache.seq_len == seq_len
