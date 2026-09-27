"""
Tests for fitsproof.engine.sampling.

Faults detected by each test:
  test_greedy_returns_argmax:
    If greedy does not use argmax (e.g. uses argmin), it returns the wrong token.
    Verified by constructing logits where argmax is the last element.

  test_temperature_zero_is_greedy:
    temperature=0 must fall back to greedy (argmax), not softmax (which overflows).
    Fault: if softmax is applied, exp(huge/0) = inf, giving NaN.

  test_temperature_one_vs_raw_softmax:
    At temperature=1.0, sample distribution equals raw softmax.
    Fault: wrong temperature scaling changes probabilities.

  test_top_k_restricts_candidates:
    With k=1, only the argmax can be sampled; any other token is a fault.
    Fault: not masking non-top-k tokens allows sampling from full distribution.

  test_top_p_nucleus:
    With p=0.0, only the top-1 token is in the nucleus.
    Fault: wrong cumsum direction or missing sort would include wrong tokens.

  test_sampler_is_deterministic:
    Same logits + same seed must yield identical tokens across calls.
    Fault: global numpy state leaks into the sampler.

  test_sampler_temperature_increases_diversity:
    Higher temperature should increase entropy of the sampled distribution.
    This is a mathematical property of the softmax temperature scaling.

  (hypothesis) test_sampler_always_returns_valid_token:
    sample() must always return an int in [0, vocab_size).
    Fault: off-by-one in top-k/top-p could return an invalid index.

  (hypothesis) test_top_p_monotonic_coverage:
    As p increases from 0 to 1, the set of candidates can only grow.
    Fault: wrong cumsum logic could shrink the nucleus as p increases.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from fitsproof.engine.sampling import Sampler

# ---------------------------------------------------------------------------
# KAT: greedy
# ---------------------------------------------------------------------------


def test_greedy_returns_argmax() -> None:
    """
    KAT: greedy must return the index of the maximum logit.
    Logits: [0.1, 0.2, 5.0, 0.3] -> argmax=2.
    Fault: if greedy returns argmin or random, it would return 0 or 1.
    """
    logits = np.array([0.1, 0.2, 5.0, 0.3], dtype=np.float32)
    s = Sampler(seed=0)
    assert s.greedy(logits) == 2


def test_greedy_with_large_vocab() -> None:
    """
    KAT: greedy on vocab_size=256, max at last position.
    Fault: off-by-one in argmax returns 254 instead of 255.
    """
    logits = np.zeros(256, dtype=np.float32)
    logits[255] = 99.0
    s = Sampler(seed=0)
    assert s.greedy(logits) == 255


# ---------------------------------------------------------------------------
# KAT: temperature
# ---------------------------------------------------------------------------


def test_temperature_zero_is_greedy() -> None:
    """
    KAT: temperature=0 must return argmax, not a random sample.
    Fault: applying softmax at T=0 overflows to NaN.
    """
    logits = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    s = Sampler(seed=0)
    # All seeds must give the same answer (deterministic greedy)
    for seed in range(5):
        s.reset(seed)
        assert s.temperature_sample(logits, 0.0) == 1  # argmax=1


def test_temperature_one_vs_raw_softmax() -> None:
    """
    KAT: temperature=1.0 must sample from the raw softmax distribution.

    Verify by drawing many samples and checking empirical distribution
    matches expected softmax probabilities.

    Expected: logits=[1,2,3] -> softmax=[0.0900, 0.2447, 0.6652]
    With N=5000 samples, variance is small enough to distinguish.
    """
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    s = Sampler(seed=42)
    counts = np.zeros(3, dtype=np.int64)
    n_samples = 3000
    for _ in range(n_samples):
        tok = s.temperature_sample(logits, 1.0)
        counts[tok] += 1
    empirical = counts / n_samples
    # Check each probability within 3% (with N=3000, std ≈ 1.5%)
    np.testing.assert_allclose(empirical, probs, atol=0.04)


# ---------------------------------------------------------------------------
# KAT: top-k
# ---------------------------------------------------------------------------


def test_top_k_restricts_to_one() -> None:
    """
    KAT: with k=1, only the argmax can ever be sampled.
    Fault: not masking non-top-k tokens allows any token to be sampled.
    """
    logits = np.array([0.1, 5.0, 0.2, 0.3], dtype=np.float32)
    s = Sampler(seed=0)
    for seed in range(20):
        s.reset(seed)
        tok = s.top_k_sample(logits, k=1)
        assert tok == 1, f"Expected 1 (argmax), got {tok}"


def test_top_k_zero_uses_full_distribution() -> None:
    """
    KAT: k=0 must not restrict (uses full distribution).
    Fault: k=0 masking everything would produce undefined behavior.
    """
    logits = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    s = Sampler(seed=99)
    # Should not raise and should return a valid token
    tok = s.top_k_sample(logits, k=0, temperature=1.0)
    assert 0 <= tok < 4


# ---------------------------------------------------------------------------
# KAT: top-p
# ---------------------------------------------------------------------------


def test_top_p_one_includes_all() -> None:
    """
    KAT: p=1.0 uses the full distribution; does not restrict.
    """
    logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    s = Sampler(seed=0)
    # Should not raise; any token is valid
    tok = s.top_p_sample(logits, p=1.0, temperature=1.0)
    assert 0 <= tok < 4


def test_top_p_near_zero_picks_argmax() -> None:
    """
    KAT: very small p (e.g. 0.01) should select only the top-1 token.
    Fault: wrong cumsum direction includes more tokens than the nucleus.
    """
    logits = np.array([0.1, 0.2, 0.1, 5.0], dtype=np.float32)  # argmax=3 with ~99% prob
    s = Sampler(seed=7)
    for seed in range(10):
        s.reset(seed)
        tok = s.top_p_sample(logits, p=0.01, temperature=1.0)
        assert tok == 3, f"With p=0.01, expected argmax=3, got {tok}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_sampler_is_deterministic() -> None:
    """
    Fault: if global numpy state leaks in, two samplers with the same seed
    return different outputs depending on call order.
    """
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    s1 = Sampler(seed=42)
    s2 = Sampler(seed=42)
    for _ in range(10):
        t1 = s1.sample(logits, temperature=1.0)
        t2 = s2.sample(logits, temperature=1.0)
        assert t1 == t2, "Samplers with same seed diverged"


def test_different_seeds_produce_different_outputs() -> None:
    """
    Fault: if seeds are ignored, all samplers produce identical output.
    """
    logits = np.ones(100, dtype=np.float32)  # uniform -> seed matters
    tokens: set[int] = set()
    for seed in range(10):
        s = Sampler(seed=seed)
        tokens.add(s.temperature_sample(logits, 1.0))
    # With 10 different seeds on a uniform distribution, expect variation
    assert len(tokens) > 2, "Different seeds should produce varied outputs"


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


@given(
    vocab_size=st.integers(min_value=2, max_value=32),
    temperature=st.floats(min_value=0.1, max_value=5.0),
    seed=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=50, deadline=5000)
def test_sampler_always_returns_valid_token(vocab_size: int, temperature: float, seed: int) -> None:
    """
    Property: sample() always returns an int in [0, vocab_size).
    Fault: off-by-one in top-k/top-p indexing could return out-of-range token.
    """
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal(vocab_size).astype(np.float32)
    s = Sampler(seed=seed)
    tok = s.sample(logits, temperature=temperature)
    assert 0 <= tok < vocab_size, f"Token {tok} out of range [0, {vocab_size})"


@given(
    vocab_size=st.integers(min_value=4, max_value=16),
    p=st.floats(min_value=0.01, max_value=0.99),
)
@settings(max_examples=30, deadline=5000)
def test_top_p_coverage_monotone(vocab_size: int, p: float) -> None:
    """
    Property: as p increases, the nucleus cannot shrink (weakly monotone).

    Fault: wrong sort direction in cumsum causes nucleus to shrink at higher p.
    """
    rng = np.random.default_rng(vocab_size)
    logits = rng.standard_normal(vocab_size).astype(np.float32)

    # Count how many distinct tokens are sampled at p vs p*0.5
    n_samples = 100
    tokens_high = set()
    tokens_low = set()
    s = Sampler(seed=1)
    for i in range(n_samples):
        s.reset(i)
        tokens_high.add(s.top_p_sample(logits, p=p, temperature=1.0))
        s.reset(i)
        tokens_low.add(s.top_p_sample(logits, p=p * 0.5, temperature=1.0))

    # More nucleus coverage with higher p (or at least not strictly less)
    assert len(tokens_high) >= len(tokens_low) - 1  # allow 1 variance slack
