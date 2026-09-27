"""
fitsproof.engine.sampling — Token sampling with seeded RNG.

Implements greedy, temperature, top-k, and top-p (nucleus) sampling.
The seeded RNG makes generation deterministic and reproducible.

Sources:
  - Holtzman et al. 2020 (The Curious Case of Neural Text Degeneration),
    https://arxiv.org/abs/1904.09751  — top-p nucleus sampling
  - Fan et al. 2018 (Hierarchical Neural Story Generation),
    https://arxiv.org/abs/1805.04833  — top-k sampling
"""

from __future__ import annotations

import numpy as np


class Sampler:
    """
    Token sampler with a fixed seeded RNG for reproducibility.

    Fault detected by tests: using np.random.choice without seeding produces
    different outputs across runs; the seeded Sampler must return the same
    token sequence for the same logits and seed.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int = 0) -> None:
        """Reset the RNG to a known state."""
        self._rng = np.random.default_rng(seed)

    def greedy(self, logits: np.ndarray) -> int:
        """
        Greedy decoding: always select the highest-probability token.

        logits: (vocab_size,) — raw unnormalised scores
        Returns: int token id

        Fault detected: if argmax is not applied, the sampler would return
        a low-probability token and break the speculative-decoding equality test.
        """
        return int(np.argmax(logits))

    def temperature_sample(self, logits: np.ndarray, temperature: float) -> int:
        """
        Temperature sampling: divide logits by T before softmax.

        temperature > 1: flatter distribution (more random)
        temperature < 1: sharper distribution (more greedy)
        temperature = 0: equivalent to greedy (handled specially)

        logits: (vocab_size,)
        Returns: int token id

        Fault detected: temperature=0 must use argmax, not softmax
        (softmax of huge values overflows to NaN).
        """
        if temperature == 0.0:
            return self.greedy(logits)
        scaled = logits.astype(np.float64) / float(temperature)
        # Numerically stable softmax
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        return int(self._rng.choice(len(probs), p=probs))

    def top_k_sample(self, logits: np.ndarray, k: int, temperature: float = 1.0) -> int:
        """
        Top-k sampling: sample only from the k highest-logit tokens.

        k: number of candidates; if k <= 0 or k >= vocab_size, falls back to full distribution.

        Fault detected: if we fail to mask non-top-k tokens, the sampler
        samples from the full vocabulary and the distribution is wrong.
        """
        vocab = len(logits)
        if k <= 0 or k >= vocab:
            return self.temperature_sample(logits, temperature)
        # Find top-k indices
        top_indices = np.argpartition(logits, -k)[-k:]
        masked = np.full(vocab, -np.inf, dtype=np.float64)
        masked[top_indices] = logits[top_indices]
        return self.temperature_sample(masked, temperature)

    def top_p_sample(self, logits: np.ndarray, p: float, temperature: float = 1.0) -> int:
        """
        Nucleus (top-p) sampling from Holtzman et al. 2020.

        Keeps the smallest set of tokens whose cumulative probability >= p.
        If p >= 1.0, samples from the full distribution.

        logits: (vocab_size,)
        p: float in (0, 1]

        Fault detected: if tokens are not sorted by probability before computing
        cumsum, the nucleus may include wrong tokens.
        """
        if p >= 1.0:
            return self.temperature_sample(logits, temperature)

        if temperature == 0.0:
            return self.greedy(logits)

        scaled = logits.astype(np.float64) / float(temperature)
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()

        # Sort descending by probability
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)

        # Keep tokens until cumsum >= p; always include at least one
        cutoff = int(np.searchsorted(cumsum, p)) + 1
        nucleus_indices = sorted_indices[:cutoff]

        nucleus_probs = probs[nucleus_indices]
        nucleus_probs /= nucleus_probs.sum()  # renormalise

        return int(self._rng.choice(nucleus_indices, p=nucleus_probs))

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> int:
        """
        Unified sampling entry point.

        Applies top-k first, then top-p, then temperature.
        temperature=0 short-circuits to greedy regardless of top-k/p.

        Fault detected: wrong order of top-k/top-p filtering changes
        the effective nucleus size.
        """
        if temperature == 0.0:
            return self.greedy(logits)
        if top_k > 0:
            return self.top_k_sample(logits, top_k, temperature)
        if top_p < 1.0:
            return self.top_p_sample(logits, top_p, temperature)
        return self.temperature_sample(logits, temperature)
