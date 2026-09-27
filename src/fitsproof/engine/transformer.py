"""
fitsproof.engine.transformer — Full transformer forward pass.

Assembles attention, FFN, and RMSNorm into a complete autoregressive
transformer. Supports both a full-sequence reference pass and an
incremental cached-decode pass.

The reference model shipped in-repo (6 layers, 384 hidden, 6Q/2KV heads,
vocab 256) is the primary test fixture for all engine tests.
"""

from __future__ import annotations

import numpy as np

from fitsproof.engine.attention import AttentionLayer, KVCache
from fitsproof.engine.model import ModelConfig
from fitsproof.engine.sampling import Sampler


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Root-mean-square layer norm (Zhang & Sennrich 2019).

    RMSNorm(x) = x / RMS(x) * weight,  where RMS(x) = sqrt(mean(x^2) + eps)

    No learnable bias (by design, following LLaMA architecture).
    Faster than LayerNorm because it omits mean subtraction.

    Fault detected: using std instead of RMS incorrectly subtracts the mean
    before normalisation; tested with a vector where mean != 0.
    """
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


# ---------------------------------------------------------------------------
# Feed-forward network (SwiGLU)
# ---------------------------------------------------------------------------


def swiglu_ffn(
    x: np.ndarray,
    gate_w: np.ndarray,
    up_w: np.ndarray,
    down_w: np.ndarray,
) -> np.ndarray:
    """
    SwiGLU feed-forward network (Shazeer 2020 / LLaMA architecture).

    FFN(x) = (SiLU(x @ gate_w.T) * (x @ up_w.T)) @ down_w.T

    SiLU(z) = z * sigmoid(z)  (Swish activation)

    This is the gated variant: the gate path controls information flow
    into the up-projection via element-wise multiplication.

    Fault detected: using ReLU instead of SiLU changes the gating
    function non-trivially (no negative values pass through).
    """
    gate = x @ gate_w.T  # (..., intermediate)
    up = x @ up_w.T      # (..., intermediate)
    # SiLU: z * sigmoid(z)
    silu_gate = gate * (1.0 / (1.0 + np.exp(-gate)))
    return (silu_gate * up) @ down_w.T  # (..., hidden)


# ---------------------------------------------------------------------------
# Full transformer
# ---------------------------------------------------------------------------


class Transformer:
    """
    Complete autoregressive transformer implementing the fitsproof engine.

    Supports:
      - forward_reference: full-sequence pass (used for KAT ground truth)
      - generate: incremental cached-decode generation loop

    Weights are loaded from a fitsproof model bundle.
    """

    def __init__(self, cfg: ModelConfig, weights: dict[str, np.ndarray]) -> None:
        self.cfg = cfg
        self.weights = weights
        self._layers: list[AttentionLayer] = []
        for i in range(cfg.num_layers):
            p = f"layer_{i}"
            self._layers.append(
                AttentionLayer(
                    cfg=cfg,
                    q_w=weights[f"{p}.q_proj"],
                    k_w=weights[f"{p}.k_proj"],
                    v_w=weights[f"{p}.v_proj"],
                    o_w=weights[f"{p}.o_proj"],
                )
            )

    def _embed(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Look up token embeddings.

        token_ids: (batch, seq_len) int32/int64
        Returns: (batch, seq_len, hidden_size)
        """
        return self.weights["embed"][token_ids]

    def _layer_forward_ref(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """Single transformer layer (reference path, full sequence)."""
        p = f"layer_{layer_idx}"
        # Pre-norm attention
        norm_x = rms_norm(x, self.weights[f"{p}.attn_norm"], self.cfg.rms_norm_eps)
        attn_out = self._layers[layer_idx].forward_reference(norm_x)
        x = x + attn_out
        # Pre-norm FFN
        norm_x2 = rms_norm(x, self.weights[f"{p}.ffn_norm"], self.cfg.rms_norm_eps)
        ffn_out = swiglu_ffn(
            norm_x2,
            self.weights[f"{p}.gate_proj"],
            self.weights[f"{p}.up_proj"],
            self.weights[f"{p}.down_proj"],
        )
        return x + ffn_out

    def _layer_forward_cached(
        self,
        x: np.ndarray,
        layer_idx: int,
        cache: KVCache,
    ) -> np.ndarray:
        """Single transformer layer (cached incremental path, single token)."""
        p = f"layer_{layer_idx}"
        norm_x = rms_norm(x, self.weights[f"{p}.attn_norm"], self.cfg.rms_norm_eps)
        attn_out = self._layers[layer_idx].forward_cached(norm_x, cache, layer_idx)
        x = x + attn_out
        norm_x2 = rms_norm(x, self.weights[f"{p}.ffn_norm"], self.cfg.rms_norm_eps)
        ffn_out = swiglu_ffn(
            norm_x2,
            self.weights[f"{p}.gate_proj"],
            self.weights[f"{p}.up_proj"],
            self.weights[f"{p}.down_proj"],
        )
        return x + ffn_out

    def forward_reference(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Full-sequence reference forward pass.

        token_ids: (batch, seq_len) int64
        Returns logits: (batch, seq_len, vocab_size)

        This is the correctness oracle for the KV-cache path.
        """
        x = self._embed(token_ids)  # (batch, seq, hidden)
        for i in range(self.cfg.num_layers):
            x = self._layer_forward_ref(x, i)
        x = rms_norm(x, self.weights["final_norm"], self.cfg.rms_norm_eps)
        return x @ self.weights["unembed"].T

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 20,
        sampler: Sampler | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> list[int]:
        """
        Autoregressive token generation using the KV cache.

        Returns the list of generated token ids (not including prompt).

        Uses cached incremental forward for decode steps (O(seq) per step
        rather than O(seq^2)).

        Fault detected: if the cache seq_len counter is wrong, RoPE offsets
        diverge from reference and outputs differ; caught by the equality test.
        """
        if sampler is None:
            sampler = Sampler(seed=0)

        cache = KVCache(self.cfg, batch=1)

        # Prefill: run reference forward over the full prompt
        if len(prompt_ids) == 0:
            raise ValueError("prompt_ids must not be empty")

        prompt = np.array(prompt_ids, dtype=np.int64)[np.newaxis, :]  # (1, seq)
        logits = self.forward_reference(prompt)  # (1, seq, vocab)
        # Prime the cache with all prompt tokens' K/V by re-running cached path
        # Reset and replay incrementally to populate cache correctly
        cache.reset()
        for pos in range(len(prompt_ids)):
            tok = np.array([[prompt_ids[pos]]], dtype=np.int64)  # (1, 1)
            x = self._embed(tok)
            for i in range(self.cfg.num_layers):
                x = self._layer_forward_cached(x, i, cache)

        # The last logits from the cached prefill drive the first new token
        x_last = self._embed(np.array([[prompt_ids[-1]]], dtype=np.int64))
        # Re-derive last hidden state from the cache path
        # We already have cache populated; logits for last position come from
        # the final cached hidden state
        cache_for_decode = KVCache(self.cfg, batch=1)
        for pos in range(len(prompt_ids)):
            tok = np.array([[prompt_ids[pos]]], dtype=np.int64)
            x = self._embed(tok)
            for i in range(self.cfg.num_layers):
                x = self._layer_forward_cached(x, i, cache_for_decode)
        x = rms_norm(x, self.weights["final_norm"], self.cfg.rms_norm_eps)
        last_logits = (x @ self.weights["unembed"].T)[0, 0]  # (vocab,)

        generated: list[int] = []
        current_logits = last_logits

        for _ in range(max_new_tokens):
            next_token = sampler.sample(current_logits, temperature, top_k, top_p)
            generated.append(next_token)
            # Decode step
            tok = np.array([[next_token]], dtype=np.int64)
            x = self._embed(tok)
            for i in range(self.cfg.num_layers):
                x = self._layer_forward_cached(x, i, cache_for_decode)
            x = rms_norm(x, self.weights["final_norm"], self.cfg.rms_norm_eps)
            current_logits = (x @ self.weights["unembed"].T)[0, 0]

        return generated
