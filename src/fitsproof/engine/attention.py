"""
fitsproof.engine.attention — Multi-head attention with GQA, RoPE, and KV cache.

Reference path: computes attention from scratch at every step (correct, slow).
Cache path: appends the current token's K/V to a growing cache and attends over it.

The correctness oracle is:
    reference_forward(tokens[:t+1]) == cache_forward_step(token_t, cache_state)

This is tested directly in tests/engine/test_attention.py.

Mathematical derivations are in docs/IMPLEMENTATION-NOTES.md.

Sources:
  - Vaswani et al. 2017 (Attention Is All You Need), Equation 1
  - Su et al. 2023 (RoPE), https://arxiv.org/abs/2104.09864
  - Ainslie et al. 2023 (GQA), https://arxiv.org/abs/2305.13245
"""

from __future__ import annotations

import numpy as np

from fitsproof.engine.model import ModelConfig


# ---------------------------------------------------------------------------
# RoPE positional encoding
# ---------------------------------------------------------------------------


def _rope_freqs(head_dim: int, max_seq: int, theta: float = 10000.0) -> np.ndarray:
    """
    Pre-compute RoPE rotation frequencies.

    From Su et al. 2023 (arXiv 2104.09864), Equation 15:
        theta_i = base^{-2i/d}  for i in [0, d/2)

    Returns shape (max_seq, head_dim/2, 2) — (cos, sin) pairs per position.

    Fault detected: wrong theta causes positional aliasing; tested against
    hand-computed values for d=4, positions 0 and 1.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    # Frequency for each dimension pair: shape (half,)
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    # Position indices: shape (max_seq,)
    positions = np.arange(max_seq, dtype=np.float32)
    # Outer product: shape (max_seq, half)
    angles = np.outer(positions, inv_freq)
    # Stack cos and sin: shape (max_seq, half, 2)
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def apply_rope(x: np.ndarray, freqs: np.ndarray, offset: int = 0) -> np.ndarray:
    """
    Apply rotary position encoding to query or key tensor.

    Args:
        x:      shape (..., seq_len, head_dim)
        freqs:  shape (max_seq, head_dim/2, 2) as returned by _rope_freqs
        offset: position offset for KV-cache incremental decode

    Returns array of same shape as x.

    Implementation of rotation formula from Su et al. 2023, Eq 34:
        [x_{2i}, x_{2i+1}] -> [x_{2i}*cos - x_{2i+1}*sin,
                                x_{2i}*sin + x_{2i+1}*cos]

    Fault detected: wrong rotation direction causes position-dependent
    dot products to differ from reference by a sign flip.
    """
    *batch, seq_len, head_dim = x.shape
    half = head_dim // 2
    # Select positions for this chunk
    pos_freqs = freqs[offset : offset + seq_len]  # (seq_len, half, 2)
    cos_vals = pos_freqs[..., 0]  # (seq_len, half)
    sin_vals = pos_freqs[..., 1]  # (seq_len, half)

    x_even = x[..., 0::2]  # (..., seq_len, half)
    x_odd = x[..., 1::2]   # (..., seq_len, half)

    out_even = x_even * cos_vals - x_odd * sin_vals
    out_odd = x_even * sin_vals + x_odd * cos_vals

    # Interleave back: alternating even/odd
    out = np.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


# ---------------------------------------------------------------------------
# Scaled dot-product attention
# ---------------------------------------------------------------------------


def _sdp_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal_mask: bool = True,
) -> np.ndarray:
    """
    Scaled dot-product attention (Vaswani et al. 2017, Eq. 1):
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        q: shape (batch, n_heads, seq_q, head_dim)
        k: shape (batch, n_kv_heads, seq_k, head_dim)  [possibly fewer heads for GQA]
        v: shape (batch, n_kv_heads, seq_k, head_dim)
        causal_mask: if True, apply autoregressive causal mask

    Returns shape (batch, n_heads, seq_q, head_dim).

    Fault detected: missing 1/sqrt(d_k) scale causes numerically unstable softmax;
    tested by checking that attention weights sum to 1.0 within tolerance.
    """
    batch, n_q, seq_q, hd = q.shape
    _, n_kv, seq_k, _ = k.shape
    scale = 1.0 / np.sqrt(hd)

    # GQA: expand KV heads to match Q heads
    # Each KV head is shared across (n_q // n_kv) query heads
    if n_kv < n_q:
        group = n_q // n_kv
        # (batch, n_kv, seq_k, hd) -> (batch, n_kv, 1, seq_k, hd)
        k_exp = k[:, :, np.newaxis, :, :]  # (batch, n_kv, 1, seq_k, hd)
        v_exp = v[:, :, np.newaxis, :, :]
        # Broadcast to (batch, n_kv, group, seq_k, hd) then reshape
        k_exp = np.broadcast_to(k_exp, (batch, n_kv, group, seq_k, hd)).reshape(
            batch, n_q, seq_k, hd
        )
        v_exp = np.broadcast_to(v_exp, (batch, n_kv, group, seq_k, hd)).reshape(
            batch, n_q, seq_k, hd
        )
    else:
        k_exp = k
        v_exp = v

    # Attention scores: (batch, n_q, seq_q, seq_k)
    scores = np.matmul(q, k_exp.transpose(0, 1, 3, 2)) * scale  # (..., seq_q, seq_k)

    if causal_mask:
        # Lower-triangular mask: position i may attend to j <= i
        mask = np.tril(np.ones((seq_q, seq_k), dtype=np.float32))
        # Extend mask from top-right if seq_k > seq_q (cache scenario)
        if seq_k > seq_q:
            # Query tokens attend to all past keys
            mask = np.ones((seq_q, seq_k), dtype=np.float32)
            for qi in range(seq_q):
                # qi-th query can see up to position (seq_k - seq_q + qi)
                mask[qi, seq_k - seq_q + qi + 1 :] = 0.0
        scores = np.where(mask[np.newaxis, np.newaxis] == 0, -1e9, scores)

    # Softmax over last dim (keys)
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn_weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)

    # Weighted sum: (batch, n_q, seq_q, hd)
    return np.matmul(attn_weights, v_exp)


# ---------------------------------------------------------------------------
# Full attention layer (reference path — recomputes from full sequence)
# ---------------------------------------------------------------------------


class AttentionLayer:
    """
    Single transformer attention layer implementing MHA/GQA with RoPE.

    Carries its own weight tensors; stateless (no KV cache stored here).
    The cache is managed externally via KVCache objects.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        q_w: np.ndarray,
        k_w: np.ndarray,
        v_w: np.ndarray,
        o_w: np.ndarray,
    ) -> None:
        self.cfg = cfg
        self.q_w = q_w  # (n_heads*head_dim, hidden)
        self.k_w = k_w  # (n_kv_heads*head_dim, hidden)
        self.v_w = v_w  # (n_kv_heads*head_dim, hidden)
        self.o_w = o_w  # (hidden, n_heads*head_dim)
        self._freqs = _rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def _project(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project hidden states to Q, K, V tensors.

        x: (batch, seq_len, hidden)
        Returns Q (batch, n_heads, seq, head_dim),
                K (batch, n_kv_heads, seq, head_dim),
                V (batch, n_kv_heads, seq, head_dim).
        """
        batch, seq, _ = x.shape
        hd = self.cfg.head_dim

        q = (x @ self.q_w.T).reshape(batch, seq, self.cfg.num_heads, hd).transpose(0, 2, 1, 3)
        k = (x @ self.k_w.T).reshape(batch, seq, self.cfg.num_kv_heads, hd).transpose(0, 2, 1, 3)
        v = (x @ self.v_w.T).reshape(batch, seq, self.cfg.num_kv_heads, hd).transpose(0, 2, 1, 3)
        return q, k, v

    def forward_reference(self, x: np.ndarray) -> np.ndarray:
        """
        Reference (non-cached) attention forward pass.

        x: (batch, seq_len, hidden)
        Returns: (batch, seq_len, hidden)

        Computes attention over the full sequence at each step.
        This is the ground-truth implementation used in KAT tests.
        """
        batch, seq, _ = x.shape
        q, k, v = self._project(x)
        # Apply RoPE at offset 0 (full sequence)
        q = apply_rope(q, self._freqs, offset=0)
        k = apply_rope(k, self._freqs, offset=0)
        attn_out = _sdp_attention(q, k, v, causal_mask=True)
        # Merge heads: (batch, seq, n_heads*head_dim)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, -1)
        return attn_out @ self.o_w.T

    def forward_cached(
        self,
        x: np.ndarray,
        cache: "KVCache",
        layer_idx: int,
    ) -> np.ndarray:
        """
        Cached (incremental) attention forward pass for a single new token.

        x: (batch, 1, hidden)  — current token's hidden state
        cache: KVCache object holding accumulated K/V tensors
        layer_idx: which layer slot in the cache to use

        Returns: (batch, 1, hidden)

        Appends new K/V to the cache, then attends over the full cached sequence.

        Fault detected: if offset is wrong, RoPE positions are off and output
        will not match reference_forward; tested in test_kv_cache_equals_reference.
        """
        batch, _, _ = x.shape
        offset = cache.seq_len
        hd = self.cfg.head_dim

        q, k_new, v_new = self._project(x)  # each: (batch, n_heads or n_kv_heads, 1, hd)
        q = apply_rope(q, self._freqs, offset=offset)
        k_new = apply_rope(k_new, self._freqs, offset=offset)

        # Append to cache and get full K/V
        k_full, v_full = cache.append(layer_idx, k_new, v_new)

        attn_out = _sdp_attention(q, k_full, v_full, causal_mask=True)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, 1, -1)
        return attn_out @ self.o_w.T


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------


class KVCache:
    """
    Per-layer key-value cache for autoregressive decoding.

    Stores K and V tensors incrementally as tokens are generated.
    Shape per layer: (batch, n_kv_heads, seq_so_far, head_dim).

    Memory usage: 2 * n_layers * n_kv_heads * seq_len * head_dim * sizeof(dtype) bytes.
    This is the quantity tracked by contract/cost.py.
    """

    def __init__(self, cfg: ModelConfig, batch: int = 1, dtype: np.dtype = np.float32) -> None:
        self.cfg = cfg
        self.batch = batch
        self.dtype = dtype
        self.seq_len: int = 0
        # One entry per layer: list of (k_tensor, v_tensor)
        self._cache: list[tuple[np.ndarray, np.ndarray] | None] = [
            None
        ] * cfg.num_layers

    def append(
        self,
        layer_idx: int,
        k_new: np.ndarray,
        v_new: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Append new K/V slice to the cache for *layer_idx*, return full K/V.

        k_new: (batch, n_kv_heads, 1, head_dim)
        v_new: (batch, n_kv_heads, 1, head_dim)

        Returns (k_full, v_full) both shape (batch, n_kv_heads, seq+1, head_dim).
        """
        prev = self._cache[layer_idx]
        if prev is None:
            k_full = k_new
            v_full = v_new
        else:
            k_prev, v_prev = prev
            k_full = np.concatenate([k_prev, k_new], axis=2)
            v_full = np.concatenate([v_prev, v_new], axis=2)

        self._cache[layer_idx] = (k_full, v_full)
        # Advance seq_len counter only once per step (after all layers process same token)
        # Caller manages this; we just update at layer 0
        if layer_idx == 0:
            self.seq_len += 1
        return k_full, v_full

    def reset(self) -> None:
        """Clear the cache."""
        self._cache = [None] * self.cfg.num_layers
        self.seq_len = 0

    def memory_bytes(self) -> int:
        """Compute current cache memory usage in bytes."""
        itemsize = np.dtype(self.dtype).itemsize
        return (
            2  # K and V
            * self.cfg.num_layers
            * self.cfg.num_kv_heads
            * self.seq_len
            * self.cfg.head_dim
            * itemsize
        )
