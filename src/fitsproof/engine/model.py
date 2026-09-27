"""
Reference model bundle for fitsproof.

Generates and loads a tiny randomly-initialised transformer for offline CI use.
Config: 6 layers, 384 hidden, 6 heads, 2 KV heads (GQA), vocab 256, max_seq 512.
Weights are deterministic (seed=42), stored as .npz archives.

This module is the only place that knows the on-disk format of a fitsproof bundle.

Bundle layout:
    config.json      — architecture and dtype metadata
    weights.npz      — all parameter tensors
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Architecture specification for a fitsproof bundle.

    Fault detected by tests: wrong shape causes silent matmul broadcast errors.
    """

    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int        # query heads
    num_kv_heads: int     # key/value heads; must divide num_heads evenly
    intermediate_size: int
    max_seq_len: int
    dtype: str            # "float32" or "float16"
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_heads

    @property
    def kv_groups(self) -> int:
        """Number of query heads per KV head (GQA group size)."""
        return self.num_heads // self.num_kv_heads

    def validate(self) -> None:
        """Raise ValueError for invalid configurations."""
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads {self.num_heads} must be divisible by num_kv_heads {self.num_kv_heads}"
            )
        if self.dtype not in {"float32", "float16"}:
            raise ValueError(f"dtype must be float32 or float16, got {self.dtype!r}")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(**d)


# ---------------------------------------------------------------------------
# Reference model dimensions
# ---------------------------------------------------------------------------

REFERENCE_CONFIG = ModelConfig(
    vocab_size=256,
    hidden_size=384,
    num_layers=6,
    num_heads=6,
    num_kv_heads=2,
    intermediate_size=1024,
    max_seq_len=512,
    dtype="float32",
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
)


# ---------------------------------------------------------------------------
# Weight generation
# ---------------------------------------------------------------------------


def _init_weights(cfg: ModelConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """
    Generate randomly initialised weight tensors for a transformer.

    Naming convention mirrors common LLM layouts for readability.
    dtype is always float32 for generation; callers may cast.
    """
    d = cfg.hidden_size
    h = cfg.num_heads
    kv_h = cfg.num_kv_heads
    hd = cfg.head_dim
    i = cfg.intermediate_size
    V = cfg.vocab_size
    scale = 0.02  # small init scale

    weights: dict[str, np.ndarray] = {}

    # Token embedding
    weights["embed"] = rng.standard_normal((V, d)).astype(np.float32) * scale

    for layer in range(cfg.num_layers):
        prefix = f"layer_{layer}"
        # Attention projections
        weights[f"{prefix}.q_proj"] = rng.standard_normal((h * hd, d)).astype(np.float32) * scale
        weights[f"{prefix}.k_proj"] = rng.standard_normal((kv_h * hd, d)).astype(np.float32) * scale
        weights[f"{prefix}.v_proj"] = rng.standard_normal((kv_h * hd, d)).astype(np.float32) * scale
        weights[f"{prefix}.o_proj"] = rng.standard_normal((d, h * hd)).astype(np.float32) * scale
        # Feed-forward (SwiGLU-style: gate + up + down)
        weights[f"{prefix}.gate_proj"] = rng.standard_normal((i, d)).astype(np.float32) * scale
        weights[f"{prefix}.up_proj"] = rng.standard_normal((i, d)).astype(np.float32) * scale
        weights[f"{prefix}.down_proj"] = rng.standard_normal((d, i)).astype(np.float32) * scale
        # RMSNorm scales (init to 1)
        weights[f"{prefix}.attn_norm"] = np.ones(d, dtype=np.float32)
        weights[f"{prefix}.ffn_norm"] = np.ones(d, dtype=np.float32)

    # Final norm + unembedding (tied to embed by default but stored separately for simplicity)
    weights["final_norm"] = np.ones(d, dtype=np.float32)
    weights["unembed"] = rng.standard_normal((V, d)).astype(np.float32) * scale

    return weights


def generate_reference_model(path: Path, seed: int = 42) -> None:
    """
    Write the reference model bundle to *path* (a directory).

    Creates: config.json, weights.npz.
    Idempotent: overwrites if already present.
    """
    path.mkdir(parents=True, exist_ok=True)
    cfg = REFERENCE_CONFIG
    cfg.validate()

    rng = np.random.default_rng(seed)
    weights = _init_weights(cfg, rng)

    with open(path / "config.json", "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)

    np.savez_compressed(path / "weights.npz", **weights)


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------


def load_bundle(path: Path) -> tuple[ModelConfig, dict[str, np.ndarray]]:
    """
    Load a fitsproof model bundle from *path*.

    Returns (config, weights_dict).
    Raises FileNotFoundError if the bundle is incomplete.
    Raises ValueError if the config is invalid.
    """
    config_path = path / "config.json"
    weights_path = path / "weights.npz"

    for p in (config_path, weights_path):
        if not p.exists():
            raise FileNotFoundError(f"Bundle missing required file: {p}")

    with open(config_path) as fh:
        cfg = ModelConfig.from_dict(json.load(fh))
    cfg.validate()

    npz = np.load(weights_path)
    weights = {k: npz[k] for k in npz.files}
    return cfg, weights


# ---------------------------------------------------------------------------
# Lazy reference model (cached after first call)
# ---------------------------------------------------------------------------

_CACHED_BUNDLE: tuple[Path, ModelConfig, dict[str, np.ndarray]] | None = None


def get_reference_bundle(base: Path | None = None) -> tuple[ModelConfig, dict[str, np.ndarray]]:
    """
    Return the in-repo reference model, generating it if necessary.

    *base* defaults to the package's own `_reference_model/` directory.
    Thread-safety: not guaranteed; tests that need isolation should call
    `load_bundle(path)` directly.
    """
    global _CACHED_BUNDLE

    if base is None:
        base = Path(__file__).parent / "_reference_model"

    if _CACHED_BUNDLE is not None and _CACHED_BUNDLE[0] == base:
        return _CACHED_BUNDLE[1], _CACHED_BUNDLE[2]

    if not (base / "weights.npz").exists():
        generate_reference_model(base)

    cfg, weights = load_bundle(base)
    _CACHED_BUNDLE = (base, cfg, weights)
    return cfg, weights
