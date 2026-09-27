"""
fitsproof.contract.probe — Machine characterisation via measurement.

Measures:
  - Memory bandwidth (streaming triad: A = B + c*C per Schönauer/McCalpin)
  - GEMM throughput at several representative shapes
  - Available system RAM and VRAM (if a GPU is present)

Results are stored in a MachineProfile dataclass and serialised to JSON.
Every measurement records *what was measured*, *when*, and *on which host*.

Sources:
  - McCalpin 1995 (STREAM benchmark), https://www.cs.virginia.edu/stream/ref.html
  - Roofline model: Williams et al. 2009 (Roofline), https://dl.acm.org/doi/10.1145/1498765.1498785
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MachineProfile:
    """
    Measured machine characteristics.

    All bandwidth/throughput values are in bytes/second and FLOPS respectively.
    memory_bytes is total available RAM in bytes.
    gpu_memory_bytes is 0 if no GPU is present or detectable.
    """

    hostname: str
    platform_str: str
    measured_at: float  # Unix timestamp
    memory_bandwidth_bps: float  # bytes/second, streaming triad
    gemm_throughput_flops: float  # FLOPS for representative matmul shape
    memory_bytes: int  # total RAM
    gpu_memory_bytes: int  # VRAM (0 if absent)
    cpu_count: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MachineProfile:
        extra = d.pop("extra", {})
        return cls(**d, extra=extra)

    @classmethod
    def from_json(cls, s: str) -> MachineProfile:
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Memory bandwidth — streaming triad (McCalpin STREAM)
# ---------------------------------------------------------------------------


def _measure_bandwidth(
    array_size: int = 8 * 1024 * 1024,
    n_trials: int = 5,
) -> float:
    """
    Measure effective memory bandwidth using the STREAM triad kernel:
        A[i] = B[i] + scalar * C[i]

    *array_size* is in float64 elements (~64 MB default for three arrays).

    Returns bandwidth in bytes/second.
    The triad reads 2 arrays and writes 1; total bytes = 3 * array_size * 8.

    Fault detected: using array_size=1 would measure cache bandwidth, not
    DRAM bandwidth; tests verify the result is plausible (> 1 GB/s).
    """
    scalar = 3.0
    bytes_per_trial = 3 * array_size * np.dtype(np.float64).itemsize

    # Warmup
    a = np.ones(array_size, dtype=np.float64)
    b = np.ones(array_size, dtype=np.float64)
    c = np.ones(array_size, dtype=np.float64)
    a[:] = b + scalar * c

    best_bw = 0.0
    for _ in range(n_trials):
        t0 = time.perf_counter()
        a[:] = b + scalar * c
        t1 = time.perf_counter()
        elapsed = t1 - t0
        if elapsed > 0:
            best_bw = max(best_bw, bytes_per_trial / elapsed)

    return best_bw


# ---------------------------------------------------------------------------
# GEMM throughput
# ---------------------------------------------------------------------------


def _measure_gemm(
    shapes: list[tuple[int, int, int]] | None = None,
    n_trials: int = 3,
) -> float:
    """
    Measure peak GEMM throughput in FLOPS using NumPy matmul.

    Shapes are (M, K, N) tuples; 2*M*K*N FLOPS per multiply.
    Returns FLOPS for the best-performing shape.

    Fault detected: if shapes produce less than 1 GFLOPS on any modern CPU,
    the probe is measuring overhead not throughput; tests verify > 1e9.
    """
    if shapes is None:
        shapes = [(256, 256, 256), (512, 512, 512), (1024, 256, 1024)]

    best_flops = 0.0
    for m, k, n in shapes:
        flops = 2 * m * k * n
        a = np.random.default_rng(0).standard_normal((m, k)).astype(np.float32)
        b = np.random.default_rng(1).standard_normal((k, n)).astype(np.float32)
        # Warmup
        _ = a @ b
        for _ in range(n_trials):
            t0 = time.perf_counter()
            _ = a @ b
            t1 = time.perf_counter()
            elapsed = t1 - t0
            if elapsed > 0:
                best_flops = max(best_flops, flops / elapsed)

    return best_flops


# ---------------------------------------------------------------------------
# Memory detection
# ---------------------------------------------------------------------------


def _get_system_ram() -> int:
    """Return total system RAM in bytes from /proc/meminfo (Linux) or sysctl (macOS)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    # MemTotal:       32768000 kB
                    kb = int(line.split()[1])
                    return kb * 1024
    except (FileNotFoundError, ValueError):
        pass
    # Fallback: use os.sysconf on other platforms
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, ValueError):
        pass
    return 0


def _get_vram_bytes() -> int:
    """
    Attempt to detect VRAM by reading /proc/driver/nvidia/gpus or nvidia-smi.

    Returns 0 if no GPU is found or the toolkit is absent (which is the
    expected case on this machine — no CUDA toolkit installed).

    This function must never raise; it returns 0 on any failure.
    """
    # Try reading from /proc/driver/nvidia (present even without full toolkit)
    nvidia_dir = Path("/proc/driver/nvidia/gpus")
    if nvidia_dir.is_dir():
        for gpu_dir in sorted(nvidia_dir.iterdir()):
            info_file = gpu_dir / "information"
            if info_file.exists():
                try:
                    text = info_file.read_text()
                    for line in text.splitlines():
                        if "Video Memory" in line:
                            # "Video Memory:    4096 MB"
                            parts = line.split(":")
                            if len(parts) == 2:
                                val = parts[1].strip().split()[0]
                                unit = (
                                    parts[1].strip().split()[1].upper()
                                    if len(parts[1].strip().split()) > 1
                                    else "MB"
                                )
                                mb = int(val)
                                if unit == "GB":
                                    return mb * 1024 * 1024 * 1024
                                return mb * 1024 * 1024
                except (ValueError, IndexError, OSError):
                    pass
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def probe(
    bandwidth_array_size: int = 8 * 1024 * 1024,
    n_bandwidth_trials: int = 5,
    n_gemm_trials: int = 3,
) -> MachineProfile:
    """
    Characterise the current machine and return a MachineProfile.

    Measures are taken from the running machine — not from spec sheets.
    The profile is timestamped so stale profiles can be detected.

    Fault detected: a profile loaded from disk without checking the timestamp
    may reflect a different machine or hardware state; tests check that
    measured_at is populated and recent.
    """
    bw = _measure_bandwidth(bandwidth_array_size, n_bandwidth_trials)
    gemm = _measure_gemm(n_trials=n_gemm_trials)
    ram = _get_system_ram()
    vram = _get_vram_bytes()

    return MachineProfile(
        hostname=platform.node(),
        platform_str=platform.platform(),
        measured_at=time.time(),
        memory_bandwidth_bps=bw,
        gemm_throughput_flops=gemm,
        memory_bytes=ram,
        gpu_memory_bytes=vram,
        cpu_count=os.cpu_count() or 1,
    )


def save_profile(profile: MachineProfile, path: Path) -> None:
    """Write MachineProfile to a JSON file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.to_json())


def load_profile(path: Path) -> MachineProfile:
    """Load MachineProfile from a JSON file at *path*."""
    if not path.exists():
        raise FileNotFoundError(f"No profile found at {path}")
    return MachineProfile.from_json(path.read_text())
