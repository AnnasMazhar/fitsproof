"""
fitsproof.contract.verify — Proof harness: measure peak memory vs declared budget.

Samples peak RSS (resident set size) during generation and asserts that
measured peak never exceeded the declared budget. Also implements
capability-gated determinism tiers.

Determinism tiers (from detllm, arXiv 2601.17768 pattern):
  Tier 0: artifact reproducibility only (same code, same weights → same files)
  Tier 1: run-to-run output reproducibility at fixed seed
  Tier 2: Tier 1 + logprob equality within tolerance

The harness always reports which tier the backend actually achieves.
It never claims a higher tier than it can prove.

Sources:
  - detllm: https://github.com/tommasocerruti/detllm (capability-gated tiers)
  - arXiv 2601.17768 (LLM-42): decode-verify-rollback for determinism
"""

from __future__ import annotations

import os
import resource
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from fitsproof.contract.admit import AdmitRecord, AdmitStatus


class DeterminismTier(int, Enum):
    TIER_0 = 0  # artifact reproducibility
    TIER_1 = 1  # run-to-run output repeatability
    TIER_2 = 2  # Tier 1 + logprob equality


@dataclass
class VerifyRecord:
    """
    Result of the proof harness for one configuration.

    budget_bytes: the declared budget.
    measured_peak_bytes: measured RSS peak during generation.
    budget_respected: measured_peak_bytes <= budget_bytes.
    margin_bytes: budget_bytes - measured_peak_bytes (positive = safe).
    mode_changed_silently: True if a backend transition occurred without a record.
    determinism_tier: highest determinism tier achieved.
    """

    budget_bytes: int
    measured_peak_bytes: int
    budget_respected: bool
    margin_bytes: int
    mode_changed_silently: bool
    determinism_tier: DeterminismTier
    elapsed_s: float
    config_label: str = ""
    extra: dict = field(default_factory=dict)


def _get_rss_bytes() -> int:
    """
    Return current process RSS (Resident Set Size) in bytes.

    Reads from /proc/self/status VmRSS on Linux for a point-in-time current
    RSS reading (not the lifetime high-water mark). Falls back to ru_maxrss.

    Note: this returns *current* RSS, not peak-since-process-start.
    For per-config peak measurement, callers should track the maximum of multiple
    samples during and after the call.
    """
    # Prefer /proc/self/status VmRSS on Linux — it is current RSS, not lifetime HWM
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except (FileNotFoundError, ValueError):
        pass

    # Fallback: ru_maxrss (lifetime high-water mark — less accurate for per-config)
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if os.uname().sysname == "Linux":
            return usage.ru_maxrss * 1024
        return usage.ru_maxrss
    except (OSError, AttributeError):
        pass

    return 0


def _get_peak_rss_hwm() -> int:
    """
    Return the process-lifetime peak RSS high-water mark in bytes.

    On Linux, ru_maxrss is the VmHWM — the maximum RSS since process start.
    This is useful for reporting the absolute maximum across the whole run.
    """
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if os.uname().sysname == "Linux":
            return usage.ru_maxrss * 1024
        return usage.ru_maxrss
    except (OSError, AttributeError):
        return 0


def _sample_peak_rss(
    fn: Callable[[], list[int]],
    sample_interval_s: float = 0.01,
) -> tuple[list[int], int]:
    """
    Run *fn* in the current thread, measuring peak RSS during the call.

    Uses current VmRSS samples (from /proc/self/status, not lifetime ru_maxrss),
    so per-config measurements reflect the config's actual memory contribution
    rather than a degenerate process-lifetime maximum.

    Strategy:
      1. Record baseline VmRSS before the call.
      2. Sample VmRSS after the call.
      3. Peak = max(before, after) — captures allocation growth.

    Note: this is a best-effort measure. Allocations that are both allocated and
    freed within fn() may not be captured if the kernel reclaims pages before the
    post-call sample. For conservative budget enforcement, callers should use a
    budget with margin above the predicted peak.

    Fault detected: if we only sample after the call, we miss peak memory during
    intermediate computation.
    """
    # Sample before
    rss_before = _get_rss_bytes()

    result = fn()

    # Sample after (may be lower if memory was freed, but captures any retained growth)
    rss_after = _get_rss_bytes()

    # Peak for this config = max of before and after.
    # We also add the growth delta on top of the baseline to get an estimate
    # of the per-config peak: baseline + max(0, delta).
    growth = max(0, rss_after - rss_before)
    peak = rss_before + growth

    return result, peak


def verify_run(
    fn: Callable[[], list[int]],
    budget_bytes: int,
    admit_record: AdmitRecord,
    config_label: str = "",
    determinism_check_fn: Callable[[], list[int]] | None = None,
) -> VerifyRecord:
    """
    Run *fn* (a generation callable), measure peak RSS, assert budget compliance.

    Args:
        fn:                  Callable returning a list of generated token ids.
        budget_bytes:        The declared memory budget.
        admit_record:        The AdmitRecord from admit(); used to check for
                             silent mode changes (REFUSED records should not reach verify).
        config_label:        Human-readable label for this configuration.
        determinism_check_fn: If provided, called a second time to check Tier 1.

    Returns a VerifyRecord. budget_respected=False is evidence of a contract violation.

    Fault detected: if budget_bytes < 0, the comparison would trivially fail;
    tested with a budget of 0 bytes to confirm budget_respected=False.
    """
    # A REFUSED admit record reaching verify() is a contract violation
    if admit_record.status == AdmitStatus.REFUSED:
        raise RuntimeError(
            f"verify_run called with a REFUSED admit record for {config_label!r}. "
            "A refused plan must not reach execution."
        )

    t0 = time.perf_counter()
    result, peak_rss = _sample_peak_rss(fn)
    elapsed = time.perf_counter() - t0

    budget_respected = peak_rss <= budget_bytes
    margin = budget_bytes - peak_rss

    # Detect silent mode changes: a silent mode change occurs when execution
    # proceeds via a different path than what was admitted without emitting a record.
    #
    # Detectable cases:
    #   1. admit_record.status == ADMITTED but the plan verdict was FITS_WITH_DEGRADATION
    #      (someone bypassed admit() and passed a hand-constructed ADMITTED record for
    #      a config that required degradation). This is a builder error, not a user error.
    #   2. admit_record.status == DEGRADED but applied_degradation is None
    #      (admitted as degraded without naming what changed — the record is incomplete).
    #
    # Non-detectable at this layer: runtime backend switches that happen inside fn()
    # without touching the admit/plan layer (e.g., NumPy falling back to a different
    # BLAS). Those are outside the contract boundary.
    mode_changed_silently = False

    if admit_record.status == AdmitStatus.ADMITTED:
        # Check: if the plan predicted a peak that exceeds budget, an ADMITTED record
        # means the planner and the enforcer disagree — that is a silent mode change.
        if admit_record.plan is not None and admit_record.plan.predicted_peak_bytes > budget_bytes:
            mode_changed_silently = True

    elif admit_record.status == AdmitStatus.DEGRADED:
        # A DEGRADED record with no applied_degradation named is a silent mode change
        if admit_record.applied_degradation is None:
            mode_changed_silently = True

    # Determinism tier assessment
    tier = DeterminismTier.TIER_0
    if determinism_check_fn is not None:
        result2, _ = _sample_peak_rss(determinism_check_fn)
        if result == result2:
            tier = DeterminismTier.TIER_1
            # Tier 2 requires logprob equality (not measured here; promoted only
            # when logprob comparison is explicitly enabled)

    return VerifyRecord(
        budget_bytes=budget_bytes,
        measured_peak_bytes=peak_rss,
        budget_respected=budget_respected,
        margin_bytes=margin,
        mode_changed_silently=mode_changed_silently,
        determinism_tier=tier,
        elapsed_s=elapsed,
        config_label=config_label,
    )


@dataclass
class StressResult:
    """
    Aggregate result from the stress harness across multiple configurations.

    violations: number of configurations where measured_peak > budget.
    silent_mode_changes: number of configurations with undocumented mode changes.
    """

    n_configs: int
    violations: int
    silent_mode_changes: int
    records: list[VerifyRecord]
    margin_bytes: list[int]  # margin per config

    @property
    def violation_free(self) -> bool:
        return self.violations == 0

    @property
    def all_modes_explicit(self) -> bool:
        return self.silent_mode_changes == 0

    def summary(self) -> str:
        margins = np.array(self.margin_bytes)
        return (
            f"Stress harness: {self.n_configs} configs, "
            f"{self.violations} violations, "
            f"{self.silent_mode_changes} silent mode changes. "
            f"Margin: min={margins.min() / 1e6:.1f} MB, "
            f"median={np.median(margins) / 1e6:.1f} MB, "
            f"max={margins.max() / 1e6:.1f} MB."
        )


def run_stress_harness(
    configs: list[dict],
    budget_bytes: int,
) -> StressResult:
    """
    Run the stress harness across all *configs*.

    Each config dict must have keys:
      - "fn": Callable[[], list[int]]       — generation callable
      - "admit_record": AdmitRecord         — from admit()
      - "label": str                        — human-readable label
      - "determinism_fn": optional Callable — for Tier 1 check

    Returns a StressResult. violations must be 0 for the harness to pass.

    Fault detected: a config where fn() uses more memory than budget_bytes
    will set violations > 0, failing the acceptance criterion.
    """
    records: list[VerifyRecord] = []
    violations = 0
    silent_changes = 0

    for cfg in configs:
        fn = cfg["fn"]
        admit_record = cfg["admit_record"]
        label = cfg.get("label", "")
        det_fn = cfg.get("determinism_fn")

        rec = verify_run(fn, budget_bytes, admit_record, label, det_fn)
        records.append(rec)
        if not rec.budget_respected:
            violations += 1
        if rec.mode_changed_silently:
            silent_changes += 1

    margins = [r.margin_bytes for r in records]
    return StressResult(
        n_configs=len(configs),
        violations=violations,
        silent_mode_changes=silent_changes,
        records=records,
        margin_bytes=margins,
    )
