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
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from fitsproof.contract.admit import AdmitRecord, AdmitStatus

# What measured_peak_bytes is. Exposed so tests (and users) can verify the
# harness is reporting a sampled peak of THIS run, never the process HWM.
MEASUREMENT_SOURCE = "sampled_vmrss"


class DeterminismTier(int, Enum):
    TIER_0 = 0  # artifact reproducibility
    TIER_1 = 1  # run-to-run output repeatability
    TIER_2 = 2  # Tier 1 + logprob equality


@dataclass
class VerifyRecord:
    """
    Result of the proof harness for one configuration.

    budget_bytes: the declared budget.
    measured_peak_bytes: sampled PEAK LIVE RSS (VmRSS) of this run — the
        measurement. Never ru_maxrss/VmHWM; see _sample_peak_rss.
    measurement_source: which quantity measured_peak_bytes is
        ("sampled_vmrss"). Exposed so the harness cannot silently revert to
        a constant (process high-water) measurement.
    hwm_bytes: process high-water mark (VmHWM) at end of run — reported as a
        separate, clearly-labelled column only, never as the measurement.
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
    hwm_bytes: int = 0
    measurement_source: str = MEASUREMENT_SOURCE
    extra: dict = field(default_factory=dict)


def _read_status_rss_hwm() -> tuple[int, int]:
    """
    Read (VmRSS, VmHWM) from /proc/self/status in a single pass.

    Both lines come from one snapshot of the file, where VmHWM >= VmRSS holds
    by construction (verified over thousands of allocation cycles). Reading
    them separately across calls does NOT compose: on some kernels VmHWM
    reads slightly lower after a large munmap than it did while the memory
    was held, so an independently-timed VmHWM read can fall below an earlier
    VmRSS sample.
    """
    rss = hwm = 0
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) * 1024
    return rss, hwm


def _get_rss_bytes() -> int:
    """
    Return the *current* process RSS (Resident Set Size) in bytes.

    Reads /proc/self/status on Linux (VmRSS — live value, not high-water mark).
    This is the right metric for delta measurement: RSS after − RSS before gives
    the net memory cost of a generation call, matching what the predictor estimates
    (model weights + KV cache + activations), without including interpreter overhead
    that was already allocated before the call.

    We choose delta measurement (RSS after − RSS before) over absolute RSS
    because the Python + NumPy baseline (~70–370 MB, machine-dependent) dwarfs the
    model cost (~39 MB for the reference bundle) and was already accounted for by
    machine.process_baseline_bytes in the prediction.  The predictor adds the
    baseline at plan time; the verifier confirms the model-specific delta.

    Falls back to ru_maxrss on non-Linux platforms (coarser, but acceptable there).
    """
    # Primary: /proc/self/status — live RSS, not HWM
    try:
        rss, _ = _read_status_rss_hwm()
        if rss:
            return rss
    except (FileNotFoundError, ValueError, OSError):
        pass

    # Fallback: ru_maxrss (HWM — will be the same across calls if nothing freed)
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if os.uname().sysname == "Linux":
            return usage.ru_maxrss * 1024
        return usage.ru_maxrss
    except (OSError, AttributeError):
        pass
    return 0


def _get_hwm_bytes() -> int:
    """
    Return the process high-water mark (VmHWM) in bytes.

    VmHWM is the maximum RSS the process has reached (per a single status
    snapshot). It cannot discriminate between configurations: after any earlier
    call raises it, every later run reports at least that same constant. It is
    reported as a separate, clearly-labelled column so the difference from the
    sampled peak stays visible — it is never the measurement.
    """
    try:
        _, hwm = _read_status_rss_hwm()
        if hwm:
            return hwm
    except (FileNotFoundError, ValueError, OSError):
        pass

    # Fallback: ru_maxrss IS the HWM on every platform
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
    sample_interval_s: float = 0.001,
) -> tuple[list[int], int, int]:
    """
    Run *fn* in the current thread and return
    (result, peak_live_rss_bytes, hwm_bytes).

    peak_live_rss_bytes is the maximum *live* RSS (VmRSS) observed before,
    during and after the call — the peak of THIS run. It is the quantity
    compared against the declared budget, because the budget covers the full
    working set (interpreter + numpy + model) and the prediction includes
    machine.process_baseline_bytes for the same reason.

    hwm_bytes is the maximum VmHWM seen across the same paired samples. Every
    RSS sample is paired with an HWM from the same status snapshot
    (VmHWM >= VmRSS), so peak_live_rss_bytes <= hwm_bytes always holds, while
    the two stay clearly distinguishable: the HWM is a cumulative process
    constant, the peak is per-run.

    Why sample during the run instead of reading after it:
      - Reading VmRSS only after fn() returns misses transient peaks: a config
        that allocates its working set and frees it before returning looks
        identical to one that allocates nothing.
      - Reading ru_maxrss/VmHWM is worse: the high-water mark never
        decreases, so once probe()'s benchmark arrays (or any earlier config)
        raise it, every later config reports the same constant — the harness
        is non-discriminating (the defect CI caught: all 25 configs reporting
        one identical margin).

    How: a background thread samples /proc/self/status every *sample_interval_s*
    while fn runs. time.sleep() and NumPy ops release the GIL, so the sampler
    is scheduled during real workloads; pre- and post-call samples guarantee a
    value even for runs shorter than the interval (those observe the run's
    steady-state RSS, which still varies with persistent allocations).

    On platforms without /proc this degrades to the pre/post readings of the
    ru_maxrss fallback in _get_rss_bytes (coarser, but the tests that require
    discrimination are Linux-only, matching CI).
    """

    def _read_pair() -> tuple[int, int]:
        try:
            rss, hwm = _read_status_rss_hwm()
            if rss:
                return rss, hwm
        except (FileNotFoundError, ValueError, OSError):
            pass
        # Fallback reads are two separate snapshots; clamp so the paired
        # invariant (peak <= hwm) survives even if VmHWM dips between reads.
        rss = _get_rss_bytes()
        return rss, max(_get_hwm_bytes(), rss)

    peak, hwm_peak = _read_pair()
    stop = threading.Event()

    def _sampler() -> None:
        nonlocal peak, hwm_peak
        while not stop.wait(sample_interval_s):
            rss, hwm = _read_pair()
            if rss > peak:
                peak = rss
            if hwm > hwm_peak:
                hwm_peak = hwm

    sampler = threading.Thread(target=_sampler, name="fitsproof-rss-sampler", daemon=True)
    sampler.start()
    try:
        result = fn()
    finally:
        stop.set()
        sampler.join()

    rss, hwm = _read_pair()
    if rss > peak:
        peak = rss
    if hwm > hwm_peak:
        hwm_peak = hwm
    return result, peak, hwm_peak


def verify_run(
    fn: Callable[[], list[int]],
    budget_bytes: int,
    admit_record: AdmitRecord,
    config_label: str = "",
    determinism_check_fn: Callable[[], list[int]] | None = None,
) -> VerifyRecord:
    """
    Run *fn* (a generation callable), measure the sampled peak live RSS of
    this run, assert budget compliance.

    measured_peak_bytes is the sampled peak of *this* call (VmRSS before/
    during/after), never the process high-water mark; hwm_bytes is reported
    alongside as a separate, clearly-labelled column. The headline number,
    margin_bytes, is budget_bytes − measured_peak_bytes, so two configs with
    genuinely different footprints cannot report the same margin.

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
    # (peak_rss, hwm) come from paired status samples: peak is this run's
    # measurement, hwm is the separate, clearly-labelled process-HWM column.
    result, peak_rss, hwm = _sample_peak_rss(fn)
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
        result2, _, _ = _sample_peak_rss(determinism_check_fn)
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
        hwm_bytes=hwm,
        measurement_source=MEASUREMENT_SOURCE,
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

    @property
    def measurement_source(self) -> str:
        """Which quantity measured_peak_bytes is (exposed for regression tests)."""
        if self.records:
            return self.records[0].measurement_source
        return MEASUREMENT_SOURCE

    def summary(self) -> str:
        margins = np.array(self.margin_bytes)
        peaks = np.array([r.measured_peak_bytes for r in self.records])
        hwm = max((r.hwm_bytes for r in self.records), default=0)
        return (
            f"Stress harness: {self.n_configs} configs, "
            f"{self.violations} violations, "
            f"{self.silent_mode_changes} silent mode changes.\n"
            f"  measurement: {self.measurement_source} (sampled peak live RSS "
            f"of each run, not the process high-water mark)\n"
            f"  margin (budget - sampled peak): min={margins.min() / 1e6:.1f} MB, "
            f"median={np.median(margins) / 1e6:.1f} MB, max={margins.max() / 1e6:.1f} MB\n"
            f"  sampled peak: min={peaks.min() / 1e6:.1f} MB, max={peaks.max() / 1e6:.1f} MB; "
            f"process VmHWM (separate column, not the measurement): {hwm / 1e6:.1f} MB"
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
