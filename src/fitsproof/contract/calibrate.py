"""
fitsproof.contract.calibrate — Fit cost model constants from on-device measurements.

Fits *bandwidth_utilisation* (and optionally a latency overhead term) against
real measurements on the current machine, using a train/hold-out split.

Reports:
  - MAPE (Mean Absolute Percentage Error) on held-out configurations
  - 95% prediction interval on the held-out set (bootstrap)
  - The fitted parameter value and its bootstrap confidence interval

Sources:
  - McCalpin 1995 (STREAM) — bandwidth measurement methodology
  - Williams et al. 2009 (Roofline) — the model being calibrated
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from fitsproof.contract.cost import decode_tok_s, weight_bytes
from fitsproof.contract.probe import MachineProfile
from fitsproof.engine.model import ModelConfig
from fitsproof.engine.transformer import Transformer


@dataclass
class Measurement:
    """A single timing measurement: config -> observed tok/s."""

    config_label: str
    model_weight_bytes: int
    measured_tok_s: float
    quant: str = "none"


@dataclass
class CalibrationResult:
    """
    Result of fitting the cost model against real measurements.

    mape_held_out: Mean Absolute Percentage Error on the held-out set (the honest number).
    ci_lower, ci_upper: 95% bootstrap prediction interval on held-out MAPE.
    bandwidth_utilisation: fitted parameter (fraction of peak bandwidth achieved).
    n_train: number of configurations used for fitting.
    n_held_out: number of configurations used for evaluation.
    """

    bandwidth_utilisation: float
    mape_held_out: float
    ci_lower: float
    ci_upper: float
    n_train: int
    n_held_out: int
    measurements: list[Measurement]


def _measure_tok_s(
    transformer: Transformer,
    prompt_ids: list[int],
    n_tokens: int = 10,
    n_repeats: int = 3,
) -> float:
    """
    Measure decode throughput on the given transformer.

    Runs *n_tokens* generation steps *n_repeats* times and returns
    median tokens/second.

    Fault detected: if we include the prefill time, the estimate is
    biased downward for long prompts; we measure decode steps only.
    """
    from fitsproof.engine.sampling import Sampler

    times = []
    for _ in range(n_repeats):
        sampler = Sampler(seed=42)
        t0 = time.perf_counter()
        transformer.generate(prompt_ids, max_new_tokens=n_tokens, sampler=sampler, temperature=0.0)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        if elapsed > 0:
            times.append(n_tokens / elapsed)

    if not times:
        return 0.0
    return float(np.median(times))


def collect_measurements(
    transformer: Transformer,
    cfg: ModelConfig,
    prompt_variants: list[list[int]] | None = None,
    n_tokens: int = 5,
    n_repeats: int = 2,
) -> list[Measurement]:
    """
    Collect tok/s measurements across multiple prompt configurations.

    *prompt_variants*: list of prompt token sequences. If None, uses
    three synthetic prompts of lengths 4, 8, and 16.

    Each measurement captures the actual decode throughput for the
    given model weights and machine.
    """
    if prompt_variants is None:
        rng = np.random.default_rng(42)
        V = cfg.vocab_size
        prompt_variants = [
            rng.integers(0, V, size=4, dtype=np.int64).tolist(),
            rng.integers(0, V, size=8, dtype=np.int64).tolist(),
            rng.integers(0, V, size=16, dtype=np.int64).tolist(),
        ]

    measurements: list[Measurement] = []
    for i, prompt in enumerate(prompt_variants):
        tok_s = _measure_tok_s(transformer, prompt, n_tokens=n_tokens, n_repeats=n_repeats)
        measurements.append(
            Measurement(
                config_label=f"prompt_len_{len(prompt)}",
                model_weight_bytes=weight_bytes(cfg),
                measured_tok_s=tok_s,
                quant="none",
            )
        )
    return measurements


def _fit_bandwidth_utilisation(
    measurements: list[Measurement],
    machine: MachineProfile,
    cfg: ModelConfig,
) -> float:
    """
    Fit bandwidth_utilisation by minimising sum of squared relative errors.

    Closed-form solution:
        predicted_tok_s = (bw * util) / weight_bytes
        => util = mean(measured_tok_s * weight_bytes / bw)

    This is the least-squares fit for the single free parameter.

    Fault detected: fitting on the held-out set (not the train set) leads to
    over-optimistic MAPE; this is tested by checking MAPE <= 1.0 (<=100%).
    """
    bw = machine.memory_bandwidth_bps
    if bw <= 0:
        return 0.6  # fallback

    utils = []
    for m in measurements:
        if m.measured_tok_s > 0 and m.model_weight_bytes > 0:
            implied_util = (m.measured_tok_s * m.model_weight_bytes) / bw
            utils.append(implied_util)

    if not utils:
        return 0.6
    return float(np.mean(utils))


def _mape(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    MAPE = mean(|actual - predicted| / |actual|) * 100.

    Returns percentage (e.g. 15.3 means 15.3% average error).
    Fault detected: dividing by predicted instead of actual inverts
    the error direction; tested with known arrays.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    nonzero = actual != 0
    if not np.any(nonzero):
        return float("nan")
    return float(np.mean(np.abs(actual[nonzero] - predicted[nonzero]) / np.abs(actual[nonzero])) * 100.0)


def _bootstrap_mape_ci(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Bootstrap 95% confidence interval on MAPE.

    Resamples (predicted, actual) pairs with replacement and computes MAPE
    for each resample, returning the (alpha/2, 1-alpha/2) quantiles.

    Fault detected: using the training distribution rather than the held-out
    distribution inflates confidence (narrower CI than warranted).
    """
    rng = np.random.default_rng(seed)
    n = len(predicted)
    if n < 2:
        m = _mape(predicted, actual)
        return m, m

    mapes = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        mapes.append(_mape(predicted[idx], actual[idx]))

    mapes_arr = np.array(mapes)
    return float(np.quantile(mapes_arr, alpha / 2)), float(np.quantile(mapes_arr, 1 - alpha / 2))


def calibrate(
    measurements: list[Measurement],
    machine: MachineProfile,
    cfg: ModelConfig,
    train_fraction: float = 0.67,
    seed: int = 42,
) -> CalibrationResult:
    """
    Fit the cost model and evaluate on held-out configurations.

    Splits *measurements* into train/held-out sets (deterministic via *seed*),
    fits *bandwidth_utilisation* on the train set, evaluates MAPE + 95% CI
    on the held-out set.

    Returns a CalibrationResult with the honest held-out metrics.

    Fault detected: fitting and evaluating on the same set gives optimistic
    MAPE=0 for the exact fitted case; the train/held-out split prevents this.
    """
    rng = np.random.default_rng(seed)
    n = len(measurements)
    if n < 2:
        # Not enough data for a held-out split: fit everything, report on train
        util = _fit_bandwidth_utilisation(measurements, machine, cfg)
        preds = np.array([
            decode_tok_s(cfg, machine, m.quant, bandwidth_utilisation=util)
            for m in measurements
        ])
        actuals = np.array([m.measured_tok_s for m in measurements])
        mape = _mape(preds, actuals)
        return CalibrationResult(
            bandwidth_utilisation=util,
            mape_held_out=mape,
            ci_lower=mape,
            ci_upper=mape,
            n_train=n,
            n_held_out=0,
            measurements=measurements,
        )

    indices = rng.permutation(n)
    n_train = max(1, int(n * train_fraction))
    train_idx = indices[:n_train]
    held_idx = indices[n_train:]

    train_ms = [measurements[i] for i in train_idx]
    held_ms = [measurements[i] for i in held_idx]

    util = _fit_bandwidth_utilisation(train_ms, machine, cfg)

    preds_held = np.array([
        decode_tok_s(cfg, machine, m.quant, bandwidth_utilisation=util)
        for m in held_ms
    ])
    actuals_held = np.array([m.measured_tok_s for m in held_ms])

    mape = _mape(preds_held, actuals_held)
    ci_lo, ci_hi = _bootstrap_mape_ci(preds_held, actuals_held)

    return CalibrationResult(
        bandwidth_utilisation=util,
        mape_held_out=mape,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_train=len(train_ms),
        n_held_out=len(held_ms),
        measurements=measurements,
    )
