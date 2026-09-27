#!/usr/bin/env python
"""
Calibration benchmark — held-out MAPE for the cost model on this machine.

Runs offline against the in-repo reference model (no network, no GPU) and
prints the raw numbers the README quotes under "Prediction accuracy".
Reproduce with:

    python scripts/calibration_demo.py

What it measures: collect_measurements() decodes with the real transformer;
calibrate() fits bandwidth_utilisation on a train split and scores MAPE on
the held-out split (fit and evaluate on disjoint sets — see
src/fitsproof/contract/calibrate.py:calibrate).
"""

from __future__ import annotations

from fitsproof.contract.calibrate import calibrate, collect_measurements
from fitsproof.contract.probe import probe
from fitsproof.engine.model import get_reference_bundle
from fitsproof.engine.transformer import Transformer


def main() -> int:
    cfg, weights = get_reference_bundle()
    machine = probe()
    print("=== Calibration demo ===")
    print(f"bandwidth: {machine.memory_bandwidth_bps / 1e9:.2f} GB/s")
    print(f"gemm:      {machine.gemm_throughput_flops / 1e9:.2f} GFLOPS")
    print(f"RAM:       {machine.memory_bytes / 1e9:.1f} GB")

    measurements = collect_measurements(Transformer(cfg, weights), cfg)
    result = calibrate(measurements, machine, cfg)
    # calibrate._mape already returns a percentage (50.3 == 50.3%).
    print(f"bandwidth_utilisation: {result.bandwidth_utilisation:.4f}")
    print(f"MAPE (held-out):       {result.mape_held_out:.1f}%")
    print(f"CI (95%):              [{result.ci_lower:.1f}%, {result.ci_upper:.1f}%]")
    print(f"n_train={result.n_train}, n_held_out={result.n_held_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
