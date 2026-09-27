"""
fitsproof CLI entry point.

Subcommands:
  probe     — characterise the current machine
  plan      — predict resource usage for a given model config
  stress    — run the stress harness
  server    — start the OpenAI-compatible HTTP server
  pareto    — sweep and emit the Pareto frontier
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="fitsproof",
        description="Predicts, enforces, and proves an LLM inference resource contract.",
    )
    sub = parser.add_subparsers(dest="command")

    # probe
    probe_p = sub.add_parser("probe", help="Characterise the current machine")
    probe_p.add_argument("--out", default=None, help="Write JSON profile to this path")

    # plan
    plan_p = sub.add_parser("plan", help="Predict resource usage")
    plan_p.add_argument("--budget-gb", type=float, default=4.0)
    plan_p.add_argument("--context", type=int, default=512)
    plan_p.add_argument("--quant", default="none")

    # server
    srv_p = sub.add_parser("server", help="Start OpenAI-compatible HTTP server")
    srv_p.add_argument("--host", default="127.0.0.1")
    srv_p.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    if args.command == "probe":
        from pathlib import Path

        from fitsproof.contract.probe import probe, save_profile

        print("Probing machine...")
        profile = probe()
        print(f"  bandwidth:  {profile.memory_bandwidth_bps / 1e9:.2f} GB/s")
        print(f"  gemm:       {profile.gemm_throughput_flops / 1e9:.2f} GFLOPS")
        print(f"  RAM:        {profile.memory_bytes / 1e9:.2f} GB")
        print(f"  VRAM:       {profile.gpu_memory_bytes / 1e9:.2f} GB")
        if args.out:
            save_profile(profile, Path(args.out))
            print(f"  saved to:   {args.out}")

    elif args.command == "plan":
        from fitsproof.contract.admit import admit
        from fitsproof.contract.plan import plan as make_plan
        from fitsproof.contract.probe import probe
        from fitsproof.engine.model import REFERENCE_CONFIG

        machine = probe()
        budget = int(args.budget_gb * 1e9)
        p = make_plan(REFERENCE_CONFIG, machine, args.context, budget, args.quant)
        record = admit(p)
        print(record.message)
        if p.degradations:
            print("Degradation options:")
            for d in p.degradations:
                fits = "fits" if d.fits_budget else "does not fit"
                print(f"  [{fits}] {d.description} -> {d.predicted_peak_bytes / 1e9:.3f} GB")

    elif args.command == "server":
        from fitsproof.engine.model import get_reference_bundle
        from fitsproof.engine.server import start_server
        from fitsproof.engine.transformer import Transformer

        print("Loading reference model...")
        cfg, weights = get_reference_bundle()
        transformer = Transformer(cfg, weights)
        print(f"Starting server on {args.host}:{args.port}")
        start_server(transformer, cfg, args.host, args.port, block=True)

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
