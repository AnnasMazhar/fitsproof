"""
fitsproof CLI entry point.

Subcommands:
  probe     — characterise the current machine
  plan      — predict resource usage for a given model config
  admit     — enforce the contract (refuses or degrades loudly)
  verify    — run the proof harness (assert measured peak <= budget)
  stress    — run the stress harness over N configurations
  server    — start the OpenAI-compatible HTTP server
  mcp       — start the MCP server (stdio transport)
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

    # admit
    admit_p = sub.add_parser("admit", help="Enforce the resource contract (refuse or degrade)")
    admit_p.add_argument("--budget-gb", type=float, default=4.0)
    admit_p.add_argument("--context", type=int, default=512)
    admit_p.add_argument("--quant", default="none")

    # verify
    verify_p = sub.add_parser("verify", help="Proof harness: assert measured peak <= budget")
    verify_p.add_argument("--budget-gb", type=float, default=4.0)
    verify_p.add_argument("--context", type=int, default=128)
    verify_p.add_argument("--quant", default="none")
    verify_p.add_argument("--tokens", type=int, default=16, help="Tokens to generate")

    # server
    srv_p = sub.add_parser("server", help="Start OpenAI-compatible HTTP server")
    srv_p.add_argument("--host", default="127.0.0.1")
    srv_p.add_argument("--port", type=int, default=8080)

    # mcp
    _mcp_p = sub.add_parser(
        "mcp",
        help="Start MCP server (stdio transport) — exposes probe/plan/admit as MCP tools",
    )

    # pareto
    _pareto_p = sub.add_parser("pareto", help="Sweep and emit the Pareto frontier")

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

    elif args.command in ("plan", "admit"):
        from fitsproof.contract.admit import admit as _admit
        from fitsproof.contract.plan import plan as make_plan
        from fitsproof.contract.probe import probe
        from fitsproof.engine.model import REFERENCE_CONFIG

        machine = probe()
        budget = int(args.budget_gb * 1e9)
        p = make_plan(REFERENCE_CONFIG, machine, args.context, budget, args.quant)
        record = _admit(p)
        print(record.message)
        if p.degradations:
            print("Degradation options:")
            for d in p.degradations:
                fits = "fits" if d.fits_budget else "does not fit"
                print(f"  [{fits}] {d.description} -> {d.predicted_peak_bytes / 1e9:.3f} GB")
        # Non-zero exit on refusal
        if record.status.value == "refused":
            return 2

    elif args.command == "verify":
        from fitsproof.contract.admit import admit as _admit
        from fitsproof.contract.plan import plan as make_plan
        from fitsproof.contract.probe import probe
        from fitsproof.contract.verify import verify_run
        from fitsproof.engine.model import REFERENCE_CONFIG, get_reference_bundle
        from fitsproof.engine.sampling import Sampler
        from fitsproof.engine.transformer import Transformer

        machine = probe()
        budget = int(args.budget_gb * 1e9)
        p = make_plan(REFERENCE_CONFIG, machine, args.context, budget, args.quant)
        record = _admit(p)
        print(record.message)
        if record.status.value == "refused":
            print("REFUSED — verify skipped (no execution on refused plans)")
            return 2

        cfg, weights = get_reference_bundle()
        transformer = Transformer(cfg, weights)
        prompt = [1, 2, 3, 4]

        vresult = verify_run(
            fn=lambda: transformer.generate(
                prompt, max_new_tokens=args.tokens, temperature=0.0, sampler=Sampler(0)
            ),
            budget_bytes=budget,
            admit_record=record,
            config_label="cli_verify",
        )
        print(f"  measured_peak:    {vresult.measured_peak_bytes / 1e6:.1f} MB")
        print(f"  budget:           {budget / 1e6:.1f} MB")
        print(f"  budget_respected: {vresult.budget_respected}")
        print(f"  margin:           {vresult.margin_bytes / 1e6:.1f} MB")
        if not vresult.budget_respected:
            print("PROOF FAILED: measured peak exceeded declared budget.")
            return 1

    elif args.command == "server":
        from fitsproof.engine.model import get_reference_bundle
        from fitsproof.engine.server import start_server
        from fitsproof.engine.transformer import Transformer

        print("Loading reference model...")
        cfg, weights = get_reference_bundle()
        transformer = Transformer(cfg, weights)
        print(f"Starting server on {args.host}:{args.port}")
        start_server(transformer, cfg, args.host, args.port, block=True)

    elif args.command == "mcp":
        from fitsproof.mcp import run_mcp_server

        run_mcp_server()

    elif args.command == "pareto":
        from fitsproof.contract.pareto import run_pareto_sweep
        from fitsproof.contract.probe import probe
        from fitsproof.engine.model import REFERENCE_CONFIG, get_reference_bundle
        from fitsproof.engine.transformer import Transformer

        cfg, weights = get_reference_bundle()
        transformer = Transformer(cfg, weights)
        machine = probe()
        table = run_pareto_sweep(transformer, cfg, machine)
        header = (
            f"{'quant':<12} {'ctx':>6} {'peak_MB':>9} "
            f"{'tok/s':>8} {'top1':>6} {'pred_MB':>8} {'dominated':>10}"
        )
        print(header)
        print("-" * len(header))
        for row in table:
            print(
                f"{row.quant:<12} {row.context_len:>6} "
                f"{row.measured_peak_mb:>9.1f} {row.tok_s:>8.2f} "
                f"{row.top1_agreement:>6.3f} {row.predicted_peak_mb:>8.1f} "
                f"{'yes' if row.dominated else 'no':>10}"
            )
        non_dom = sum(1 for r in table if not r.dominated)
        print(f"Total configs: {len(table)}")
        print(f"Non-dominated: {non_dom}")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
