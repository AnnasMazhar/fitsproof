"""
fitsproof CLI entry point.

Subcommands:
  probe     — characterise the current machine
  plan      — predict resource usage for a given model config
  admit     — enforce the contract (refuses or degrades loudly)
  verify    — run the proof harness (assert measured peak <= budget)
  stress    — run the stress harness over >=20 configurations
  server    — start the OpenAI-compatible HTTP server (alias: serve)
  mcp       — start the MCP server (stdio transport)
  pareto    — sweep and emit the Pareto frontier
"""

from __future__ import annotations

import argparse
import math
import sys

from fitsproof import __version__


def _budget_bytes(gb: float) -> int | None:
    """Convert --budget-gb to bytes; None if not a positive finite number."""
    if not math.isfinite(gb) or gb <= 0:
        return None
    return int(gb * 1e9)


def _load_model_config(path: str | None) -> object:
    """Load a ModelConfig from a fitsproof bundle dir, or the built-in fixture."""
    if path is None:
        from fitsproof.engine.model import REFERENCE_CONFIG

        return REFERENCE_CONFIG
    from pathlib import Path

    from fitsproof.engine.model import load_bundle

    cfg, _weights = load_bundle(Path(path))
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="fitsproof",
        description="Predicts, enforces, and proves an LLM inference resource contract.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"fitsproof {__version__}",
        help="Print the fitsproof version and exit",
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
    plan_p.add_argument(
        "--model", default=None, help="Path to a fitsproof bundle dir (default: built-in fixture)"
    )

    # admit
    admit_p = sub.add_parser("admit", help="Enforce the resource contract (refuse or degrade)")
    admit_p.add_argument("--budget-gb", type=float, default=4.0)
    admit_p.add_argument("--context", type=int, default=512)
    admit_p.add_argument("--quant", default="none")
    admit_p.add_argument(
        "--model", default=None, help="Path to a fitsproof bundle dir (default: built-in fixture)"
    )

    # verify
    verify_p = sub.add_parser("verify", help="Proof harness: assert measured peak <= budget")
    verify_p.add_argument("--budget-gb", type=float, default=4.0)
    verify_p.add_argument("--context", type=int, default=128)
    verify_p.add_argument("--quant", default="none")
    verify_p.add_argument("--tokens", type=int, default=16, help="Tokens to generate")

    # stress
    stress_p = sub.add_parser(
        "stress", help="Run the stress harness across >=20 configurations (exit 1 on violation)"
    )
    stress_p.add_argument("--budget-gb", type=float, default=4.0)
    stress_p.add_argument("--context", type=int, default=64)
    stress_p.add_argument("--quant", default="none")

    # server (alias: serve — binary surface name)
    srv_p = sub.add_parser("server", aliases=["serve"], help="Start OpenAI-compatible HTTP server")
    srv_p.add_argument("--host", default="127.0.0.1")
    srv_p.add_argument("--port", type=int, default=8080)
    srv_p.add_argument(
        "--budget-gb", type=float, default=4.0, help="Memory budget enforced on every request"
    )

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

    elif args.command == "plan":
        # plan — show the prediction (peak, CI, tok/s, verdict) without enforcing.
        # admit — enforce the contract (ADMITTED/REFUSED/DEGRADED, exit 2 on refusal).
        # They are separate commands: plan lets you inspect what the contract predicts
        # before you commit to enforcement; admit is the gate you wire into CI.
        from fitsproof.contract.plan import plan as make_plan
        from fitsproof.contract.probe import probe

        budget = _budget_bytes(args.budget_gb)
        if budget is None:
            print(
                f"ERROR: --budget-gb must be a positive finite number, got {args.budget_gb}",
                file=sys.stderr,
            )
            return 2
        try:
            model_cfg = _load_model_config(args.model)
            machine = probe()
            p = make_plan(model_cfg, machine, args.context, budget, args.quant)
        except (ValueError, OSError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        # Show the prediction — peak, CI, tok/s — not an enforcement decision.
        lo_gb = p.predicted_peak_ci[0] / 1e9
        hi_gb = p.predicted_peak_ci[1] / 1e9
        tok_lo = p.predicted_tok_s_ci[0]
        tok_hi = p.predicted_tok_s_ci[1]
        verdict_str = p.verdict.value  # fits / fits_with_degradation / does_not_fit
        print(
            f"predicted peak:  {p.predicted_peak_bytes / 1e9:.3f} GB  "
            f"(95% CI: [{lo_gb:.3f}, {hi_gb:.3f}] GB)"
        )
        print(f"predicted tok/s: {p.predicted_tok_s:.1f}  (95% CI: [{tok_lo:.1f}, {tok_hi:.1f}])")
        print(f"budget:          {p.budget_bytes / 1e9:.3f} GB")
        print(f"verdict:         {verdict_str}")
        if p.degradations:
            print("degradation options:")
            for d in p.degradations:
                fits = "fits" if d.fits_budget else "does not fit"
                print(
                    f"  [{fits}] {d.description} -> {d.predicted_peak_bytes / 1e9:.3f} GB"
                    f"  ({d.predicted_tok_s:.1f} tok/s)"
                )
        # plan exits 0 even on does_not_fit — it describes, it does not enforce.

    elif args.command == "admit":
        from fitsproof.contract.admit import admit as _admit
        from fitsproof.contract.plan import plan as make_plan
        from fitsproof.contract.probe import probe

        budget = _budget_bytes(args.budget_gb)
        if budget is None:
            print(
                f"ERROR: --budget-gb must be a positive finite number, got {args.budget_gb}",
                file=sys.stderr,
            )
            return 2
        try:
            model_cfg = _load_model_config(args.model)
            machine = probe()
            p = make_plan(model_cfg, machine, args.context, budget, args.quant)
        except (ValueError, OSError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        record = _admit(p)
        print(record.message)
        if p.degradations:
            print("Degradation options:")
            for d in p.degradations:
                fits = "fits" if d.fits_budget else "does not fit"
                print(f"  [{fits}] {d.description} -> {d.predicted_peak_bytes / 1e9:.3f} GB")
        # Non-zero exit on refusal or near-boundary warning
        if record.status.value == "refused":
            return 2
        if record.status.value == "near_boundary":
            return 1

    elif args.command == "verify":
        from fitsproof.contract.admit import admit as _admit
        from fitsproof.contract.plan import plan as make_plan
        from fitsproof.contract.probe import probe
        from fitsproof.contract.verify import verify_run
        from fitsproof.engine.model import REFERENCE_CONFIG, get_reference_bundle
        from fitsproof.engine.sampling import Sampler
        from fitsproof.engine.transformer import Transformer

        budget = _budget_bytes(args.budget_gb)
        if budget is None:
            print(
                f"ERROR: --budget-gb must be a positive finite number, got {args.budget_gb}",
                file=sys.stderr,
            )
            return 2
        machine = probe()
        try:
            p = make_plan(REFERENCE_CONFIG, machine, args.context, budget, args.quant)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
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
        print(
            f"  measurement:      {vresult.measurement_source} (sampled peak live RSS of this run)"
        )
        print(
            f"  process VmHWM:    {vresult.hwm_bytes / 1e6:.1f} MB (separate column, not the measurement)"
        )
        print(f"  budget:           {budget / 1e6:.1f} MB")
        print(f"  budget_respected: {vresult.budget_respected}")
        print(f"  margin:           {vresult.margin_bytes / 1e6:.1f} MB")
        if not vresult.budget_respected:
            print("PROOF FAILED: measured peak exceeded declared budget.")
            return 1

    elif args.command == "stress":
        import numpy as np

        from fitsproof.contract.admit import admit as _admit
        from fitsproof.contract.plan import plan as make_plan
        from fitsproof.contract.probe import probe
        from fitsproof.contract.verify import run_stress_harness
        from fitsproof.engine.model import REFERENCE_CONFIG, get_reference_bundle
        from fitsproof.engine.sampling import Sampler
        from fitsproof.engine.transformer import Transformer

        budget = _budget_bytes(args.budget_gb)
        if budget is None:
            print(
                f"ERROR: --budget-gb must be a positive finite number, got {args.budget_gb}",
                file=sys.stderr,
            )
            return 2
        machine = probe()
        try:
            p = make_plan(REFERENCE_CONFIG, machine, args.context, budget, args.quant)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        record = _admit(p)
        print(record.message)
        if record.status.value == "refused":
            return 2
        if record.status.value == "near_boundary":
            return 1

        cfg, weights = get_reference_bundle()
        transformer = Transformer(cfg, weights)
        rng = np.random.default_rng(42)
        configs = []
        for prompt_len in (2, 4, 6, 8, 10):
            for decode_len in (1, 2, 4, 8, 16):
                prompt = rng.integers(0, cfg.vocab_size, size=prompt_len, dtype=np.int64).tolist()
                sampler = Sampler(seed=prompt_len + decode_len)
                configs.append(
                    {
                        "fn": lambda p=prompt, d=decode_len, s=sampler: transformer.generate(
                            p, max_new_tokens=d, temperature=0.0, sampler=s
                        ),
                        "admit_record": record,
                        "label": f"prompt{prompt_len}_decode{decode_len}",
                    }
                )
        result = run_stress_harness(configs, budget_bytes=budget)
        print(result.summary())
        if not (result.violation_free and result.all_modes_explicit):
            return 1

    elif args.command in ("server", "serve"):
        from fitsproof.engine.model import get_reference_bundle
        from fitsproof.engine.server import start_server
        from fitsproof.engine.transformer import Transformer

        print("Loading reference model...")
        cfg, weights = get_reference_bundle()
        transformer = Transformer(cfg, weights)
        budget = _budget_bytes(args.budget_gb)
        if budget is None:
            print(
                f"ERROR: --budget-gb must be a positive finite number, got {args.budget_gb}",
                file=sys.stderr,
            )
            return 2
        print(f"Starting server on {args.host}:{args.port} (budget: {budget / 1e9:.3f} GB)")
        start_server(transformer, cfg, args.host, args.port, block=True, budget_bytes=budget)

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
                f"{row.measured_peak_bytes / 1e6:>9.1f} {row.measured_tok_s:>8.2f} "
                f"{row.top1_agreement:>6.3f} {row.predicted_peak_bytes / 1e6:>8.1f} "
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
