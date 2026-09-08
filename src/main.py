from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.orchestrator import ResearchOrchestrator
from src.utils.cycle_config import autorun_safety_decision, build_next_cycle_config
from src.utils.cycle_runner import run_bounded_cycles
from src.utils.run_tracker import load_run_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Template-based IE/OR research automation assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run the research automation pipeline")
    run.add_argument("--config", required=True, help="Path to YAML config")
    prepare = sub.add_parser("prepare-next", help="Create a next-cycle config from a completed run directory")
    prepare.add_argument("--run-dir", required=True, help="Completed run directory containing config_used.yaml and next_config_patch.yaml")
    prepare.add_argument("--output", help="Optional output path for the generated YAML config")
    prepare.add_argument(
        "--autorun-safe-only",
        action="store_true",
        help="Run the generated next-cycle config only if the previous run has no human-required safety gate",
    )
    cycles = sub.add_parser("run-cycles", help="Run bounded safe research cycles from an initial config")
    cycles.add_argument("--config", required=True, help="Initial YAML config")
    cycles.add_argument("--max-cycles", type=int, default=2, help="Maximum number of cycles to attempt")
    cycles.add_argument("--chain-output", help="Optional path for cycle_chain.json")
    resume = sub.add_parser("resume", help="Resume or restart from an incomplete run directory")
    resume.add_argument("--run-dir", required=True, help="Run directory containing run_status.json and config_used.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config_path = Path(args.config)
        config = load_config(config_path)
        run_dir = ResearchOrchestrator(config, project_root=Path.cwd()).run()
        print(f"Run complete: {run_dir}")
        return 0
    if args.command == "prepare-next":
        output = build_next_cycle_config(args.run_dir, args.output)
        print(f"Next-cycle config written: {output}")
        if args.autorun_safe_only:
            safe, reasons = autorun_safety_decision(args.run_dir)
            if not safe:
                print("Autorun skipped because human review is required:")
                for reason in reasons:
                    print(f"- {reason}")
                return 0
            config = load_config(output)
            run_dir = ResearchOrchestrator(config, project_root=Path.cwd()).run()
            print(f"Autorun complete: {run_dir}")
        return 0
    if args.command == "run-cycles":
        report = run_bounded_cycles(args.config, args.max_cycles, Path.cwd(), args.chain_output)
        print(f"Cycle chain complete: {len(report.records)} record(s)")
        for record in report.records:
            print(f"- cycle {record.cycle_index}: {record.status}")
        return 0
    if args.command == "resume":
        run_dir = Path(args.run_dir)
        status = load_run_status(run_dir)
        if status.get("status") == "completed":
            print(f"Run already completed: {run_dir}")
            return 0
        config_path = run_dir / "config_used.yaml"
        if not config_path.exists():
            print(f"Cannot resume: missing {config_path}")
            return 1
        config = load_config(config_path)
        new_run_dir = ResearchOrchestrator(config, project_root=Path.cwd()).run()
        print(f"Resume restarted run from saved config: {new_run_dir}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
