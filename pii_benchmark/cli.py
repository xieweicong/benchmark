"""Command line interface for the benchmark harness."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, parse_csv_ints, parse_csv_strings, selected_models
from .report import load_rows, write_csv, write_markdown
from .runner import BenchmarkRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pii-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a benchmark suite.")
    run_parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/pii-redaction.json"))
    run_parser.add_argument("--models", help="Comma-separated model names/types/model ids to run.")
    run_parser.add_argument("--sizes", help="Comma-separated benchmark token buckets.")
    run_parser.add_argument("--repeats", type=int)
    run_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    run_parser.add_argument(
        "--opf-checkpoint",
        help="Checkpoint directory for OPF/openai privacy-filter. Overrides OPF_CHECKPOINT.",
    )
    run_parser.add_argument("--quality-limit", type=int)
    run_parser.add_argument("--no-quality", action="store_true")
    run_parser.add_argument("--no-batch-compare", action="store_true")
    run_parser.add_argument("--power-watts", type=float, help="Hardware power budget for tok/s/W.")
    run_parser.add_argument("--power-note", help="Free-form note for the power measurement or mode.")
    run_parser.add_argument("--out", help="JSONL output path.")
    run_parser.add_argument("--markdown-report", help="Optional Markdown report path.")
    run_parser.add_argument("--csv", help="Optional CSV summary path.")

    report_parser = subparsers.add_parser("report", help="Generate reports from JSONL files.")
    report_parser.add_argument("--input", nargs="+", required=True, help="JSONL file(s) or directories.")
    report_parser.add_argument("--markdown", required=True, help="Markdown report output path.")
    report_parser.add_argument("--csv", help="Optional CSV output path.")

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "report":
        return _report(args)
    return 2


def _run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.sizes:
        config["speed_sizes"] = parse_csv_ints(args.sizes)
    if args.repeats is not None:
        config["repeats"] = args.repeats
    if args.power_watts:
        config["power_watts"] = args.power_watts
        os.environ["PII_BENCH_POWER_WATTS"] = str(args.power_watts)
    if args.power_note:
        os.environ["PII_BENCH_POWER_NOTE"] = args.power_note

    models = selected_models(config, parse_csv_strings(args.models))
    if args.opf_checkpoint:
        for model in models:
            if model.get("type") == "opf":
                model["checkpoint"] = args.opf_checkpoint
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out or PROJECT_ROOT / "results" / f"run-{stamp}.jsonl")
    runner = BenchmarkRunner(
        config=config,
        models=models,
        out_path=out,
        project_root=PROJECT_ROOT,
        device=args.device,
        speed_sizes=config.get("speed_sizes"),
        repeats=config.get("repeats"),
        quality_limit=args.quality_limit,
        run_quality=not args.no_quality,
        run_batch_compare=not args.no_batch_compare,
    )
    result_path = runner.run()
    rows = load_rows([result_path])
    if args.markdown_report:
        write_markdown(rows, args.markdown_report)
    if args.csv:
        write_csv(rows, args.csv)
    print(f"Wrote {result_path}")
    if args.markdown_report:
        print(f"Wrote {args.markdown_report}")
    if args.csv:
        print(f"Wrote {args.csv}")
    return 0


def _report(args: argparse.Namespace) -> int:
    rows = load_rows(args.input)
    write_markdown(rows, args.markdown)
    if args.csv:
        write_csv(rows, args.csv)
    print(f"Wrote {args.markdown}")
    if args.csv:
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
