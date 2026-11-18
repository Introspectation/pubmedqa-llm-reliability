"""Run from the repository root with the existing venv's Python."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .dataset import ROOT, load_dataset
from .inference import OllamaRunner
from .smoke import run_smoke, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="PubMedQA: validation, 20-case smoke, and final baseline scoring")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["validate-dataset", "smoke"]:
        command = subparsers.add_parser(name)
        command.add_argument("--parquet", type=Path, help="Offline copy of the pinned HF Parquet")
        command.add_argument("--refresh-ids", action="store_true", help="Reverify IDs against pinned GitHub JSON")
        command.add_argument("--output", type=Path, help="Validation JSON file or new smoke output directory")
    subparsers.add_parser("check-ollama", help="Verify target digest without inference")
    scoring = subparsers.add_parser(
        "final-score-baselines", help="Read official test gold and score both fixed baselines at final scoring only",
    )
    scoring.add_argument("--parquet", type=Path, help="Offline copy of the pinned HF Parquet")
    scoring.add_argument("--output", type=Path, required=True, help="New final baseline report JSON file")
    full = subparsers.add_parser("full-inference", help="Resume the frozen official-500 run, then validate and score")
    full.add_argument("--parquet", type=Path, help="Offline copy of the pinned HF Parquet")
    full.add_argument("--output", type=Path, default=ROOT / "runs/phase1-full", help="Resumable run directory")
    full.add_argument("--smoke-manifest", type=Path, default=ROOT / "runs/smoke-m0-m1/manifest.json")
    args = parser.parse_args()
    try:
        if args.command == "check-ollama":
            runner = OllamaRunner()
            print(json.dumps({"model_digest": runner.verify_model(), "server_version": runner.server_version}, indent=2))
            return 0
        if args.command == "final-score-baselines":
            from .scoring import final_score_baselines

            if args.output.exists():
                raise FileExistsError("Final baseline report already exists; choose a new output path.")
            report = final_score_baselines(parquet=args.parquet)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps({"output": str(args.output), **report}, indent=2))
            return 0
        if args.command == "full-inference":
            from .benchmark import run_full_benchmark

            cases, report = load_dataset(parquet=args.parquet)
            summary = run_full_benchmark(cases, report, args.output, args.smoke_manifest, parquet=args.parquet)
            print(json.dumps({"output": str(args.output), **summary}, indent=2))
            return 0
        cases, report = load_dataset(parquet=args.parquet, refresh_ids=args.refresh_ids)
        if args.command == "validate-dataset":
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.output, report.to_dict())
            print(json.dumps(report.to_dict(), indent=2))
            return 0
        output = args.output or ROOT / "runs" / datetime.now(timezone.utc).strftime("smoke-%Y%m%dT%H%M%S%fZ")
        summary = run_smoke(cases, report, output)
        print(json.dumps({"output": str(output), **summary}, indent=2))
        return 1 if summary["inference_errors"] else 0
    except Exception as exc:
        # No dataset contents, gold values, or reserved PMIDs in CLI errors.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
