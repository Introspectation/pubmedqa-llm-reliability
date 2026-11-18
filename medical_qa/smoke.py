"""Exactly 20 gold-blind cases. No gold scoring, judge, or prompt selection."""

import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .dataset import ROOT, Case, source_lock, validate_pmids
from .inference import MODEL, OPTIONS, PROMPT_SHA256, OllamaRunner
from .validation import RESPONSE_SCHEMA

SMOKE_SIZE = 20
SELECTION_SALT = "pubmedqa-m0-m1-v1"


def select_smoke_cases(cases: tuple[Case, ...]) -> tuple[Case, ...]:
    validate_pmids(case.pmid for case in cases)
    ranked = sorted(cases, key=lambda case: hashlib.sha256(
        f"{SELECTION_SALT}:{case.pmid}".encode("ascii")
    ).hexdigest())
    return tuple(ranked[:SMOKE_SIZE])


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_smoke(cases, selection_report, output: Path, *, runner=None) -> dict:
    runner = runner or OllamaRunner()
    digest = runner.verify_model()
    sample = select_smoke_cases(cases)
    # An existing result can never be silently replaced or appended to.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": SMOKE_SIZE,
        "selection": "ascending SHA256(salt + ':' + PMID), first 20",
        "selection_salt": SELECTION_SALT,
        "pmids": [case.pmid for case in sample],
        "gold_blind": True,
        "model": MODEL,
        "model_digest": digest,
        "ollama_server_version": runner.server_version,
        "options": OPTIONS,
        "think": False,
        "response_schema": RESPONSE_SCHEMA,
        "prompt_sha256": PROMPT_SHA256,
        "sources": source_lock(),
        "dataset_validation": selection_report.to_dict(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ["datasets", "ollama", "pydantic", "pyarrow"]},
        "requirements_sha256": hashlib.sha256((ROOT / "requirements.txt").read_bytes()).hexdigest(),
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted((ROOT / "medical_qa").glob("*.py"))},
    }
    write_json(output / "manifest.json", manifest)
    statuses = Counter()
    latencies = []
    started = perf_counter()
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for number, case in enumerate(sample, 1):
            case_started = perf_counter()
            try:
                prediction = runner.predict(case).to_dict()
            except Exception as exc:
                # Keep technical failures in the denominator; no retry or prompt repair.
                prediction = {
                    "pmid": case.pmid, "status": "inference_error",
                    "latency_seconds": perf_counter() - case_started,
                    "error_type": type(exc).__name__,
                }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
            statuses[prediction["status"]] += 1
            latencies.append(prediction["latency_seconds"])
            print(f"{number:02d}/20 PMID={case.pmid} status={prediction['status']}", flush=True)
    summary = {
        "attempted": SMOKE_SIZE,
        "grounded": statuses["grounded"],
        "ungrounded": statuses["ungrounded"],
        "invalid_json": statuses["invalid_json"],
        "inference_errors": statuses["inference_error"],
        "valid_json": statuses["grounded"] + statuses["ungrounded"],
        "wall_seconds": perf_counter() - started,
        "mean_request_seconds": sum(latencies) / SMOKE_SIZE,
        "gold_scoring_performed": False,
        "judge_calls": 0,
    }
    write_json(output / "summary.json", summary)
    manifest["status"] = "completed" if not summary["inference_errors"] else "completed_with_errors"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "manifest.json", manifest)
    return summary
