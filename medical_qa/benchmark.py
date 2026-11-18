"""Resumable official-500 inference; validation and gold scoring come last."""

import hashlib
import json
import math
import os
import platform
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from filelock import FileLock

from .baselines import LABELS, baseline_definitions, baseline_predictions
from .dataset import ROOT, DatasetValidationError, load_test_pmids, source_lock, validate_pmids
from .inference import MODEL, OPTIONS, PROMPT_SHA256, OllamaRunner, build_messages
from .scoring import load_gold_for_final_scoring, score_official_predictions
from .validation import RESPONSE_SCHEMA, AnswerValidationError, parse_answer, validate_evidence

METRIC_POLICY = {
    "denominator": 500,
    "invalid": "Invalid JSON/schema, incomplete generation, or unexpected thinking output.",
    "accuracy": "All 500 cases; invalid responses count as incorrect, without label extraction or repair.",
    "macro_f1": "Unweighted mean for yes/no/maybe, including invalid responses as false negatives.",
    "grounded_rate": "Valid, complete responses passing the frozen exact-quote check divided by 500.",
    "latency": "Successful HTTP chat response times, including model loading, excluding digest preflight.",
    "p95": "95th percentile with linear interpolation (NumPy method='linear').",
    "retry": "Transport errors stop the run; restart retries only PMIDs without a completed response.",
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def frozen_request(case):
    """Same /api/chat arguments as the frozen smoke runner, in the same order."""
    return dict(model=MODEL, messages=build_messages(case), format=RESPONSE_SCHEMA,
                think=False, stream=False, options=OPTIONS)


def request_fingerprint(case):
    raw = json.dumps(frozen_request(case), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_smoke_reference(path, runner):
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("status") != "completed" or reference.get("sample_size") != 20:
        raise ValueError("Expected a completed 20-case smoke manifest.")
    current = {
        "model": MODEL, "model_digest": runner.verify_model(),
        "ollama_server_version": runner.server_version,
        "options": OPTIONS, "think": False, "response_schema": RESPONSE_SCHEMA,
        "prompt_sha256": PROMPT_SHA256, "sources": source_lock(),
        "requirements_sha256": _sha(ROOT / "requirements.txt"),
    }
    # JSON serialization also preserves schema property order and numeric types.
    for key, value in current.items():
        if json.dumps(value) != json.dumps(reference.get(key)):
            raise ValueError(f"Frozen smoke configuration changed: {key}.")
    prompt = (ROOT / "medical_qa/prompt.txt").read_text(encoding="utf-8")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != reference["prompt_sha256"]:
        raise ValueError("Prompt file differs from the frozen smoke prompt.")
    for name in ["inference.py", "validation.py"]:
        if _sha(ROOT / "medical_qa" / name) != reference["code_sha256"][name]:
            raise ValueError(f"Frozen smoke code changed: {name}.")
    for name, expected in reference["packages"].items():
        if version(name) != expected:
            raise ValueError(f"Frozen smoke package version changed: {name}.")
    return reference, current


def capture_response(runner, case):
    """Persist raw completed responses; do not inspect gold or validate quotes here."""
    runner.verify_model()
    started = perf_counter()
    response = runner.client.chat(**frozen_request(case))
    elapsed = perf_counter() - started
    runner.verify_model()
    return {
        "pmid": case.pmid, "request_sha256": request_fingerprint(case),
        "recorded_at_utc": _now(), "raw_response": response.message.content or "",
        "latency_seconds": elapsed, "done": bool(response.done),
        "done_reason": response.done_reason,
        "thinking_returned": bool(response.message.thinking),
        "prompt_eval_count": response.prompt_eval_count, "eval_count": response.eval_count,
    }


def read_completed_rows(path, cases):
    """Recover a torn final write, reject corruption/duplicates, verify input hashes."""
    if not path.exists():
        return {}, 0
    by_id = {case.pmid: case for case in cases}
    completed, recovered_bytes = {}, 0
    with path.open("rb+") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                if not line.endswith(b"\n"):
                    recovered_bytes = len(line)
                    handle.seek(start)
                    handle.truncate()
                    handle.flush()
                    os.fsync(handle.fileno())
                    break
                raise ValueError("Corrupt complete JSONL row; refusing to skip or overwrite it.") from exc
            if not isinstance(row, dict):
                raise ValueError("A prediction journal row is not an object.")
            pmid = row.get("pmid")
            if not isinstance(pmid, str) or pmid not in by_id or pmid in completed:
                raise ValueError("Prediction journal has duplicate or non-official PMIDs.")
            if row.get("request_sha256") != request_fingerprint(by_id[pmid]):
                raise ValueError("Stored prediction input differs from the frozen request.")
            latency = row.get("latency_seconds")
            if (isinstance(latency, bool) or not isinstance(latency, (int, float))
                    or not math.isfinite(latency) or latency < 0
                    or not isinstance(row.get("raw_response"), str)
                    or not isinstance(row.get("done"), bool)
                    or not isinstance(row.get("thinking_returned"), bool)
                    or row.get("done_reason") not in ("stop", "length", "load", "unload", None)):
                raise ValueError("Prediction journal metadata is malformed.")
            completed[pmid] = row
            if not line.endswith(b"\n"):
                # A complete JSON object without its final newline is still reusable.
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return completed, recovered_bytes


def validate_completed_cases(cases, completed):
    ids = validate_pmids(case.pmid for case in cases)
    if set(completed) != set(ids):
        raise DatasetValidationError("All 500 official responses are required before validation and gold scoring.")
    validated = []
    for case in cases:
        row = completed[case.pmid]
        answer, evidence, schema_valid, error = None, None, False, None
        try:
            answer = parse_answer(row["raw_response"])
            schema_valid = True
            evidence = validate_evidence(answer.evidence, case.contexts)
        except AnswerValidationError as exc:
            error = str(exc)
        protocol_valid = row["done"] and row["done_reason"] == "stop" and not row["thinking_returned"]
        if not protocol_valid:
            error = "incomplete_generation_or_unexpected_thinking"
        valid = schema_valid and protocol_valid
        status = ("grounded" if evidence.grounded else "ungrounded") if valid else "invalid"
        validated.append({
            **row, "status": status, "schema_valid": schema_valid,
            "protocol_valid": protocol_valid, "error": error,
            "answer": answer.model_dump() if answer else None,
            "evidence_check": asdict(evidence) if evidence else None,
            "scoring_label": answer.label if valid else "invalid",
        })
    return validated


def _markdown_report(summary):
    qwen = summary["results"][MODEL]
    lines = ["# Phase 1: official 500-case results", "",
             "Frozen smoke prompt, model digest, schema order and generation settings; no judge calls.", "",
             "| System | N | Accuracy | Macro-F1 | Invalid | Grounded | Mean latency | P95 latency |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, metrics in summary["results"].items():
        extra = (f"{summary['invalid_rate']:.2%} | {summary['grounded_rate']:.2%} | "
                 f"{summary['latency_seconds']['mean']:.3f}s | {summary['latency_seconds']['p95']:.3f}s") if name == MODEL else "N/A | N/A | N/A | N/A"
        lines.append(f"| {name} | {metrics['n']} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.4f} | {extra} |")
    lines += ["", "Both baselines predict yes and are identical; majority is fixed from published PQA-L priors.",
              "Baseline grounding/latency rates are not applicable to these label-only rules.", "",
              "## Per-class metrics", "",
              "| System | Class | Precision | Recall | F1 | Support |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, metrics in summary["results"].items():
        for label in LABELS:
            lines.append(f"| {name} | {label} | {metrics['per_class_precision'][label]:.4f} | "
                         f"{metrics['per_class_recall'][label]:.4f} | {metrics['per_class_f1'][label]:.4f} | "
                         f"{metrics['per_class_support'][label]} |")
    for title, metrics in [(MODEL, qwen), ("Both identical baselines", summary["results"]["always_yes"])]:
        matrix = metrics["confusion_matrix"]
        lines += ["", f"## Confusion matrix: {title}", "", "Rows = gold; columns = prediction.", "",
                  "| Gold | yes | no | maybe | invalid |", "| --- | ---: | ---: | ---: | ---: |"]
        for label, values in zip(LABELS, matrix["values"]):
            values = values + [0] * (4 - len(values))
            lines.append(f"| {label} | " + " | ".join(str(value) for value in values) + " |")
    lines += ["", "## Interpretation and provenance", "",
              f"- {summary['counts']['invalid']} invalid, {summary['counts']['grounded']} grounded, "
              f"{summary['counts']['ungrounded']} ungrounded out of 500. Invalid outputs count as incorrect.",
              "- Grounded means a normalized >=20-character quote occurs within one supplied section; it does not prove entailment or clinical reliability.",
              "- The official 500 test IDs were selected before reading examples. Reserved records were not inspected or evaluated.",
              "- JSON and exact-quote checks were applied to all 500 completed responses before gold access. No repair, prompt tuning, or judge calls.",
              f"- Model digest: `{summary['frozen_config']['model_digest']}`.",
              f"- Prompt SHA-256: `{summary['frozen_config']['prompt_sha256']}`.",
              f"- Settings: `{json.dumps(summary['frozen_config']['options'])}`, `think=false`.",
              "- Latency includes model load when present and excludes digest preflight; p95 uses linear interpolation.",
              f"- Transport failures recorded across attempts: {summary['transport_errors']}.",
              "- Full precision metrics, hashes, baselines' citation and software provenance are in `results-summary.json` and `manifest.json`.", ""]
    return "\n".join(lines)


def finalize_run(cases, completed, selection_report, manifest, output, *, parquet=None):
    """500-response barrier -> frozen checks saved -> gold -> metrics and reports."""
    ids = validate_pmids(case.pmid for case in cases)
    if tuple(ids) != load_test_pmids():
        raise DatasetValidationError("Final scoring cohort differs from official test PMIDs.")
    baselines = baseline_predictions(ids)  # Freeze before reading gold.
    validated = validate_completed_cases(cases, completed)
    temporary = output / "validated-predictions.jsonl.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in validated:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / "validated-predictions.jsonl")
    manifest.update(status="validated", completed_cases=500, validated_at_utc=_now())
    _atomic_json(output / "manifest.json", manifest)
    # This is the first gold access. Both raw and checked files already contain 500 rows.
    gold, gold_report = load_gold_for_final_scoring(parquet=parquet)
    if gold_report != selection_report:
        raise DatasetValidationError("Final gold selection differs from the blind inference cohort.")
    predicted = {row["pmid"]: row["scoring_label"] for row in validated}
    results = {MODEL: score_official_predictions(ids, gold, predicted, allow_invalid=True)}
    results.update({name: score_official_predictions(ids, gold, values) for name, values in baselines.items()})
    counts = Counter(row["status"] for row in validated)
    import numpy as np

    latencies = [row["latency_seconds"] for row in validated]
    error_file = output / "errors.jsonl"
    summary = {
        "stage": "phase1_full_no_judge", "completed_at_utc": _now(), "n": 500,
        "gold_scoring_performed": True, "judge_calls": 0,
        "frozen_config": manifest["frozen_config"], "metric_policy": METRIC_POLICY,
        "dataset_validation": selection_report.to_dict(),
        "counts": {status: counts[status] for status in ["grounded", "ungrounded", "invalid"]},
        "schema_invalid_count": sum(not row["schema_valid"] for row in validated),
        "generation_error_count": sum(not row["protocol_valid"] for row in validated),
        "invalid_rate": counts["invalid"] / 500, "grounded_rate": counts["grounded"] / 500,
        "latency_seconds": {"n": 500, "mean": float(np.mean(latencies)),
                            "p95": float(np.percentile(latencies, 95, method="linear")),
                            "sum": float(sum(latencies))},
        "transport_errors": len(error_file.read_text("utf-8").splitlines()) if error_file.exists() else 0,
        "baselines_identical": baselines["always_yes"] == baselines["published_majority"],
        "baseline_definitions": baseline_definitions(), "results": results,
        "predictions_sha256": _sha(output / "predictions.jsonl"),
        "validated_predictions_sha256": _sha(output / "validated-predictions.jsonl"),
        "scikit_learn_version": version("scikit-learn"), "numpy_version": version("numpy"),
    }
    _atomic_json(output / "results-summary.json", summary)
    (output / "report.md").write_text(_markdown_report(summary), encoding="utf-8")
    manifest.update(status="completed", gold_scoring_performed=True, finished_at_utc=_now(),
                    summary_sha256=_sha(output / "results-summary.json"), report_sha256=_sha(output / "report.md"))
    _atomic_json(output / "manifest.json", manifest)
    return summary


def run_full_benchmark(cases, selection_report, output, smoke_manifest, *, parquet=None, runner=None):
    cases = tuple(sorted(cases, key=lambda case: int(case.pmid)))
    ids = validate_pmids(case.pmid for case in cases)
    if ids != load_test_pmids():
        raise DatasetValidationError("Full inference is restricted to the official 500 test PMIDs.")
    runner = runner or OllamaRunner()
    reference, frozen = verify_smoke_reference(smoke_manifest, runner)
    if selection_report.to_dict() != reference["dataset_validation"]:
        raise DatasetValidationError("Dataset selection differs from the smoke reference.")
    code_hashes = {path.name: _sha(path) for path in sorted((ROOT / "medical_qa").glob("*.py"))}
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".run.lock"), timeout=0):
        manifest_path, journal = output / "manifest.json", output / "predictions.jsonl"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text("utf-8"))
            if (manifest.get("frozen_config") != frozen or manifest.get("pmids") != list(ids)
                    or manifest.get("metric_policy") != METRIC_POLICY
                    or manifest.get("code_sha256") != code_hashes
                    or manifest.get("smoke_manifest_sha256") != _sha(smoke_manifest)):
                raise ValueError("Resume configuration/code/cohort differs from the original full run.")
            manifest["resume_count"] += 1
        else:
            if any(item.name != ".run.lock" for item in output.iterdir()):
                raise ValueError("Cannot adopt existing run files without their run manifest.")
            manifest = {
                "stage": "phase1_full_no_judge", "status": "running", "started_at_utc": _now(),
                "pmids": list(ids), "completed_cases": 0, "gold_blind_inference": True,
                "gold_scoring_performed": False, "judge_calls": 0, "resume_count": 0,
                "frozen_config": frozen, "metric_policy": METRIC_POLICY,
                "dataset_validation": selection_report.to_dict(), "code_sha256": code_hashes,
                "smoke_manifest_sha256": _sha(smoke_manifest),
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: version(name) for name in ["datasets", "ollama", "pydantic", "pyarrow", "numpy", "scikit-learn"]},
            }
        completed, recovered = read_completed_rows(journal, cases)
        manifest.update(status="running", completed_cases=len(completed),
                        recovered_tail_bytes=manifest.get("recovered_tail_bytes", 0) + recovered)
        _atomic_json(manifest_path, manifest)
        print(f"Official cases: 500; completed: {len(completed)}; remaining: {500 - len(completed)}", flush=True)
        try:
            with journal.open("a", encoding="utf-8", newline="\n") as handle:
                for case in cases:
                    if case.pmid in completed:
                        continue
                    try:
                        row = capture_response(runner, case)
                    except BaseException as exc:
                        with (output / "errors.jsonl").open("a", encoding="utf-8", newline="\n") as errors:
                            _append_jsonl(errors, {"pmid": case.pmid, "at_utc": _now(), "error_type": type(exc).__name__})
                        raise
                    _append_jsonl(handle, row)
                    completed[case.pmid] = row
                    if len(completed) % 25 == 0:
                        manifest["completed_cases"] = len(completed)
                        _atomic_json(manifest_path, manifest)
                    print(f"{len(completed):03d}/500 PMID={case.pmid} latency={row['latency_seconds']:.3f}s", flush=True)
        except BaseException:
            manifest.update(status="interrupted", completed_cases=len(completed))
            _atomic_json(manifest_path, manifest)
            raise
        return finalize_run(cases, completed, selection_report, manifest, output, parquet=parquet)
