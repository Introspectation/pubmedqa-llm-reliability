"""Prepare the fixed pilot. Gold is used for stratification only, never exported."""

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_qa.dataset import load_dataset, load_test_pmids, source_lock
from medical_qa.scoring import load_gold_for_final_scoring
from medical_qa.validation import parse_answer, validate_evidence
from scripts.run_judge import INPUT_FIELDS, atomic_json, now, text_hash


def select_pilot_ids(eligible, gold, *, seed=42):
    correct = sorted((pmid for pmid, row in eligible.items() if row["answer"] == gold[pmid]), key=int)
    wrong = sorted((pmid for pmid, row in eligible.items() if row["answer"] != gold[pmid]), key=int)
    if len(correct) < 10 or len(wrong) < 10:
        raise ValueError("Pilot requires at least ten eligible correct and ten eligible wrong answers.")
    rng = random.Random(seed)
    selected = rng.sample(correct, 10) + rng.sample(wrong, 10)
    rng.shuffle(selected)  # Do not reveal strata through row order.
    return selected


def write_pilot(output, selected, eligible, rubric, provenance):
    if len(selected) != 20 or len(set(selected)) != 20 or not set(selected).issubset(eligible):
        raise ValueError("Pilot must contain 20 unique eligible cases.")
    inputs = [eligible[pmid] for pmid in selected]
    if any(set(row) != INPUT_FIELDS for row in inputs):
        raise ValueError("Only blinded pilot fields may be exported.")
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite human annotations.
    atomic_json(output / "pilot_ids.json", selected)
    with (output / "pilot_inputs.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in inputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "human_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pmid", "question", "quote", "answer", "my_label"], lineterminator="\n")
        writer.writeheader()
        for row in inputs:
            writer.writerow({key: row[key] for key in ["pmid", "question", "quote", "answer"]} | {"my_label": ""})
    manifest = {
        "created_at_utc": now(), "seed": 42, "sample_size": 20,
        "stratum_sample_sizes": {"correct": 10, "wrong": 10},
        "selection_method": "Numerically sorted pools; Random(42).sample(correct,10), sample(wrong,10), then shuffle.",
        "gold_usage": "Preparation/stratification only. No gold or per-case correctness is exported.",
        "human_labels_policy": "Human-owned file; never read by the judge runner or overwritten on preparation.",
        "hash_method": "New pilot/rubric hashes use UTF-8 text with LF-normalized newlines.",
        "pilot_ids_sha256": text_hash(output / "pilot_ids.json"),
        "pilot_inputs_sha256": text_hash(output / "pilot_inputs.jsonl"),
        "human_labels_template_sha256": text_hash(output / "human_labels.csv"),
        "rubric_sha256": text_hash(rubric), "judge_calls_during_preparation": 0,
        **provenance,
    }
    atomic_json(output / "pilot_manifest.json", manifest)
    return manifest


def prepare_pilot(phase1_dir, predictions, parquet, output, rubric):
    if output.exists():
        raise FileExistsError("Pilot output already exists; refusing to overwrite IDs or human labels.")
    summary_path = phase1_dir / "results-summary.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    if summary["n"] != 500 or summary["frozen_config"]["sources"] != source_lock():
        raise ValueError("Phase 1 summary does not match the pinned official-500 source.")
    if hashlib.sha256(predictions.read_bytes()).hexdigest() != summary["validated_predictions_sha256"]:
        raise ValueError("Validated prediction bytes differ from the published Phase 1 checksum.")
    rows = [json.loads(line) for line in predictions.read_text("utf-8").splitlines()]
    official = set(load_test_pmids())
    if len(rows) != 500 or {row["pmid"] for row in rows} != official:
        raise ValueError("Phase 1 predictions must contain all 500 unique official IDs.")
    cases, selection = load_dataset(parquet=parquet)
    by_id = {case.pmid: case for case in cases}
    eligible = {}
    for row in rows:
        if row["status"] != "grounded":
            continue
        case = by_id[row["pmid"]]
        answer = parse_answer(row["raw_response"])
        if (not row["schema_valid"] or not row["protocol_valid"]
                or not validate_evidence(answer.evidence, case.contexts).grounded):
            raise ValueError("A Phase 1 grounded row failed the frozen quote/schema checks.")
        eligible[case.pmid] = {
            "pmid": case.pmid, "question": case.question,
            "abstract": "\n\n".join(case.contexts), "quote": answer.evidence, "answer": answer.label,
        }
    if len(eligible) != 483 or len(eligible) != summary["counts"]["grounded"]:
        raise ValueError("Expected exactly 483 quote-matched Phase 1 cases.")
    gold, gold_selection = load_gold_for_final_scoring(parquet=parquet)
    if selection != gold_selection:
        raise ValueError("Gold and input cohorts differ.")
    selected = select_pilot_ids(eligible, gold, seed=42)
    return write_pilot(output, selected, eligible, rubric, {
        "eligible_quote_matched_cases": len(eligible),
        "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "source_validated_predictions_sha256": summary["validated_predictions_sha256"],
        "source_frozen_config": summary["frozen_config"],
    })


def main():
    parser = argparse.ArgumentParser(description="Prepare a seed-42 10-correct/10-wrong gold-blind judge pilot; never run a judge.")
    parser.add_argument("--phase1-dir", type=Path, default=ROOT / "results/phase1-full")
    parser.add_argument("--predictions", type=Path, default=ROOT / "runs/phase1-full/validated-predictions.jsonl")
    parser.add_argument("--parquet", type=Path, help="Offline copy of the pinned HF Parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "results/judge-pilot")
    parser.add_argument("--rubric", type=Path, default=ROOT / "prompts/judge_rubric_v1.txt")
    args = parser.parse_args()
    try:
        manifest = prepare_pilot(args.phase1_dir, args.predictions, args.parquet, args.output, args.rubric)
        print(json.dumps({"output": str(args.output), "sample_size": manifest["sample_size"],
                          "eligible": manifest["eligible_quote_matched_cases"], "strata": manifest["stratum_sample_sizes"],
                          "judge_calls": 0}, indent=2))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
