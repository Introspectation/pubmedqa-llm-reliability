"""Q5.1: re-score saved Q4 predictions. No inference, training or Q4b access.

Learner task: complete one dictionary lookup in align_labels().
Run --check-inputs first; --run stays blocked until the exercise passes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FOLDER = Path(__file__).resolve().parent
INPUTS = FOLDER / "q4_audit_inputs"
LABELS = ("yes", "no", "maybe")
MODELS = ("base", "A", "B")
# The reference pin below is the original downloaded Q4 aggregate, not a new result.
REFERENCE_SHA256 = "731cf4e8a96ba3b8e1daca57436463fcd6b53cc82b7c97e416139a21559af70a"
INPUT_SHA256 = {
    "base_predictions.json": "515c1b969748cd25f3fc44131198eecc432e264de3a92c65ca3e7cba5bf543c1",
    "A_predictions.json": "575ff9ab7f7aa73b0471bb10eb7985e1d8363f2c145b3a8842a60cc75a3ed649",
    "B_predictions.json": "1492a4d5eeef3cacab5a3a8ae5a9febfaf241c1430eaddf49880af01c3c92095",
    "manifest.json": "415a290922cd554d2676b3ddb60d9a3d0282fb70b61630ceed442ecfb5b1de6d",
}


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def align_labels(ordered_pmids, labels_by_pmid):
    """Return the labels in PMID order, regardless of input file row order."""
    aligned = []
    for pmid in ordered_pmids:
        # SENIN SATIRIN: bu PMID'nin degerini sozlukten al.
        # labels_by_pmid bir sozluk; pmid o an aradigimiz anahtar.
        # Baska bir ornek: city_by_code['34'] o anahtarin degerini verir.
        label = labels_by_pmid[pmid]
        if label is Ellipsis:
            raise NotImplementedError("align_labels icindeki sozluk erisimini sen tamamla.")
        aligned.append(label)
    return aligned


def check_exercise():
    # Two different orders deliberately expose accidental row-by-row alignment.
    sample = {"10": "label-for-10", "20": "label-for-20"}
    if align_labels(["20", "10"], sample) != ["label-for-20", "label-for-10"]:
        raise ValueError("Etiketler dosya sirasina gore degil, istenen PMID sirasina gore alinmali.")
    if align_labels(["10", "20"], sample) != ["label-for-10", "label-for-20"]:
        raise ValueError("Her PMID kendi etiketini almali.")


def validate_predictions(rows, official_ids):
    """The completed set idea: count, uniqueness, membership, then label validity."""
    from medical_qa.dataset import canonical_pmid

    if not isinstance(rows, list) or len(rows) != 500:
        raise ValueError("Each model needs exactly 500 prediction rows.")
    ids = []
    result = {}
    for row in rows:
        if not isinstance(row, dict) or "pmid" not in row or "prediction" not in row:
            raise ValueError("Each prediction needs pmid and prediction fields.")
        pmid = canonical_pmid(row["pmid"])
        if row["prediction"] not in LABELS:
            raise ValueError("Unexpected prediction label.")
        ids.append(pmid)
        result[pmid] = row["prediction"]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prediction PMIDs; dictionary conversion must not hide them.")
    if set(ids) != set(official_ids):
        missing = sorted(set(official_ids) - set(ids), key=int)
        extra = sorted(set(ids) - set(official_ids), key=int)
        raise ValueError(f"Official PMID mismatch: missing={missing[:10]}, extra={extra[:10]}")
    return result


def prepare_inputs(input_dir=INPUTS, reference_path=FOLDER / "q4_heldout_eval.json"):
    """No gold columns, questions, abstracts, or Q4b files are read here."""
    from medical_qa.dataset import load_test_pmids

    ids = load_test_pmids()
    input_dir, reference_path = Path(input_dir), Path(reference_path)
    reference_raw = reference_path.read_bytes()
    if sha256(reference_raw) != REFERENCE_SHA256:
        raise ValueError("Original Q4 aggregate checksum differs.")
    reference = json.loads(reference_raw)
    if reference.get("status") != "complete" or reference.get("n") != 500:
        raise ValueError("Original Q4 must be complete.")
    provenance = json.loads((input_dir / "source_manifest.json").read_bytes())
    entries = {item["file"]: item for item in provenance["files"]}
    if len(provenance["files"]) != 4 or set(entries) != set(INPUT_SHA256):
        raise ValueError("Input provenance must list exactly the four source files.")
    objects = {}
    for name, expected in INPUT_SHA256.items():
        raw = (input_dir / name).read_bytes()
        if sha256(raw) != expected or entries[name]["sha256"] != expected or len(raw) != entries[name]["bytes"]:
            raise ValueError("Original Q4 input checksum differs: " + name)
        objects[name] = json.loads(raw)
    manifest = objects["manifest.json"]
    if manifest["protocol"] != reference["protocol"] or manifest["adapters"] != reference["adapters"]:
        raise ValueError("Prediction manifest and Q4 aggregate identify different runs.")
    predictions = {}
    for model in MODELS:
        saved = objects[model + "_predictions.json"]
        predictions[model] = validate_predictions(saved["predictions"], ids)
        if saved["timing"] != reference["models"][model]["timing"]:
            raise ValueError("Prediction timing/provenance differs: " + model)
    return {"ids": ids, "predictions": predictions, "reference": reference,
            "provenance": provenance, "reference_sha256": REFERENCE_SHA256}


def metrics_from_labels(truth, predicted):
    if not truth or len(truth) != len(predicted) or any(label not in LABELS for label in truth + predicted):
        raise ValueError("Nonempty, equally sized valid label lists are required.")
    matrix = [[0] * 3 for _ in LABELS]
    for actual, prediction in zip(truth, predicted):
        matrix[LABELS.index(actual)][LABELS.index(prediction)] += 1
    n = len(truth)
    correct = sum(matrix[index][index] for index in range(3))
    per_class = {}
    for index, label in enumerate(LABELS):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        # 2TP/(2TP+FP+FN) is equivalent to 2PR/(P+R), including zero cases.
        f1 = 2 * tp / (support + predicted_count) if support + predicted_count else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": support, "predicted_count": predicted_count}
    return {"n": n, "correct": correct, "accuracy": correct / n,
            "macro_f1": sum(item["f1"] for item in per_class.values()) / 3,
            "per_class": per_class,
            "confusion_matrix": {"row_labels": list(LABELS), "column_labels": list(LABELS), "values": matrix}}


def compare_metrics(actual, expected):
    mismatches = []
    if actual["n"] != expected["n"]:
        mismatches.append("n")
    for name in ("accuracy", "macro_f1"):
        if not math.isclose(actual[name], expected[name], rel_tol=0, abs_tol=1e-12):
            mismatches.append(name)
    if actual["confusion_matrix"] != expected["confusion_matrix"]:
        mismatches.append("confusion_matrix")
    return mismatches


def get_final_gold(parquet):
    # Existing loader projects ONLY pubid/final_decision for official test IDs.
    # The other 500 labels, questions and abstracts are not exposed to this audit.
    from medical_qa.scoring import load_gold_for_final_scoring
    gold, selection = load_gold_for_final_scoring(parquet=parquet)
    return gold, selection.to_dict()


def audit(input_dir=INPUTS, reference_path=FOLDER / "q4_heldout_eval.json", parquet=None):
    check_exercise()  # Learning gate executes before any gold access.
    prepared = prepare_inputs(input_dir, reference_path)
    ids = prepared["ids"]
    predictions = {model: align_labels(ids, prepared["predictions"][model]) for model in MODELS}
    if any(len(labels) != len(ids) or any(label not in LABELS for label in labels)
           for labels in predictions.values()):
        raise ValueError("Aligned predictions are invalid; stopped before gold access.")
    # All three official-500 prediction sets have been verified and frozen above.
    gold, selection = get_final_gold(parquet)
    if set(gold) != set(ids):
        raise ValueError("Gold IDs differ from prediction IDs.")
    truth = align_labels(ids, gold)
    models = {}
    for model in MODELS:
        recomputed = metrics_from_labels(truth, predictions[model])
        expected = prepared["reference"]["models"][model]
        mismatches = compare_metrics(recomputed, expected)
        models[model] = {"recomputed": recomputed, "matches_reference": not mismatches,
                         "mismatched_fields": mismatches,
                         "reference_accuracy": expected["accuracy"], "reference_macro_f1": expected["macro_f1"]}
    return {"status": "passed" if all(m["matches_reference"] for m in models.values()) else "mismatch",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "audit_type": "per_PMID_rescore_of_frozen_Q4_predictions", "n": len(ids),
            "models": models, "gold_selection": selection,
            "source_files": prepared["provenance"], "reference_sha256": prepared["reference_sha256"],
            "audit_script_sha256": sha256(Path(__file__).read_bytes()),
            "new_inference": False, "training": False, "Q4b_accessed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-inputs", action="store_true")
    modes.add_argument("--check-exercise", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--parquet", type=Path, default=ROOT / ".cache/train-00000-of-00001.parquet")
    parser.add_argument("--output", type=Path, default=FOLDER / "q4_rescore_validation.json")
    args = parser.parse_args()
    try:
        if args.check_inputs:
            prepared = prepare_inputs()
            print(json.dumps({"status": "inputs_verified", "models": {
                model: {"rows": len(values), "unique_official_pmids": len(values)}
                for model, values in prepared["predictions"].items()}, "gold_read": False}, indent=2))
            return
        if args.check_exercise:
            check_exercise()
            print("Alignment exercise passed. No gold labels were read.")
            return
        if args.output.exists():
            raise ValueError("Output exists; refusing overwrite. Use a new --output path.")
        report = audit(parquet=args.parquet)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print("Audit status:", report["status"])
        for model, result in report["models"].items():
            metrics = result["recomputed"]
            print(model, "accuracy:", metrics["accuracy"], "macro-F1:", metrics["macro_f1"],
                  "matches reference:", result["matches_reference"])
        print("New report:", args.output)
        if report["status"] != "passed":
            raise SystemExit(1)
    except (ValueError, NotImplementedError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
