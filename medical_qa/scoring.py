"""Gold access is confined to an explicitly invoked final scoring operation."""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .baselines import LABELS, baseline_definitions, baseline_predictions
from .dataset import (
    DatasetValidationError, canonical_pmid, load_test_pmids, official_test_indices,
    resolve_parquet_source, source_lock, validate_pmids,
)


def load_gold_for_final_scoring(*, parquet: Path | None = None):
    """Read only official test labels, with no questions, contexts, or conclusions.

    The first read projects IDs alone to validate source uniqueness and membership.
    The second read projects labels with an official-ID filter. Reserved labels are
    never converted to Python values, displayed, analyzed, or written to a cache.
    """
    import pyarrow.parquet as pq

    ids = load_test_pmids()
    path = resolve_parquet_source(parquet)
    source_ids = pq.read_table(path, columns=["pubid"]).column("pubid").to_pylist()
    _, report = official_test_indices(source_ids, ids)
    selected = pq.read_table(
        path, columns=["pubid", "final_decision"],
        filters=[("pubid", "in", [int(pmid) for pmid in ids])],
    )
    if selected.num_rows != len(ids):
        raise DatasetValidationError("Final scoring requires exactly 500 official test labels.")
    gold = {}
    for row in selected.to_pylist():
        pmid, label = canonical_pmid(row["pubid"]), row["final_decision"]
        if pmid not in ids or pmid in gold:
            raise DatasetValidationError("Final gold IDs differ from the official test list.")
        if not isinstance(label, str) or label not in LABELS:
            raise DatasetValidationError("Invalid official gold label at final scoring.")
        gold[pmid] = label
    if set(gold) != set(ids):
        raise DatasetValidationError("Final gold IDs differ from the official test list.")
    return gold, report


def score_official_predictions(official_pmids, gold: dict[str, str], predictions: dict[str, str],
                               *, allow_invalid: bool = False) -> dict:
    """Align by PMID, reject partial/extra records, and score all three classes."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    ids = validate_pmids(official_pmids)
    if set(gold) != set(ids) or set(predictions) != set(ids):
        raise DatasetValidationError("Gold and predictions must each cover exactly the official 500 PMIDs.")
    truth, predicted = [gold[pmid] for pmid in ids], [predictions[pmid] for pmid in ids]
    prediction_labels = LABELS + (("invalid",) if allow_invalid else ())
    if any(not isinstance(label, str) or label not in LABELS for label in truth) or any(
        not isinstance(label, str) or label not in prediction_labels for label in predicted
    ):
        raise DatasetValidationError("Scoring labels must be yes, no, or maybe.")
    precision, recall, per_class, support = precision_recall_fscore_support(
        truth, predicted, labels=LABELS, zero_division=0,
    )
    matrix = confusion_matrix(truth, predicted, labels=prediction_labels)[:len(LABELS)]
    return {
        "n": len(ids),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(sum(per_class) / len(LABELS)),
        "per_class_f1": {label: float(value) for label, value in zip(LABELS, per_class)},
        "per_class_precision": {label: float(value) for label, value in zip(LABELS, precision)},
        "per_class_recall": {label: float(value) for label, value in zip(LABELS, recall)},
        "per_class_support": {label: int(value) for label, value in zip(LABELS, support)},
        "confusion_matrix": {
            "row_labels": list(LABELS), "column_labels": list(prediction_labels),
            "values": matrix.tolist(),
        },
    }


def final_score_baselines(*, parquet: Path | None = None) -> dict:
    """Explicit final scoring boundary; never called by loading or smoke tests."""
    ids = load_test_pmids()
    # Freeze the ID-only predictions before any test gold is read.
    predictions = baseline_predictions(ids)
    definitions = baseline_definitions()
    gold, report = load_gold_for_final_scoring(parquet=parquet)
    return {
        "stage": "final_baseline_scoring",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_scoring_performed": True,
        "gold_source": "Pinned HF pqa_labeled final_decision, selected by official test PMID.",
        "dataset_validation": report.to_dict(),
        "sources": source_lock(),
        "scikit_learn_version": version("scikit-learn"),
        "label_order": list(LABELS),
        "zero_division": 0,
        "baseline_definitions": definitions,
        "baselines_identical": predictions["always_yes"] == predictions["published_majority"],
        "identity_note": "Published PQA-L majority is yes; published_majority is identical to always_yes.",
        "results": {name: score_official_predictions(ids, gold, values)
                    for name, values in predictions.items()},
    }
