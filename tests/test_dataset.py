import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

from medical_qa.dataset import (
    DatasetValidationError, canonical_pmid, load_dataset, load_test_pmids,
    parse_official_test_pmids, select_official_cases, validate_pmids,
)


def toy_dataset():
    # Deliberately interleave held-out rows with test rows. Excluded text is null;
    # selection must occur before validating text, not after reading every row.
    ids = list(range(1, 1001))[::-1]
    return Dataset.from_dict({
        "pubid": ids,
        "question": [f"Toy question {i}?" if i <= 500 else None for i in ids],
        "context": [
            {"contexts": [f"A toy abstract for record {i}."],
             "labels": ["GOLD_HINT_SENTINEL"], "reasoning_free_pred": ["HINT_SENTINEL"]}
            if i <= 500 else None for i in ids
        ],
        "final_decision": ["GOLD_SENTINEL"] * 1000,
        "long_answer": ["CONCLUSION_SENTINEL"] * 1000,
    })


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.ids = tuple(str(i) for i in range(1, 501))

    def test_exact_official_selection_from_hf_train_style_source(self):
        cases, report = select_official_cases(toy_dataset(), self.ids)
        self.assertEqual(tuple(case.pmid for case in cases), self.ids)
        self.assertEqual(report.matched_unique_cases, 500)
        self.assertEqual(report.excluded_cases, 500)
        self.assertEqual(report.source_unique_pmids, 1000)
        exported = json.dumps([asdict(case) for case in cases])
        for forbidden in ["GOLD_SENTINEL", "CONCLUSION_SENTINEL", "HINT_SENTINEL", "final_decision", "long_answer"]:
            self.assertNotIn(forbidden, exported)
        self.assertEqual(set(asdict(cases[0])), {"pmid", "question", "contexts"})

    def test_official_json_uses_keys_without_gold_values(self):
        raw = json.dumps({pmid: {"unused": "DO_NOT_INSPECT_GOLD"} for pmid in self.ids}).encode()
        self.assertEqual(parse_official_test_pmids(raw), self.ids)

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(DatasetValidationError):
            parse_official_test_pmids(b'{"1":"yes","1":"no"}')

    def test_official_json_must_be_an_object(self):
        with self.assertRaises(DatasetValidationError):
            parse_official_test_pmids(json.dumps(list(self.ids)).encode())

    def test_wrong_id_count_and_duplicate_ids_are_rejected(self):
        for ids in [self.ids[:-1], self.ids + ("501",), self.ids[:-1] + ("1",)]:
            with self.subTest(count=len(ids)), self.assertRaises(DatasetValidationError):
                validate_pmids(ids)

    def test_missing_official_match_is_rejected(self):
        ids = self.ids[:-1] + ("1001",)
        with self.assertRaises(DatasetValidationError):
            select_official_cases(toy_dataset(), ids)

    def test_duplicate_source_pmid_is_rejected(self):
        dataset = toy_dataset()
        dataset = dataset.select(list(range(999)) + [998])
        with self.assertRaises(DatasetValidationError):
            select_official_cases(dataset, self.ids)

    def test_wrong_source_size_is_rejected(self):
        with self.assertRaises(DatasetValidationError):
            select_official_cases(toy_dataset().select(range(999)), self.ids)

    def test_malformed_selected_abstract_is_rejected(self):
        dataset = toy_dataset().to_dict()
        dataset["context"][-1]["contexts"] = []
        with self.assertRaises(DatasetValidationError):
            select_official_cases(Dataset.from_dict(dataset), self.ids)

    def test_pmid_types_are_strict(self):
        self.assertEqual(canonical_pmid(123), "123")
        for value in [True, 1.0, "01", " 1", "-1", "0", "", None]:
            with self.subTest(value=value), self.assertRaises(DatasetValidationError):
                canonical_pmid(value)

    def test_vendored_official_list_matches_pinned_fingerprint(self):
        ids = load_test_pmids()
        self.assertEqual(len(ids), 500)
        self.assertEqual(len(set(ids)), 500)

    def test_tampered_official_id_cache_is_rejected(self):
        with patch("medical_qa.dataset.source_lock", return_value={"test_pmids_sha256": "0" * 64}):
            with self.assertRaises(DatasetValidationError):
                load_test_pmids()

    def test_wrong_parquet_checksum_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.parquet"
            path.write_bytes(b"wrong source")
            with self.assertRaises(DatasetValidationError):
                load_dataset(parquet=path)
