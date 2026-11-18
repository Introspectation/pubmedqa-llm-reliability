import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from medical_qa.__main__ import main
from medical_qa.baselines import baseline_predictions
from medical_qa.dataset import DatasetValidationError, SelectionReport
from medical_qa.scoring import final_score_baselines, load_gold_for_final_scoring, score_official_predictions


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.ids = tuple(str(i) for i in range(1, 501))
        # Toy unit-test labels only. No real dataset or real test gold is loaded.
        labels = ["yes"] * 275 + ["no"] * 150 + ["maybe"] * 75
        self.gold = dict(zip(self.ids, labels))
        self.report = SelectionReport(1000, 1000, 500, 500, 500, "toy-fingerprint")

    def test_known_always_yes_accuracy_macro_f1_and_per_class_f1(self):
        predictions = baseline_predictions(self.ids)
        for values in predictions.values():
            metrics = score_official_predictions(self.ids, self.gold, values)
            self.assertEqual(metrics["n"], 500)
            self.assertAlmostEqual(metrics["accuracy"], 0.55)
            self.assertAlmostEqual(metrics["per_class_f1"]["yes"], 22 / 31)
            self.assertEqual(metrics["per_class_f1"]["no"], 0)
            self.assertEqual(metrics["per_class_f1"]["maybe"], 0)
            self.assertAlmostEqual(metrics["macro_f1"], 22 / 93)
            self.assertAlmostEqual(metrics["per_class_precision"]["yes"], 0.55)
            self.assertEqual(metrics["per_class_recall"]["yes"], 1)
            self.assertEqual(metrics["confusion_matrix"]["values"], [[275, 0, 0], [150, 0, 0], [75, 0, 0]])

    def test_invalid_outputs_remain_in_denominator_and_confusion_matrix(self):
        predictions = {pmid: "invalid" for pmid in self.ids}
        metrics = score_official_predictions(self.ids, self.gold, predictions, allow_invalid=True)
        self.assertEqual(metrics["accuracy"], 0)
        self.assertEqual(metrics["macro_f1"], 0)
        self.assertEqual(metrics["confusion_matrix"]["column_labels"], ["yes", "no", "maybe", "invalid"])
        self.assertEqual(metrics["confusion_matrix"]["values"], [[0, 0, 0, 275], [0, 0, 0, 150], [0, 0, 0, 75]])
        with self.assertRaises(DatasetValidationError):
            score_official_predictions(self.ids, self.gold, predictions)

    def test_alignment_is_by_pmid_not_mapping_iteration_order(self):
        reversed_gold = dict(reversed(list(self.gold.items())))
        metrics = score_official_predictions(self.ids, self.gold, reversed_gold)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["macro_f1"], 1)
        self.assertEqual(metrics["per_class_f1"], {"yes": 1, "no": 1, "maybe": 1})

    def test_all_three_classes_stay_in_macro_average_when_absent(self):
        gold = {pmid: "yes" for pmid in self.ids}
        metrics = score_official_predictions(self.ids, gold, gold)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["per_class_f1"], {"yes": 1, "no": 0, "maybe": 0})
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 3)

    def test_partial_prediction_sets_cannot_be_scored(self):
        predictions = {pmid: "yes" for pmid in self.ids[:20]}
        with self.assertRaises(DatasetValidationError):
            score_official_predictions(self.ids, self.gold, predictions)

    def test_missing_or_reserved_ids_cannot_be_scored(self):
        wrong_gold = dict(self.gold)
        wrong_gold["1001"] = wrong_gold.pop("500")
        extra_predictions = {**self.gold, "1001": "no"}
        for gold, predictions in [(wrong_gold, self.gold), (self.gold, extra_predictions)]:
            with self.subTest(), self.assertRaises(DatasetValidationError):
                score_official_predictions(self.ids, gold, predictions)

    def test_invalid_gold_and_prediction_labels_are_rejected(self):
        for label in ["YES", "unknown", None, True, 1, ["yes"]]:
            wrong = {**self.gold, "1": label}
            for gold, predictions in [(wrong, self.gold), (self.gold, wrong)]:
                with self.subTest(label=label), self.assertRaises(DatasetValidationError):
                    score_official_predictions(self.ids, gold, predictions)

    def test_final_gold_source_must_pass_the_pinned_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory) / "wrong.parquet"
            wrong.write_bytes(b"not the pinned dataset")
            with patch("pyarrow.parquet.read_table") as read:
                with self.assertRaises(DatasetValidationError):
                    load_gold_for_final_scoring(parquet=wrong)
            read.assert_not_called()

    def test_final_scoring_freezes_predictions_before_opening_gold(self):
        events = []

        def freeze(ids):
            events.append("predictions")
            return baseline_predictions(ids)

        def read_gold(**kwargs):
            self.assertEqual(events, ["predictions"])
            events.append("gold")
            # Even when every toy gold label is no, published majority stays yes.
            return {pmid: "no" for pmid in self.ids}, self.report

        with patch("medical_qa.scoring.load_test_pmids", return_value=self.ids), \
             patch("medical_qa.scoring.baseline_predictions", side_effect=freeze), \
             patch("medical_qa.scoring.load_gold_for_final_scoring", side_effect=read_gold):
            report = final_score_baselines()
        self.assertEqual(events, ["predictions", "gold"])
        self.assertTrue(report["baselines_identical"])
        self.assertIn("identical to always_yes", report["identity_note"])
        self.assertEqual(report["baseline_definitions"]["published_majority"]["predicted_label"], "yes")
        self.assertEqual(report["results"]["always_yes"], report["results"]["published_majority"])
        self.assertEqual(report["results"]["published_majority"]["accuracy"], 0)
        self.assertEqual(report["results"]["published_majority"]["macro_f1"], 0)
        self.assertNotIn("gold_labels", report)


class GoldBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.ids = tuple(str(i) for i in range(1, 501))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "toy.parquet"
        self.rows = [{
            "pubid": i,
            "final_decision": "yes" if i <= 500 else "RESERVED_LABEL_MUST_NOT_BE_INSPECTED",
            "question": "NOT_NEEDED_FOR_FINAL_BASELINES",
            "long_answer": "NEVER_READ_CONCLUSION",
        } for i in reversed(range(1, 1001))]
        id_patch = patch("medical_qa.scoring.load_test_pmids", return_value=self.ids)
        path_patch = patch("medical_qa.scoring.resolve_parquet_source", return_value=self.path)
        id_patch.start()
        path_patch.start()
        self.addCleanup(id_patch.stop)
        self.addCleanup(path_patch.stop)

    def write_fixture(self):
        pq.write_table(pa.Table.from_pylist(self.rows), self.path)

    def test_only_filtered_official_labels_are_exposed(self):
        self.write_fixture()
        with patch("pyarrow.parquet.read_table", wraps=pq.read_table) as read:
            gold, report = load_gold_for_final_scoring(parquet=self.path)
        self.assertEqual(gold, {pmid: "yes" for pmid in self.ids})
        self.assertEqual(report.matched_unique_cases, 500)
        self.assertEqual(report.excluded_cases, 500)
        self.assertEqual(read.call_count, 2)
        self.assertEqual(read.call_args_list[0].kwargs, {"columns": ["pubid"]})
        self.assertEqual(read.call_args_list[1].kwargs["columns"], ["pubid", "final_decision"])
        self.assertEqual(read.call_args_list[1].kwargs["filters"], [("pubid", "in", list(range(1, 501)))])

    def test_missing_official_id_fails_before_reading_labels(self):
        self.rows[-1]["pubid"] = 1001
        self.write_fixture()
        with patch("pyarrow.parquet.read_table", wraps=pq.read_table) as read:
            with self.assertRaises(DatasetValidationError):
                load_gold_for_final_scoring(parquet=self.path)
        self.assertEqual(read.call_count, 1)

    def test_duplicate_source_id_fails_before_reading_labels(self):
        self.rows[0]["pubid"] = self.rows[1]["pubid"]
        self.write_fixture()
        with patch("pyarrow.parquet.read_table", wraps=pq.read_table) as read:
            with self.assertRaises(DatasetValidationError):
                load_gold_for_final_scoring(parquet=self.path)
        self.assertEqual(read.call_count, 1)

    def test_invalid_official_label_fails_without_echoing_its_value(self):
        self.rows[-1]["final_decision"] = "INVALID_GOLD_SENTINEL"
        self.write_fixture()
        with self.assertRaises(DatasetValidationError) as caught:
            load_gold_for_final_scoring(parquet=self.path)
        self.assertNotIn("INVALID_GOLD_SENTINEL", str(caught.exception))


class FinalScoringCliTests(unittest.TestCase):
    def test_blind_validation_and_smoke_commands_do_not_open_gold(self):
        report = SelectionReport(1000, 1000, 500, 500, 500, "toy-fingerprint")
        for command in ["validate-dataset", "smoke"]:
            with self.subTest(command=command), \
                 patch("sys.argv", ["medical_qa", command]), redirect_stdout(io.StringIO()), \
                 patch("medical_qa.__main__.load_dataset", return_value=((), report)), \
                 patch("medical_qa.__main__.run_smoke", return_value={"inference_errors": 0}), \
                 patch("medical_qa.scoring.load_gold_for_final_scoring") as read_gold:
                self.assertEqual(main(), 0)
            read_gold.assert_not_called()

    def test_final_command_is_separate_from_inference_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            report = {"gold_scoring_performed": True, "baselines_identical": True, "results": {}}
            argv = ["medical_qa", "final-score-baselines", "--parquet", "toy.parquet", "--output", str(output)]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()), \
                 patch("medical_qa.scoring.final_score_baselines", return_value=report) as score, \
                 patch("medical_qa.__main__.load_dataset", side_effect=AssertionError("No blind dataset load")), \
                 patch("medical_qa.__main__.OllamaRunner", side_effect=AssertionError("No Ollama")):
                self.assertEqual(main(), 0)
            score.assert_called_once_with(parquet=Path("toy.parquet"))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)

    def test_existing_final_report_is_refused_before_opening_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("original", encoding="utf-8")
            argv = ["medical_qa", "final-score-baselines", "--output", str(output)]
            with patch("sys.argv", argv), redirect_stderr(io.StringIO()), \
                 patch("medical_qa.scoring.final_score_baselines") as score:
                self.assertEqual(main(), 1)
            score.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "original")
