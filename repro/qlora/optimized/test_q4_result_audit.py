"""Offline input/metric checks. Real test gold is never read by these tests."""

from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import q4_result_audit as audit


class AuditChecks(unittest.TestCase):
    def toy_ids_and_rows(self):
        ids = tuple(str(index) for index in range(1, 501))
        return ids, [{"pmid": pmid, "prediction": "yes"} for pmid in reversed(ids)]

    def test_real_saved_inputs_cover_the_same_500_before_gold(self):
        with patch.object(audit, "get_final_gold") as gold:
            prepared = audit.prepare_inputs()
        self.assertEqual(len(prepared["ids"]), 500)
        for model in audit.MODELS:
            self.assertEqual(set(prepared["predictions"][model]), set(prepared["ids"]))
        gold.assert_not_called()

    def test_duplicate_ids_are_not_hidden_by_dictionary_conversion(self):
        ids, rows = self.toy_ids_and_rows()
        rows[-1]["pmid"] = rows[0]["pmid"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit.validate_predictions(rows, ids)

    def test_missing_extra_or_incomplete_prediction_ids_are_rejected(self):
        ids, rows = self.toy_ids_and_rows()
        with self.assertRaisesRegex(ValueError, "500"):
            audit.validate_predictions(rows[:-1], ids)
        rows[0]["pmid"] = "999999999"
        with self.assertRaisesRegex(ValueError, "missing=.*extra="):
            audit.validate_predictions(rows, ids)

    def test_invalid_prediction_labels_and_pmids_are_rejected(self):
        ids, rows = self.toy_ids_and_rows()
        rows[0]["prediction"] = "invalid"
        with self.assertRaisesRegex(ValueError, "label"):
            audit.validate_predictions(rows, ids)
        rows[0]["prediction"] = "yes"
        rows[0]["pmid"] = "0001"
        with self.assertRaises(ValueError):
            audit.validate_predictions(rows, ids)

    def test_changed_source_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (*audit.INPUT_SHA256, "source_manifest.json"):
                shutil.copyfile(audit.INPUTS / name, Path(directory) / name)
            path = Path(directory) / "base_predictions.json"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                audit.prepare_inputs(directory)

    def test_metrics_match_an_independently_counted_toy_matrix(self):
        result = audit.metrics_from_labels(["yes", "yes", "no", "maybe"],
                                          ["yes", "no", "no", "yes"])
        self.assertEqual(result["confusion_matrix"]["values"], [[1, 1, 0], [0, 1, 0], [1, 0, 0]])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 7 / 18)
        self.assertEqual(result["per_class"]["yes"]["f1"], 0.5)
        self.assertEqual(result["per_class"]["no"]["precision"], 0.5)
        self.assertEqual(result["per_class"]["no"]["recall"], 1.0)

    def test_absent_classes_have_zero_f1_and_stay_in_macro_average(self):
        result = audit.metrics_from_labels(["yes"] * 3, ["yes"] * 3)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(result["per_class"]["maybe"]["f1"], 0)

    def test_unfinished_exercise_blocks_all_gold_access(self):
        with patch.object(audit, "align_labels", side_effect=NotImplementedError("TEST UNFINISHED")), \
                patch.object(audit, "prepare_inputs") as inputs, patch.object(audit, "get_final_gold") as gold:
            with self.assertRaises(NotImplementedError):
                audit.audit()
        inputs.assert_not_called()
        gold.assert_not_called()

    def test_input_failure_stops_before_gold(self):
        with patch.object(audit, "check_exercise"), \
                patch.object(audit, "prepare_inputs", side_effect=ValueError("bad inputs")), \
                patch.object(audit, "get_final_gold") as gold:
            with self.assertRaisesRegex(ValueError, "bad inputs"):
                audit.audit()
        gold.assert_not_called()

    def test_invalid_alignment_stops_before_gold(self):
        with patch.object(audit, "check_exercise"), \
                patch.object(audit, "align_labels", return_value=["invalid"]), \
                patch.object(audit, "get_final_gold") as gold:
            with self.assertRaisesRegex(ValueError, "Aligned predictions"):
                audit.audit()
        gold.assert_not_called()

    def fixture(self):
        ids = ("1", "2", "3", "4")
        gold = {"1": "yes", "2": "yes", "3": "no", "4": "maybe"}
        predictions = {"4": "yes", "2": "no", "1": "yes", "3": "no"}
        reference_metrics = {"n": 4, "accuracy": 0.5, "macro_f1": 7 / 18,
                             "confusion_matrix": {"row_labels": list(audit.LABELS),
                                 "column_labels": list(audit.LABELS),
                                 "values": [[1, 1, 0], [0, 1, 0], [1, 0, 0]]}}
        prepared = {"ids": ids, "predictions": {model: predictions for model in audit.MODELS},
                    "reference": {"models": {model: deepcopy(reference_metrics) for model in audit.MODELS}},
                    "provenance": {"fixture": True}, "reference_sha256": "TEST ONLY"}
        return prepared, gold

    def test_complete_audit_aligns_by_pmid_and_matches_all_three_models(self):
        prepared, gold = self.fixture()
        with patch.object(audit, "align_labels", side_effect=lambda ids, values: [values[pmid] for pmid in ids]), \
                patch.object(audit, "prepare_inputs", return_value=prepared), \
                patch.object(audit, "get_final_gold", return_value=(gold, {"fixture": True})):
            report = audit.audit()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(all(result["matches_reference"] for result in report["models"].values()))
        self.assertFalse(report["new_inference"])
        self.assertFalse(report["Q4b_accessed"])

    def test_metric_mismatch_is_reported_instead_of_replacing_the_reference(self):
        prepared, gold = self.fixture()
        prepared["reference"]["models"]["A"]["accuracy"] = 0.9
        before = deepcopy(prepared["reference"])
        with patch.object(audit, "align_labels", side_effect=lambda ids, values: [values[pmid] for pmid in ids]), \
                patch.object(audit, "prepare_inputs", return_value=prepared), \
                patch.object(audit, "get_final_gold", return_value=(gold, {"fixture": True})):
            report = audit.audit()
        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(report["models"]["A"]["mismatched_fields"], ["accuracy"])
        self.assertEqual(prepared["reference"], before)

    def test_existing_audit_report_is_protected_before_rescoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text("EXISTING REPORT", encoding="utf-8")
            with patch.object(sys, "argv", ["audit", "--run", "--output", str(path)]), \
                    patch.object(audit, "audit") as run, patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    audit.main()
            self.assertEqual(stopped.exception.code, 2)
            run.assert_not_called()
            self.assertEqual(path.read_text("utf-8"), "EXISTING REPORT")

    def test_learner_function_can_remain_unfinished_or_be_completed(self):
        try:
            audit.check_exercise()
        except NotImplementedError:
            pass  # The delivered exercise is deliberately unfinished.


if __name__ == "__main__":
    unittest.main(verbosity=2)
