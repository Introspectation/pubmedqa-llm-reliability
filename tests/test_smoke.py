import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from medical_qa.dataset import Case, DatasetValidationError, SelectionReport
from medical_qa.smoke import run_smoke, select_smoke_cases


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.cases = tuple(Case(str(i), "Toy question?", ("A toy abstract for offline unit tests.",))
                           for i in range(1, 501))

    def test_twenty_unique_cases_are_independent_of_source_order(self):
        first = select_smoke_cases(self.cases)
        self.assertEqual(len(first), 20)
        self.assertEqual(len({case.pmid for case in first}), 20)
        self.assertEqual(first, select_smoke_cases(self.cases[::-1]))

    def test_sample_does_not_depend_on_question_or_abstract(self):
        changed = tuple(Case(case.pmid, "Changed text?", ("Different abstract.",)) for case in self.cases)
        self.assertEqual([case.pmid for case in select_smoke_cases(self.cases)],
                         [case.pmid for case in select_smoke_cases(changed)])

    def test_non_500_input_is_rejected(self):
        with self.assertRaises(DatasetValidationError):
            select_smoke_cases(self.cases[:-1])

    def test_failures_are_saved_and_counted_without_retry(self):
        runner = Mock()
        runner.verify_model.return_value = "test-digest"
        runner.server_version = "test-version"
        runner.predict.side_effect = TimeoutError("test timeout")
        report = SelectionReport(1000, 1000, 500, 500, 500, "test-fingerprint")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            with patch("builtins.print"):
                summary = run_smoke(self.cases, report, output, runner=runner)
            self.assertEqual(summary["attempted"], 20)
            self.assertEqual(summary["inference_errors"], 20)
            self.assertEqual(summary["judge_calls"], 0)
            self.assertFalse(summary["gold_scoring_performed"])
            self.assertEqual(runner.predict.call_count, 20)
            rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual([row["pmid"] for row in rows], manifest["pmids"])
            self.assertEqual(manifest["status"], "completed_with_errors")
            with self.assertRaises(FileExistsError):
                run_smoke(self.cases, report, output, runner=runner)
