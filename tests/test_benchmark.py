import copy
import hashlib
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from medical_qa.benchmark import (
    capture_response, read_completed_rows, request_fingerprint, run_full_benchmark,
    validate_completed_cases, verify_smoke_reference,
)
from medical_qa.dataset import ROOT, Case, DatasetValidationError, SelectionReport, load_test_pmids, pmid_fingerprint, source_lock
from medical_qa.inference import MODEL, OPTIONS, PROMPT_SHA256, OllamaRunner
from medical_qa.validation import RESPONSE_SCHEMA


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        # Public IDs only; every question, abstract, model response and gold below is a toy fixture.
        self.ids = load_test_pmids()
        self.cases = tuple(Case(pmid, "Toy question?", ("A sufficiently long toy quotation for unit testing.",))
                           for pmid in self.ids)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.smoke = Path(self.directory.name) / "toy-smoke-manifest.json"
        self.selection = SelectionReport(1000, 1000, 500, 500, 500, pmid_fingerprint(self.ids))
        # A portable synthetic manifest: tests do not require ignored local runs.
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        self.reference = {
            "status": "completed", "sample_size": 20, "model": MODEL,
            "model_digest": json.loads((ROOT / "model.lock.json").read_text())["digest"],
            "ollama_server_version": "toy-server-version", "options": OPTIONS,
            "think": False, "response_schema": RESPONSE_SCHEMA, "prompt_sha256": PROMPT_SHA256,
            "sources": source_lock(), "requirements_sha256": sha(ROOT / "requirements.txt"),
            "code_sha256": {name: sha(ROOT / "medical_qa" / name) for name in ["inference.py", "validation.py"]},
            "packages": {name: version(name) for name in ["datasets", "ollama", "pydantic", "pyarrow"]},
            "dataset_validation": self.selection.to_dict(),
        }
        self.smoke.write_text(json.dumps(self.reference), encoding="utf-8")
        self.runner = OllamaRunner(client=Mock())
        self.runner.verify_model = Mock(return_value=self.reference["model_digest"])
        self.runner.server_version = self.reference["ollama_server_version"]
        self.response = SimpleNamespace(
            done=True, done_reason="stop", prompt_eval_count=100, eval_count=20,
            message=SimpleNamespace(thinking=None, content=json.dumps({
                "label": "yes", "evidence": self.cases[0].contexts[0],
            })),
        )
        self.runner.client.chat.return_value = self.response
        self.output = Path(self.directory.name) / "full"

    def row(self, case):
        return {
            "pmid": case.pmid, "request_sha256": request_fingerprint(case),
            "raw_response": self.response.message.content, "latency_seconds": 1.0,
            "done": True, "done_reason": "stop", "thinking_returned": False,
            "prompt_eval_count": 100, "eval_count": 20,
        }

    def test_full_request_is_identical_to_frozen_smoke_request(self):
        self.runner.predict(self.cases[0])
        smoke_kwargs = copy.deepcopy(self.runner.client.chat.call_args.kwargs)
        row = capture_response(self.runner, self.cases[0])
        full_kwargs = self.runner.client.chat.call_args.kwargs
        self.assertEqual(json.dumps(smoke_kwargs), json.dumps(full_kwargs))
        self.assertEqual(row["request_sha256"], request_fingerprint(self.cases[0]))
        self.assertNotIn("answer", row)  # Validation is a later stage.

    def test_frozen_reference_rejects_changed_digest_settings_and_schema_order(self):
        self.runner.verify_model.return_value = "changed-digest"
        with self.assertRaises(ValueError):
            verify_smoke_reference(self.smoke, self.runner)
        self.runner.verify_model.return_value = self.reference["model_digest"]
        with patch("medical_qa.benchmark.OPTIONS", {"temperature": 1}):
            with self.assertRaises(ValueError):
                verify_smoke_reference(self.smoke, self.runner)
        schema = copy.deepcopy(self.reference["response_schema"])
        schema["properties"] = dict(reversed(list(schema["properties"].items())))
        with patch("medical_qa.benchmark.RESPONSE_SCHEMA", schema):
            with self.assertRaises(ValueError):
                verify_smoke_reference(self.smoke, self.runner)
        self.runner.client.chat.assert_not_called()

    def test_resume_skips_completed_ids_and_gold_waits_for_all_validations(self):
        self.runner.client.chat.side_effect = [self.response, ConnectionError("toy interruption")]
        with patch("medical_qa.benchmark.load_gold_for_final_scoring") as gold, redirect_stdout(io.StringIO()):
            with self.assertRaises(ConnectionError):
                run_full_benchmark(self.cases, self.selection, self.output, self.smoke, runner=self.runner)
            gold.assert_not_called()
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")
        self.assertFalse(manifest["gold_scoring_performed"])
        self.assertEqual(manifest["completed_cases"], 1)
        # Seed previously completed toy records, leaving exactly one request pending.
        with (self.output / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            for case in self.cases[1:-1]:
                handle.write(json.dumps(self.row(case)) + "\n")
        self.runner.client.chat.side_effect = None
        self.runner.client.chat.reset_mock()

        def final_gold(**kwargs):
            raw = (self.output / "predictions.jsonl").read_text().splitlines()
            checked = (self.output / "validated-predictions.jsonl").read_text().splitlines()
            self.assertEqual(len(raw), 500)
            self.assertEqual(len(checked), 500)
            stage = json.loads((self.output / "manifest.json").read_text())
            self.assertEqual(stage["status"], "validated")
            return {pmid: "yes" for pmid in self.ids}, self.selection

        with patch("medical_qa.benchmark.load_gold_for_final_scoring", side_effect=final_gold), \
             redirect_stdout(io.StringIO()):
            summary = run_full_benchmark(self.cases, self.selection, self.output, self.smoke, runner=self.runner)
        self.assertEqual(self.runner.client.chat.call_count, 1)
        completed, _ = read_completed_rows(self.output / "predictions.jsonl", self.cases)
        self.assertEqual(set(completed), set(self.ids))
        self.assertEqual(summary["counts"], {"grounded": 500, "ungrounded": 0, "invalid": 0})
        self.assertEqual(summary["latency_seconds"]["p95"], 1)
        self.assertEqual(summary["judge_calls"], 0)
        self.assertEqual(summary["transport_errors"], 1)
        self.assertTrue(summary["baselines_identical"])
        self.assertTrue((self.output / "report.md").exists())
        self.runner.client.chat.reset_mock()
        with patch("medical_qa.benchmark.load_gold_for_final_scoring", side_effect=final_gold), \
             redirect_stdout(io.StringIO()):
            run_full_benchmark(self.cases, self.selection, self.output, self.smoke, runner=self.runner)
        self.runner.client.chat.assert_not_called()

    def test_partial_final_validation_is_refused(self):
        rows = {case.pmid: self.row(case) for case in self.cases[:20]}
        with self.assertRaises(DatasetValidationError):
            validate_completed_cases(self.cases, rows)

    def test_invalid_and_truncated_outputs_are_not_repaired_or_dropped(self):
        rows = {case.pmid: self.row(case) for case in self.cases}
        rows[self.ids[0]]["raw_response"] = "not JSON"
        rows[self.ids[1]]["raw_response"] = '{"label":"yes","evidence":"short"}'
        rows[self.ids[2]]["done_reason"] = "length"
        rows[self.ids[3]]["thinking_returned"] = True
        validated = validate_completed_cases(self.cases, rows)
        self.assertEqual(Counter(row["status"] for row in validated),
                         {"grounded": 496, "ungrounded": 1, "invalid": 3})
        self.assertEqual(validated[0]["scoring_label"], "invalid")
        self.assertEqual(validated[1]["scoring_label"], "yes")
        self.assertEqual(validated[2]["scoring_label"], "invalid")
        self.assertEqual(validated[3]["scoring_label"], "invalid")

    def test_partial_tail_is_recovered_without_losing_completed_rows(self):
        path = Path(self.directory.name) / "rows.jsonl"
        first = json.dumps(self.row(self.cases[0]))
        tail = b'{"pmid":'
        path.write_bytes(first.encode() + b"\n" + tail)
        rows, recovered = read_completed_rows(path, self.cases)
        self.assertEqual(list(rows), [self.ids[0]])
        self.assertEqual(recovered, len(tail))
        self.assertEqual(path.read_bytes(), first.encode() + b"\n")

    def test_valid_final_row_without_newline_is_preserved(self):
        path = Path(self.directory.name) / "rows.jsonl"
        path.write_text(json.dumps(self.row(self.cases[0])), encoding="utf-8")
        rows, recovered = read_completed_rows(path, self.cases)
        self.assertEqual(len(rows), 1)
        self.assertEqual(recovered, 0)
        self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_duplicate_foreign_or_changed_request_rows_are_refused(self):
        path = Path(self.directory.name) / "rows.jsonl"
        valid = self.row(self.cases[0])
        for rows in [[valid, valid], [{**valid, "pmid": "1"}], [{**valid, "request_sha256": "changed"}]]:
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                read_completed_rows(path, self.cases)

    def test_resume_refuses_changed_policy_or_code(self):
        self.runner.client.chat.side_effect = ConnectionError("toy interruption")
        with redirect_stdout(io.StringIO()), self.assertRaises(ConnectionError):
            run_full_benchmark(self.cases, self.selection, self.output, self.smoke, runner=self.runner)
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["code_sha256"]["benchmark.py"] = "changed"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            run_full_benchmark(self.cases, self.selection, self.output, self.smoke, runner=self.runner)
