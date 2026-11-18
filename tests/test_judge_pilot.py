import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from medical_qa.dataset import ROOT, Case, SelectionReport, load_test_pmids, pmid_fingerprint, source_lock
from scripts.prepare_judge_pilot import prepare_pilot, select_pilot_ids, write_pilot
from scripts.run_judge import judge_request, load_pilot, parse_judgment, run_judge, text_hash


class JudgePilotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "pilot"
        self.rubric = ROOT / "prompts/judge_rubric_v1.txt"
        self.ids = load_test_pmids()
        # All text, predictions, gold values and responses below are toy fixtures.
        self.eligible = {pmid: {
            "pmid": pmid, "question": "Toy question?", "abstract": "A sufficiently long toy quotation for testing.",
            "quote": "A sufficiently long toy quotation for testing.", "answer": "yes" if i < 20 else "no",
        } for i, pmid in enumerate(self.ids[:40])}
        self.gold = {pmid: "yes" for pmid in self.eligible}
        self.selected = select_pilot_ids(self.eligible, self.gold)
        self.manifest = write_pilot(self.output, self.selected, self.eligible, self.rubric, {"fixture": True})
        self.client = Mock()
        self.client.list.return_value = SimpleNamespace(models=[SimpleNamespace(model="qwen3:8b", digest="toy-digest")])
        self.response = SimpleNamespace(
            done=True, done_reason="stop", prompt_eval_count=123, eval_count=12,
            message=SimpleNamespace(thinking=None, content='{"label":"supports","reason":"The quote directly addresses the question."}'),
        )
        self.client.chat.return_value = self.response
        version_patch = patch("scripts.run_judge.server_version", return_value="toy-server")
        version_patch.start()
        self.addCleanup(version_patch.stop)

    def test_seed_42_selects_ten_correct_and_ten_wrong_without_order_dependence(self):
        self.assertEqual(len(self.selected), 20)
        self.assertEqual(len(set(self.selected)), 20)
        self.assertEqual(sum(self.eligible[p]["answer"] == self.gold[p] for p in self.selected), 10)
        reverse = dict(reversed(list(self.eligible.items())))
        self.assertEqual(self.selected, select_pilot_ids(reverse, self.gold, seed=42))
        self.assertEqual(set(self.manifest["stratum_sample_sizes"]), {"correct", "wrong"})

    def test_insufficient_strata_are_refused(self):
        with self.assertRaises(ValueError):
            select_pilot_ids(dict(list(self.eligible.items())[:25]), self.gold)

    def test_human_csv_has_exact_blind_columns_and_empty_labels(self):
        with (self.output / "human_labels.csv").open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, ["pmid", "question", "quote", "answer", "my_label"])
        self.assertEqual([row["pmid"] for row in rows], self.selected)
        self.assertTrue(all(row["my_label"] == "" for row in rows))
        self.assertEqual(json.loads((self.output / "pilot_ids.json").read_text()), self.selected)

    def test_existing_human_work_is_never_overwritten(self):
        csv_path = self.output / "human_labels.csv"
        csv_path.write_text("HUMAN_WORK", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            write_pilot(self.output, self.selected, self.eligible, self.rubric, {})
        self.assertEqual(csv_path.read_text(), "HUMAN_WORK")

    def test_input_whitelist_rejects_gold_even_if_hash_is_updated(self):
        path = self.output / "pilot_inputs.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["gold_label"] = "SECRET_GOLD_SENTINEL"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest = json.loads((self.output / "pilot_manifest.json").read_text())
        manifest["pilot_inputs_sha256"] = text_hash(path)
        (self.output / "pilot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_pilot(self.output / "pilot_ids.json", self.rubric)

    def test_changed_rubric_is_refused_before_any_judge_request(self):
        changed = self.root / "changed-rubric.txt"
        changed.write_text(self.rubric.read_text() + " changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            run_judge("qwen3:8b", self.output / "pilot_ids.json", changed, self.output, client=self.client)
        self.client.list.assert_not_called()
        self.client.chat.assert_not_called()

    def test_requests_are_gold_blind_json_temperature_zero_and_qwen_no_think(self):
        case = self.eligible[self.selected[0]]
        rubric = self.rubric.read_text(encoding="utf-8")
        for model in ["qwen3:8b", "qwen3:1.7b", "library/qwen3:4b"]:
            request = judge_request(model, rubric, case)
            self.assertEqual(request["format"], "json")
            self.assertEqual(request["options"], {"temperature": 0})
            self.assertIs(request["think"], False)
            content = request["messages"][0]["content"]
            self.assertTrue(content.endswith("/no_think"))
            self.assertNotIn(case["pmid"], content)
            self.assertIn("QUESTION: " + case["question"], content)
            self.assertIn("ABSTRACT: " + case["abstract"], content)
            self.assertIn("QUOTE: " + case["quote"], content)
            self.assertIn("ANSWER: " + case["answer"], content)
            self.assertIn('{"label": "...", "reason": "<= 20 words"}', content)
        request = judge_request("llama3.2:3b", rubric, case)
        self.assertNotIn("think", request)
        self.assertFalse(request["messages"][0]["content"].endswith("/no_think"))

    def test_template_braces_inside_inputs_are_not_interpreted_recursively(self):
        case = dict(self.eligible[self.selected[0]], question="Does literal {quote} change?")
        request = judge_request("qwen3:8b", self.rubric.read_text(), case)
        self.assertIn("QUESTION: Does literal {quote} change?", request["messages"][0]["content"])

    def test_judge_json_labels_and_twenty_word_limit(self):
        for label in ["supports", "partial", "does_not_support"]:
            raw = json.dumps({"label": label, "reason": " ".join(["word"] * 20)})
            self.assertEqual(parse_judgment(raw)["label"], label)
        bad = [
            json.dumps({"label": "supports", "reason": " ".join(["word"] * 21)}),
            '{"label":"yes","reason":"wrong label"}',
            '{"label":"supports","reason":123}',
            '{"label":"supports","reason":"fine","gold":"extra"}',
            '{"label":"supports","label":"partial","reason":"duplicate"}',
            '```json\n{"label":"supports","reason":"wrapped"}\n```',
            '[]', 'null',
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_judgment(raw)

    def test_resume_skips_completed_pmids_and_ignores_human_csv(self):
        csv_path = self.output / "human_labels.csv"
        csv_path.write_text("HUMAN_WORK_MUST_NOT_BE_READ", encoding="utf-8")
        self.client.chat.side_effect = [self.response, ConnectionError("toy interruption")]
        with redirect_stdout(io.StringIO()), self.assertRaises(ConnectionError):
            run_judge("qwen3:8b", self.output / "pilot_ids.json", self.rubric, self.output, client=self.client)
        journal = self.output / "qwen3_8b.jsonl"
        first_row = journal.read_bytes()
        with journal.open("ab") as handle:
            handle.write(b'{"pmid":')  # Simulate a torn final append.
        self.client.chat.side_effect = None
        self.client.chat.reset_mock()
        with redirect_stdout(io.StringIO()):
            run_judge("qwen3:8b", self.output / "pilot_ids.json", self.rubric, self.output, client=self.client)
        self.assertEqual(self.client.chat.call_count, 19)
        self.assertTrue(journal.read_bytes().startswith(first_row))
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        self.assertEqual([row["pmid"] for row in rows], self.selected)
        self.assertTrue(all(row["valid"] for row in rows))
        self.assertEqual(csv_path.read_text(), "HUMAN_WORK_MUST_NOT_BE_READ")
        self.client.chat.reset_mock()
        with redirect_stdout(io.StringIO()):
            run_judge("qwen3:8b", self.output / "pilot_ids.json", self.rubric, self.output, client=self.client)
        self.client.chat.assert_not_called()

    def test_invalid_judge_output_is_preserved_without_retries(self):
        self.client.chat.return_value.message.content = "invalid JSON"
        with redirect_stdout(io.StringIO()):
            journal = run_judge("qwen3:8b", self.output / "pilot_ids.json", self.rubric, self.output, client=self.client)
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        self.assertEqual(self.client.chat.call_count, 20)
        self.assertTrue(all(not row["valid"] and row["judgment"] is None for row in rows))

    def test_changed_digest_is_refused_on_resume(self):
        self.client.chat.side_effect = ConnectionError("toy interruption")
        with redirect_stdout(io.StringIO()), self.assertRaises(ConnectionError):
            run_judge("qwen3:8b", self.output / "pilot_ids.json", self.rubric, self.output, client=self.client)
        self.client.list.return_value.models[0].digest = "different-digest"
        self.client.chat.reset_mock()
        with self.assertRaises(ValueError):
            run_judge("qwen3:8b", self.output / "pilot_ids.json", self.rubric, self.output, client=self.client)
        self.client.chat.assert_not_called()

    def test_full_preparation_uses_only_grounded_strata_and_preserves_source(self):
        phase1 = self.root / "phase1"
        phase1.mkdir()
        predictions = phase1 / "validated-predictions.jsonl"
        quote = "A sufficiently long toy quotation for testing."
        rows = [{"pmid": pmid, "status": "grounded" if i < 483 else "ungrounded",
                 "schema_valid": True, "protocol_valid": True,
                 "raw_response": json.dumps({"label": "yes", "evidence": quote})}
                for i, pmid in enumerate(self.ids)]
        predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        summary = {"n": 500, "counts": {"grounded": 483},
                   "validated_predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                   "frozen_config": {"sources": source_lock()}}
        summary_path = phase1 / "results-summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        before = (summary_path.read_bytes(), predictions.read_bytes())
        cases = tuple(Case(pmid, "Toy question?", (quote,)) for pmid in self.ids)
        selection = SelectionReport(1000, 1000, 500, 500, 500, pmid_fingerprint(self.ids))
        toy_gold = {pmid: "yes" if i < 250 else "no" for i, pmid in enumerate(self.ids)}
        output = self.root / "prepared"
        with patch("scripts.prepare_judge_pilot.load_dataset", return_value=(cases, selection)), \
             patch("scripts.prepare_judge_pilot.load_gold_for_final_scoring", return_value=(toy_gold, selection)):
            manifest = prepare_pilot(phase1, predictions, None, output, self.rubric)
        selected = json.loads((output / "pilot_ids.json").read_text())
        self.assertEqual(sum(toy_gold[p] == "yes" for p in selected), 10)
        self.assertTrue(set(selected).issubset(self.ids[:483]))
        self.assertEqual(manifest["eligible_quote_matched_cases"], 483)
        self.assertEqual((summary_path.read_bytes(), predictions.read_bytes()), before)
        self.assertEqual(manifest["judge_calls_during_preparation"], 0)
        with patch("scripts.prepare_judge_pilot.load_gold_for_final_scoring") as gold:
            with self.assertRaises(FileExistsError):
                prepare_pilot(phase1, predictions, None, output, self.rubric)
            gold.assert_not_called()
