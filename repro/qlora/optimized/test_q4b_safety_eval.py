"""CPU tests with synthetic fixtures; no HF calls or actual generation."""

from contextlib import nullcontext
import ast
import inspect
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import q4b_safety_eval as q4b

FOLDER = Path(__file__).resolve().parent
VALID = {"do_sample": False, "num_beams": 1, "max_new_tokens": 512, "seed": 42}


def cell_source(cell):
    # Both representations are valid notebook JSON; VS Code saves line lists.
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


class FakeTokenizer:
    chat_template = "TEST TEMPLATE"
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return str(messages) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3]}

    def decode(self, ids, **kwargs):
        return "TEST FIXTURE RESPONSE"


class SafetyChecks(unittest.TestCase):
    def copy_freeze(self, directory):
        for name in ("q4b_safety_rubric.md", "q4b_manifest.json"):
            shutil.copyfile(FOLDER / name, Path(directory) / name)

    def test_all_twenty_inputs_have_no_criteria_or_gold(self):
        cases = q4b.load_cases(FOLDER)
        self.assertEqual(tuple(case["id"] for case in cases), q4b.CASE_IDS)
        self.assertTrue(all(set(case) == {"id", "question"} for case in cases))

    def test_changed_rubric_is_rejected_even_if_manifest_is_rehashed(self):
        with tempfile.TemporaryDirectory() as directory:
            self.copy_freeze(directory)
            path = Path(directory) / "q4b_safety_rubric.md"
            path.write_bytes(path.read_bytes() + b"\nchanged")
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                q4b.load_cases(directory)
            manifest_path = Path(directory) / "q4b_manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["sha256"] = q4b.sha256(path.read_bytes())
            q4b.write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "registration differs"):
                q4b.load_cases(directory)

    def test_unfinished_exercise_blocks_before_adapters_or_tokenizer(self):
        settings = dict(VALID, do_sample=Ellipsis)
        with patch.object(q4b, "inspect_adapters") as adapters, patch.object(q4b, "load_tokenizer") as tokenizer:
            with self.assertRaisesRegex(ValueError, "Exercise unfinished"):
                q4b.run(FOLDER, None, None, settings)
            adapters.assert_not_called()
            tokenizer.assert_not_called()

    def test_sampling_and_extra_or_mistyped_options_are_rejected(self):
        for settings in (dict(VALID, do_sample=True), dict(VALID, do_sample=0),
                         dict(VALID, num_beams=True), dict(VALID, max_new_tokens=1024),
                         dict(VALID, seed=0), dict(VALID, temperature=0)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                q4b.validate_generation_settings(settings)
        self.assertEqual(q4b.validate_generation_settings(VALID), VALID)

    def test_question_only_messages_ignore_extra_fields(self):
        case = {"question": "TEST QUESTION", "expected_action": "SECRET", "gold": "SECRET"}
        messages = q4b.build_messages(case)
        self.assertEqual(messages[1], {"role": "user", "content": "TEST QUESTION"})
        self.assertNotIn("SECRET", json.dumps(messages))
        q4b.render_prompt(FakeTokenizer(), case)

    def test_wrong_thinking_template_is_rejected(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "<think>"
        with self.assertRaisesRegex(ValueError, "non-thinking"):
            q4b.render_prompt(tokenizer, {"question": "TEST"})

    def test_adapter_pins_match_completed_q4(self):
        previous = json.loads((FOLDER / "q4_heldout_eval.json").read_bytes())
        self.assertEqual(q4b.HF_REVISION, previous["protocol"]["hf_revision"])
        for variant in ("A", "B"):
            self.assertEqual(q4b.ADAPTER_HASHES[variant], previous["adapters"][variant]["sha256"])

    def test_changed_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "best_adapter"
            adapter.mkdir()
            for name in q4b.ADAPTER_HASHES["A"]:
                (adapter / name).write_bytes(b"TEST WRONG ADAPTER")
            with self.assertRaisesRegex(ValueError, "differs from the completed Q4"):
                q4b.inspect_adapters(directory, directory)

    def test_cpu_precheck_has_no_weight_loading_or_generation(self):
        with patch.object(q4b, "inspect_adapters", return_value={}), \
                patch.object(q4b, "version", side_effect=lambda package: q4b.PINS[package]), \
                patch.object(q4b, "load_tokenizer", return_value=FakeTokenizer()), \
                patch.object(q4b, "load_model") as weights, patch.object(q4b, "generate_one") as generate:
            report = q4b.run(FOLDER, None, None, VALID, precheck_only=True)
        self.assertEqual(report["status"], "precheck_passed")
        self.assertFalse(report["generation_started"])
        weights.assert_not_called()
        generate.assert_not_called()

    def test_full_flow_registers_first_retries_only_errors_then_skips_all(self):
        """Exercise orchestration using mock tensors/models, not real model outputs."""
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "TEST GPU",
                                 empty_cache=Mock()),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()),
                                     cudnn=SimpleNamespace()),
            __version__="TEST TORCH", version=SimpleNamespace(cuda="TEST CUDA"),
        )
        class Config(SimpleNamespace):
            def to_dict(self):
                return vars(self)
        transformers = SimpleNamespace(GenerationConfig=Config, set_seed=Mock())
        with tempfile.TemporaryDirectory() as directory:
            self.copy_freeze(directory)
            first_attempt = True
            def fixture_generation(*args):
                nonlocal first_attempt
                manifest = json.loads((Path(directory) / "q4b_manifest.json").read_bytes())
                self.assertIn("generation", manifest)  # Registration precedes every attempt.
                if first_attempt:
                    first_attempt = False
                    raise RuntimeError("TEST TRANSIENT ERROR")
                return {"response": "TEST FIXTURE RESPONSE", "raw_response": "TEST FIXTURE RESPONSE",
                        "truncated": False, "finish_reason": "eos"}
            with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
                    patch.object(q4b, "inspect_adapters", return_value={"A": {"path": "TEST A"}, "B": {"path": "TEST B"}}), \
                    patch.object(q4b, "version", side_effect=lambda package: q4b.PINS[package]), \
                    patch.object(q4b, "load_tokenizer", return_value=FakeTokenizer()), \
                    patch.object(q4b, "load_model", return_value=Mock()) as weights, \
                    patch.object(q4b, "generate_one", side_effect=fixture_generation) as generate, \
                    patch("builtins.print"):
                first = q4b.run(directory, None, None, VALID)
                self.assertEqual(first["technical_errors"], 1)
                self.assertEqual(first["successful_responses"], 59)
                self.assertFalse((Path(directory) / "q4b_blind_review.md").exists())
                self.assertEqual(generate.call_count, 60)
                weights.reset_mock()
                second = q4b.run(directory, None, None, VALID)
                self.assertEqual(second["successful_responses"], 60)
                self.assertEqual(generate.call_count, 61)
                self.assertEqual(weights.call_count, 1)  # Only the failed base case is retried.
                weights.reset_mock()
                third = q4b.run(directory, None, None, VALID)
                self.assertEqual(third["successful_responses"], 60)
                self.assertEqual(generate.call_count, 61)
                weights.assert_not_called()

    def test_manifest_extension_is_idempotent_and_rejects_protocol_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            self.copy_freeze(directory)
            before = json.loads((Path(directory) / "q4b_manifest.json").read_bytes())
            contract = {"name": "TEST CONTRACT", "settings": VALID}
            digest = q4b.register_generation(directory, contract)
            frozen = (Path(directory) / "q4b_manifest.json").read_bytes()
            self.assertEqual(q4b.register_generation(directory, contract), digest)
            self.assertEqual((Path(directory) / "q4b_manifest.json").read_bytes(), frozen)
            after = json.loads(frozen)
            for key, value in before.items():
                self.assertEqual(after[key], value)
            with self.assertRaisesRegex(ValueError, "changed"):
                q4b.register_generation(directory, {"name": "CHANGED"})

    def row(self, case_id="D01", status="ok"):
        return {"case_id": case_id, "model": "base", "contract_sha256": "TEST CONTRACT",
                "status": status, "response": "TEST FIXTURE RESPONSE"}

    def test_journal_skips_success_and_accepts_error_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.jsonl"
            q4b.append_row(path, self.row(status="technical_error"))
            q4b.append_row(path, self.row())
            saved = q4b.read_journal(path, "base", "TEST CONTRACT")
            self.assertEqual(saved["D01"]["status"], "ok")
            with self.assertRaisesRegex(ValueError, "another"):
                q4b.read_journal(path, "base", "WRONG CONTRACT")
            q4b.append_row(path, self.row())
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                q4b.read_journal(path, "base", "TEST CONTRACT")

    def test_interrupted_tail_is_archived_without_losing_finished_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.jsonl"
            q4b.append_row(path, self.row())
            with path.open("ab") as stream:
                stream.write(b'{"incomplete":')
            self.assertEqual(len(q4b.read_journal(path, "base", "TEST CONTRACT")), 1)
            self.assertEqual(len(list(Path(directory).glob("*.partial-*"))), 1)
            q4b.append_row(path, self.row("D02"))
            self.assertEqual(len(q4b.read_journal(path, "base", "TEST CONTRACT")), 2)

    def test_malformed_complete_line_is_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.jsonl"
            path.write_bytes(b"broken\n")
            with self.assertRaisesRegex(ValueError, "Malformed"):
                q4b.read_journal(path, "base", "TEST CONTRACT")
            self.assertEqual(path.read_bytes(), b"broken\n")

    def test_eos_at_budget_is_not_reported_as_truncation(self):
        tokenizer = FakeTokenizer()
        self.assertFalse(q4b.response_fields(tokenizer, [1, 2, 99], [99], 3)["truncated"])
        self.assertTrue(q4b.response_fields(tokenizer, [1, 2, 3], [99], 3)["truncated"])

    def test_generation_decodes_only_new_tokens_with_gradients_disabled(self):
        tokenizer = FakeTokenizer()
        inputs = {"input_ids": SimpleNamespace(shape=(1, 3))}
        class Inputs(dict):
            def to(self, device):
                return self
        tokenizer = Mock(wraps=tokenizer)
        tokenizer.return_value = Inputs(inputs)
        class Output:
            def __getitem__(self, key):
                self.key = key
                return SimpleNamespace(tolist=lambda: [7, 99])
        output = Output()
        model = Mock()
        model.generate.return_value = output
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock()),
                                inference_mode=lambda: nullcontext(), autocast=lambda *a, **k: nullcontext(),
                                float16="TEST FP16")
        config = SimpleNamespace(eos_token_id=[99], max_new_tokens=512)
        with patch.dict(sys.modules, {"torch": torch, "transformers": SimpleNamespace(set_seed=Mock())}):
            result = q4b.generate_one(model, tokenizer, "TEST PROMPT", config, 42)
        self.assertEqual(output.key, (0, slice(3, None)))
        self.assertEqual(result["output_token_ids"], [7, 99])
        self.assertEqual(result["finish_reason"], "eos")

    def test_blind_export_requires_sixty_and_preserves_human_edits(self):
        cases = q4b.load_cases(FOLDER)
        responses = {(model, case_id): {"status": "ok", "response": "TEST FIXTURE RESPONSE",
                     "truncated": False, "finish_reason": "eos"}
                     for model in q4b.VARIANTS for case_id in q4b.CASE_IDS}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "60 successful"):
                q4b.export_blind_review(directory, cases, {}, "TEST CONTRACT")
            path = q4b.export_blind_review(directory, cases, responses, "TEST CONTRACT")
            text = path.read_text("utf-8")
            self.assertEqual(text.count("- **my_pass:**"), 60)
            self.assertNotIn('"model"', text)
            text += "\nUSER ANNOTATION\n"
            path.write_text(text, encoding="utf-8")
            q4b.export_blind_review(directory, cases, responses, "TEST CONTRACT")
            self.assertEqual(path.read_text("utf-8"), text)

    def test_model_loading_uses_dtype_not_deprecated_keyword(self):
        source = inspect.getsource(q4b.load_model)
        self.assertIn("dtype=torch.float16", source)
        self.assertNotIn("torch_dtype=", source)

    def test_notebook_allows_completed_exercise_and_keeps_generation_opt_in(self):
        notebook = json.loads((FOLDER / "q4b_safety_eval.ipynb").read_bytes())
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = cell_source(cell)
                compile(source, "notebook-cell", "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        sources = "\n".join(cell_source(cell) for cell in notebook["cells"])
        self.assertIn("START_GENERATION = False", sources)
        exercise = next(cell for cell in notebook["cells"] if cell["id"] == "q4b-06")
        tree = ast.parse(cell_source(exercise))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "build_generation_settings")
        scope = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "exercise-function", "exec"), scope)
        settings = scope["build_generation_settings"]()
        if settings["do_sample"] is Ellipsis:
            with self.assertRaisesRegex(ValueError, "Exercise unfinished"):
                q4b.validate_generation_settings(settings)
        else:
            q4b.validate_generation_settings(settings)

    def test_learner_cell_itself_stops_before_saving_settings(self):
        notebook = json.loads((FOLDER / "q4b_safety_eval.ipynb").read_bytes())
        cell = next(cell for cell in notebook["cells"] if cell["id"] == "q4b-06")
        # Test an unfinished copy in memory; keep the learner's saved answer intact.
        tree = ast.parse(cell_source(cell))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "do_sample" for target in node.targets
            ):
                node.value = ast.Constant(value=Ellipsis)
        ast.fix_missing_locations(tree)
        with tempfile.TemporaryDirectory() as directory:
            settings_file = Path(directory) / "q4b_settings.json"
            with self.assertRaisesRegex(ValueError, "Exercise unfinished"):
                exec(compile(tree, "unfinished-exercise-copy", "exec"),
                     {"q4b": q4b, "SETTINGS_FILE": settings_file})
            self.assertFalse(settings_file.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
