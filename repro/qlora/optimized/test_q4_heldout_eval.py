"""CPU-only Q4 checks; no HF model loading or real held-out scoring."""

import ast
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import q4_heldout_eval as q4


class Q4Checks(unittest.TestCase):
    def test_exact_prompt_and_scorer_match_the_optimized_notebook(self):
        source = Path(__file__).resolve().parents[1] / "optimization/pubmedqa_qlora_optimized_colab.ipynb"
        notebook = json.loads(source.read_text("utf-8"))
        functions = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse(cell["source"]).body:
                if isinstance(node, ast.FunctionDef):
                    functions[node.name] = ast.dump(ast.parse(ast.get_source_segment(cell["source"], node)))
        for name in ("build_messages", "render_prompt", "choose_labels", "candidate_token_ids",
                     "left_pad_sequences", "predict_label_scores", "score_label_predictions"):
            self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(q4, name)))), functions[name])
        self.assertEqual(q4.sha256(q4.SYSTEM_PROMPT.encode()), q4.BASELINE["prompt_sha256"])

    def test_overlap_is_reported_for_each_split(self):
        with self.assertRaises(q4.Q4Blocked) as raised:
            q4.check_overlap({"1", "2"}, {"train": ["1"], "dev": ["2"], "artificial": ["3"]})
        self.assertEqual(raised.exception.report["counts"], {"train": 1, "dev": 1, "artificial": 0})

    def test_overlap_stops_before_source_or_model_loading(self):
        error = q4.Q4Blocked("overlap", {"counts": {"train": 1}})
        with patch.object(q4, "inspect_data", side_effect=error), patch.object(q4, "load_heldout_inputs") as inputs, patch.object(q4, "load_base") as model:
            with self.assertRaises(q4.Q4Blocked):
                q4.run_q4(Path("unused"), None, None, Path("unused.json"))
            inputs.assert_not_called()
            model.assert_not_called()

    def test_blocked_precheck_writes_report_without_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            error = q4.Q4Blocked("overlap", {"counts": {"train": 1}})
            with patch.object(sys, "argv", ["q4", "--project-root", directory, "--precheck-only"]), patch.object(q4, "inspect_data", side_effect=error), patch("builtins.print"):
                with self.assertRaises(SystemExit) as stopped:
                    q4.main()
                self.assertEqual(stopped.exception.code, 2)
            path = Path(directory) / "runs/qlora/optimized/q4_precheck.json"
            result = json.loads(path.read_text("utf-8"))
            self.assertEqual(result["status"], "blocked_overlap")
            self.assertIsNone(result["models"])
            self.assertFalse(result["gpu_execution"])

    def test_gold_is_not_read_until_all_three_models_finish(self):
        cases = [{"pmid": str(pmid)} for pmid in range(1, 501)]
        with patch("pyarrow.parquet.read_table") as read:
            with self.assertRaises(ValueError):
                q4.final_scoring_rows(cases, {"base": [], "A": []}, b"not a parquet")
            read.assert_not_called()

    def test_loader_uses_dtype_and_explicit_non_thinking_check(self):
        tree = ast.parse(inspect.getsource(q4.load_base))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        model = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained"
                     and isinstance(node.func.value, ast.Name) and node.func.value.id == "AutoModelForCausalLM")
        self.assertIn("dtype", [item.arg for item in model.keywords])
        self.assertNotIn("torch_dtype", [item.arg for item in model.keywords])
        template = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "apply_chat_template")
        flag = next(item.value for item in template.keywords if item.arg == "enable_thinking")
        self.assertIs(ast.literal_eval(flag), False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
