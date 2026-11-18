import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from medical_qa.dataset import Case
from medical_qa.inference import InferenceError, MODEL, OPTIONS, OllamaRunner, build_messages
from medical_qa.validation import RESPONSE_SCHEMA


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.case = Case("123", "A toy question?", ("This is a sufficiently long verbatim quote.",))
        self.client = Mock()
        self.runner = OllamaRunner(client=self.client)
        self.client.list.return_value = SimpleNamespace(models=[SimpleNamespace(
            model=MODEL, digest=self.runner.lock["digest"],
        )])
        self.client.chat.return_value = SimpleNamespace(
            done=True, done_reason="stop", prompt_eval_count=100, eval_count=20,
            message=SimpleNamespace(content=json.dumps({
                "label": "maybe", "evidence": self.case.contexts[0],
            }), thinking=None),
        )
        version_patch = patch("medical_qa.inference.ollama_version", return_value="test-version")
        version_patch.start()
        self.addCleanup(version_patch.stop)

    def test_prompt_contains_only_allowed_example_fields(self):
        messages = build_messages(self.case)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(set(payload), {"question", "contexts"})
        self.assertNotIn("pmid", payload)
        for forbidden in ["final_decision", "long_answer", "reasoning_free_pred", "gold"]:
            self.assertNotIn(forbidden, messages[1]["content"])

    def test_chat_uses_exact_schema_and_frozen_options(self):
        prediction = self.runner.predict(self.case)
        self.assertEqual(prediction.status, "grounded")
        kwargs = self.client.chat.call_args.kwargs
        self.assertEqual(kwargs["model"], "qwen3:1.7b")
        self.assertEqual(kwargs["format"], RESPONSE_SCHEMA)
        self.assertIs(kwargs["think"], False)
        self.assertIs(kwargs["stream"], False)
        self.assertEqual(kwargs["options"], OPTIONS)
        self.assertEqual(kwargs["options"]["temperature"], 0)
        self.assertEqual(kwargs["options"]["seed"], 0)

    def test_changed_model_is_rejected_before_inference(self):
        self.client.list.return_value.models[0].digest = "changed"
        with self.assertRaises(InferenceError):
            self.runner.predict(self.case)
        self.client.chat.assert_not_called()

    def test_missing_target_is_rejected(self):
        self.client.list.return_value.models = []
        with self.assertRaises(InferenceError):
            self.runner.verify_model()

    def test_invalid_json_is_recorded_without_repair_or_retry(self):
        self.client.chat.return_value.message.content = "not JSON"
        prediction = self.runner.predict(self.case)
        self.assertEqual(prediction.status, "invalid_json")
        self.assertIsNone(prediction.evidence_check)
        self.assertEqual(self.client.chat.call_count, 1)

    def test_ungrounded_response_remains_valid_json(self):
        self.client.chat.return_value.message.content = '{"label":"yes","evidence":"a quote absent from this abstract"}'
        prediction = self.runner.predict(self.case)
        self.assertEqual(prediction.status, "ungrounded")
        self.assertEqual(prediction.evidence_check["reason"], "not_found")

    def test_truncation_is_a_technical_failure(self):
        self.client.chat.return_value.done_reason = "length"
        with self.assertRaises(InferenceError):
            self.runner.predict(self.case)

    def test_unexpected_thinking_is_rejected(self):
        self.client.chat.return_value.message.thinking = "unexpected reasoning"
        with self.assertRaises(InferenceError):
            self.runner.predict(self.case)
