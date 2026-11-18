import unittest

from medical_qa.validation import AnswerValidationError, normalize, parse_answer, validate_evidence


class JsonTests(unittest.TestCase):
    def test_all_three_labels(self):
        for label in ["yes", "no", "maybe"]:
            answer = parse_answer('{"label":"' + label + '","evidence":"An exact quote."}')
            self.assertEqual(answer.label, label)

    def test_invalid_json_and_schema(self):
        invalid = [
            "", '```json\n{"label":"yes","evidence":"quote"}\n```',
            '{"label":"yes","evidence":"quote"} trailing',
            '{"label":"YES","evidence":"quote"}',
            '{"label":"unknown","evidence":"quote"}',
            '{"label":true,"evidence":"quote"}',
            '{"label":"yes","evidence":123}',
            '{"label":"yes","evidence":null}',
            '{"label":"yes"}', '{"evidence":"quote"}',
            '{"label":"yes","evidence":"quote","explanation":"extra"}',
            '[]', 'null', '"yes"',
            '{"label":"yes","label":"no","evidence":"quote"}',
            '{"label":"yes","evidence":"first","evidence":"second"}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(AnswerValidationError):
                parse_answer(raw)

    def test_json_validity_is_separate_from_evidence_grounding(self):
        answer = parse_answer('{"label":"maybe","evidence":""}')
        self.assertFalse(validate_evidence(answer.evidence, ["Some abstract text."]).grounded)


class EvidenceTests(unittest.TestCase):
    def test_normalization_is_exact_specification(self):
        self.assertEqual(normalize("  MIXED\tCase\nText\u00a0  Here "), "mixed case text here")

    def test_lowercase_and_whitespace_match(self):
        result = validate_evidence("  STUDY\n Results\tWere Promising. ", ["The study results were promising. End."])
        self.assertTrue(result.grounded)
        self.assertEqual(result.context_index, 0)

    def test_exact_twenty_character_boundary(self):
        self.assertTrue(validate_evidence("x" * 20, ["x" * 20]).grounded)
        result = validate_evidence("x" * 19, ["x" * 20])
        self.assertFalse(result.grounded)
        self.assertEqual(result.reason, "too_short")

    def test_length_is_measured_after_normalization(self):
        result = validate_evidence("   Short \n quote   ", ["short quote"])
        self.assertFalse(result.grounded)
        self.assertEqual(result.normalized_characters, 11)

    def test_nonexistent_or_paraphrased_quote_is_not_found(self):
        result = validate_evidence("Treatment helped every participant.", ["Some participants improved."])
        self.assertFalse(result.grounded)
        self.assertEqual(result.reason, "not_found")

    def test_quote_cannot_cross_context_sections(self):
        quote = "a sufficiently long first section and a second section"
        self.assertFalse(validate_evidence(quote, ["a sufficiently long first section", "and a second section"]).grounded)

    def test_match_in_any_section(self):
        result = validate_evidence("a quote of sufficient length", ["unrelated", "Here is a quote of sufficient length."])
        self.assertTrue(result.grounded)
        self.assertEqual(result.context_index, 1)

    def test_punctuation_is_not_stripped(self):
        self.assertFalse(validate_evidence("a quote of sufficient length!", ["a quote of sufficient length."]).grounded)

    def test_no_contexts_cannot_match(self):
        self.assertFalse(validate_evidence("a quote of sufficient length", []).grounded)
