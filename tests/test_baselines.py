import unittest

from medical_qa.baselines import (
    PUBLISHED_MAJORITY_LABEL, PUBLISHED_PQA_L_PRIORS_PERCENT,
    baseline_definitions, baseline_predictions,
)
from medical_qa.dataset import DatasetValidationError


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.ids = tuple(str(i) for i in range(1, 501))

    def test_majority_comes_from_jin_table_one_priors(self):
        self.assertEqual(dict(PUBLISHED_PQA_L_PRIORS_PERCENT), {"yes": 55.2, "no": 33.8, "maybe": 11.0})
        self.assertEqual(PUBLISHED_MAJORITY_LABEL, "yes")
        definition = baseline_definitions()["published_majority"]
        self.assertIn("Jin et al. (2019)", definition["priors_citation"])
        self.assertIn("Table 1", definition["priors_citation"])

    def test_both_baselines_are_identical_yes_predictions_for_all_500_ids(self):
        predictions = baseline_predictions(self.ids)
        self.assertEqual(set(predictions), {"always_yes", "published_majority"})
        self.assertEqual(predictions["always_yes"], predictions["published_majority"])
        self.assertEqual(set(predictions["always_yes"]), set(self.ids))
        self.assertEqual(set(predictions["always_yes"].values()), {"yes"})

    def test_prediction_generation_is_independent_of_id_order(self):
        self.assertEqual(baseline_predictions(self.ids), baseline_predictions(self.ids[::-1]))

    def test_published_priors_cannot_be_mutated(self):
        with self.assertRaises(TypeError):
            PUBLISHED_PQA_L_PRIORS_PERCENT["no"] = 100

    def test_partial_or_duplicate_id_sets_are_rejected(self):
        for ids in [self.ids[:20], self.ids[:-1] + ("1",)]:
            with self.subTest(size=len(ids)), self.assertRaises(DatasetValidationError):
                baseline_predictions(ids)
