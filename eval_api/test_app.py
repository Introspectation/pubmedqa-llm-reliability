import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from eval_api.app import create_app, PHASE1_SOURCE, Q4_SOURCE, ROOT


class ResultsApiTests(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(TemporaryDirectory())
        self.database = Path(folder) / "results.sqlite3"
        self.client = self.enterContext(TestClient(create_app(self.database)))

    def test_health_and_run_index(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        response = self.client.get("/runs")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [
            {"id": "phase1-full", "protocol": "ollama_generated_json", "n": 500, "source": PHASE1_SOURCE},
            {"id": "q4-heldout", "protocol": "label_score_v1", "n": 500, "source": Q4_SOURCE},
        ])

    def test_saved_metrics_match_their_source(self):
        phase1 = self.client.get("/runs/phase1-full").json()
        q4 = self.client.get("/runs/q4-heldout").json()
        phase1_source = json.loads((ROOT / PHASE1_SOURCE).read_bytes())
        q4_source = json.loads((ROOT / Q4_SOURCE).read_bytes())
        for response, source in ((phase1, phase1_source["results"]), (q4, q4_source["models"])):
            self.assertEqual(set(response["models"]), set(source))
            for name, score in response["models"].items():
                self.assertEqual(set(score), {"accuracy", "macro_f1", "confusion_matrix"})
                for field, value in score.items():
                    self.assertEqual(value, source[name][field])
        self.assertEqual(phase1["exact_quote_match_rate"], phase1_source["grounded_rate"])
        self.assertEqual(phase1["invalid_rate"], phase1_source["invalid_rate"])
        self.assertNotIn("exact_quote_match_rate", q4)

    def test_missing_run_and_write_requests(self):
        self.assertEqual(self.client.get("/runs/unknown").status_code, 404)
        self.assertEqual(self.client.post("/runs").status_code, 405)

    def test_restarting_with_the_same_database_preserves_the_run_index(self):
        with TestClient(create_app(self.database)) as restarted:
            self.assertEqual(restarted.get("/runs").json(), self.client.get("/runs").json())
