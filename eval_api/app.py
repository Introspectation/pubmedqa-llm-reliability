"""Read-only HTTP access to the published PubMedQA aggregate results."""

from contextlib import asynccontextmanager, closing
import json
from pathlib import Path
import sqlite3

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "runs" / "eval-api" / "results.sqlite3"
PHASE1_SOURCE = "results/phase1-full/results-summary.json"
Q4_SOURCE = "repro/qlora/optimized/q4_heldout_eval.json"


def model_scores(models):
    fields = ("accuracy", "macro_f1", "confusion_matrix")
    return {name: {field: score[field] for field in fields} for name, score in models.items()}


def import_reports(database):
    phase1 = json.loads((ROOT / PHASE1_SOURCE).read_bytes())
    q4 = json.loads((ROOT / Q4_SOURCE).read_bytes())
    rows = [
        (
            "phase1-full", "ollama_generated_json", phase1["n"], PHASE1_SOURCE,
            json.dumps({
                "models": model_scores(phase1["results"]),
                "exact_quote_match_rate": phase1["grounded_rate"],
                "invalid_rate": phase1["invalid_rate"],
            }),
        ),
        (
            "q4-heldout", q4["protocol"]["name"], q4["n"], Q4_SOURCE,
            json.dumps({"models": model_scores(q4["models"])}),
        ),
    ]
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                protocol TEXT NOT NULL,
                n INTEGER NOT NULL,
                source TEXT NOT NULL,
                metrics TEXT NOT NULL
            )
        """)
        db.executemany("""
            INSERT INTO runs (id, protocol, n, source, metrics)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                protocol = excluded.protocol, n = excluded.n,
                source = excluded.source, metrics = excluded.metrics
        """, rows)


def create_app(database: Path = DEFAULT_DB) -> FastAPI:
    database = database.resolve()

    @asynccontextmanager
    async def lifespan(_app):
        import_reports(database)
        yield

    app = FastAPI(title="PubMedQA Evaluation Results", lifespan=lifespan)
    database_uri = database.as_uri() + "?mode=ro"

    def read_database():
        db = sqlite3.connect(database_uri, uri=True)
        db.row_factory = sqlite3.Row
        return closing(db)

    @app.get("/health")
    def health():
        with read_database() as db:
            db.execute("SELECT 1 FROM runs LIMIT 1").fetchone()
        return {"status": "ok"}

    @app.get("/runs")
    def runs():
        with read_database() as db:
            return [dict(row) for row in db.execute("SELECT id, protocol, n, source FROM runs ORDER BY id")]

    @app.get("/runs/{run_id}")
    def run(run_id: str):
        with read_database() as db:
            row = db.execute(
                "SELECT id, protocol, n, source, metrics FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Evaluation run not found.")
        result = dict(row)
        result.update(json.loads(result.pop("metrics")))
        return result

    return app


app = create_app()
