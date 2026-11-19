# Evaluation Results API

A local, read-only FastAPI service for the frozen [Ollama Phase 1](../results/phase1-full/results-summary.json) and [Hugging Face Q4](../repro/qlora/optimized/q4_heldout_eval.json) summaries. On startup it imports aggregate results into `runs/eval-api/results.sqlite3`. Restarting refreshes the known rows from the saved artifacts. Each run includes its protocol and source path. Requests use a read-only database connection.

Run from the repository root with Python 3.12.

```powershell
python -m venv runs/eval-api/venv
.\runs\eval-api\venv\Scripts\python.exe -m pip install -r eval_api/requirements.txt
.\runs\eval-api\venv\Scripts\python.exe -m uvicorn eval_api.app:app --host 127.0.0.1 --port 8000
```

| Path | Response |
| --- | --- |
| `GET /health` | Database availability |
| `GET /runs` | Run IDs, protocols, sample counts, and source paths |
| `GET /runs/phase1-full` | Ollama and rule-based baseline scores, confusion matrices, exact-quote match rate, and invalid rate |
| `GET /runs/q4-heldout` | HF Base/A/B scores and confusion matrices under `label_score_v1` |
| `GET /docs` | Interactive API documentation |

The Ollama run generates JSON labels and evidence. Q4 scores label probabilities under a shared protocol. The exact-quote rate measures textual matching and applies to Phase 1 only. The API reads aggregate files directly and requires only its own dependencies and Python's built-in SQLite.

Run the API tests with the same virtual environment. The workflow runs the same command on GitHub Actions.

```powershell
.\runs\eval-api\venv\Scripts\python.exe -m unittest discover -s eval_api -v
```
