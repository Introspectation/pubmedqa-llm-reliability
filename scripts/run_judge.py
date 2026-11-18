"""Run one local Ollama judge on the sealed, gold-blind 20-case pilot."""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filelock import FileLock
from medical_qa.dataset import load_test_pmids

HOST = "http://127.0.0.1:11434"
OPTIONS = {"temperature": 0}
JUDGE_LABELS = {"supports", "partial", "does_not_support"}
INPUT_FIELDS = {"pmid", "question", "abstract", "quote", "answer"}


def text_hash(path):
    """Hash UTF-8 text with canonical LF newlines, including across Git checkouts."""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_pilot(pilot_ids, rubric_path):
    """Only read sealed, whitelisted inputs. Never read gold or the human CSV."""
    directory = pilot_ids.parent
    manifest = json.loads((directory / "pilot_manifest.json").read_text("utf-8"))
    inputs_path = directory / "pilot_inputs.jsonl"
    for key, path in [("pilot_ids_sha256", pilot_ids), ("pilot_inputs_sha256", inputs_path),
                      ("rubric_sha256", rubric_path)]:
        if text_hash(path) != manifest[key]:
            raise ValueError("Sealed pilot input or rubric changed: " + key)
    ids = json.loads(pilot_ids.read_text("utf-8"))
    if (not isinstance(ids, list) or len(ids) != 20 or any(not isinstance(p, str) for p in ids)
            or len(set(ids)) != 20 or not set(ids).issubset(load_test_pmids())):
        raise ValueError("Pilot must contain 20 unique official test PMIDs.")
    rows = [json.loads(line) for line in inputs_path.read_text("utf-8").splitlines()]
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != INPUT_FIELDS:
            raise ValueError("Pilot inputs must contain only PMID, question, abstract, quote and answer.")
        if any(not isinstance(value, str) or not value.strip() for value in row.values()):
            raise ValueError("Pilot input fields must be nonempty strings.")
        if row["answer"] not in {"yes", "no", "maybe"} or row["pmid"] in by_id:
            raise ValueError("Invalid or duplicate pilot input.")
        by_id[row["pmid"]] = row
    if set(by_id) != set(ids):
        raise ValueError("Pilot IDs and blinded inputs differ.")
    return [by_id[pmid] for pmid in ids], manifest


def judge_request(model, rubric, case):
    # Replace only original template placeholders, once. JSON braces and input
    # text such as a literal '{answer}' are preserved without recursive formatting.
    keys = re.findall(r"\{(question|abstract|quote|answer)\}", rubric)
    if sorted(keys) != ["abstract", "answer", "question", "quote"]:
        raise ValueError("Rubric must contain each of the four input placeholders exactly once.")
    content = re.sub(r"\{(question|abstract|quote|answer)\}", lambda m: case[m.group(1)], rubric)
    qwen3 = model.rsplit("/", 1)[-1].lower().startswith("qwen3")
    if qwen3:
        content += "\n/no_think"
    request = dict(model=model, messages=[{"role": "user", "content": content}],
                   stream=False, format="json", options=OPTIONS)
    if qwen3:
        request["think"] = False
    return request


def request_hash(request):
    value = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def parse_judgment(raw):
    try:
        result = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid_json") from exc
    if not isinstance(result, dict) or set(result) != {"label", "reason"}:
        raise ValueError("invalid_judge_schema")
    if not isinstance(result["label"], str) or result["label"] not in JUDGE_LABELS:
        raise ValueError("invalid_judge_label")
    if not isinstance(result["reason"], str) or len(result["reason"].split()) > 20:
        raise ValueError("reason_must_be_a_string_of_at_most_20_words")
    return result


def model_digest(client, model):
    for item in client.list().models:
        if item.model == model:
            return item.digest
    raise ValueError("Requested model is not installed in local Ollama: " + model)


def server_version():
    with urllib.request.urlopen(HOST + "/api/version", timeout=5) as response:
        return json.load(response)["version"]


def read_completed(path, requests):
    """Keep complete rows, recover only a torn final write, reject mixed runs."""
    completed = {}
    if not path.exists():
        return completed
    with path.open("rb+") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                if line.endswith(b"\n"):
                    raise ValueError("Corrupt completed judge row; refusing to skip it.") from exc
                handle.seek(start)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                break
            if not isinstance(row, dict):
                raise ValueError("Malformed judge row.")
            pmid = row.get("pmid")
            if not isinstance(pmid, str) or pmid not in requests or pmid in completed:
                raise ValueError("Duplicate or foreign PMID in judge output.")
            if row.get("request_sha256") != request_hash(requests[pmid]):
                raise ValueError("Completed judge request changed.")
            latency = row.get("latency_seconds")
            if (not isinstance(row.get("raw_response"), str) or not isinstance(row.get("valid"), bool)
                    or isinstance(latency, bool) or not isinstance(latency, (int, float))
                    or not math.isfinite(latency) or latency < 0):
                raise ValueError("Malformed completed judge metadata.")
            completed[pmid] = row
            if not line.endswith(b"\n"):
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return completed


def run_judge(model, pilot_ids, rubric_path, output_dir, *, client=None):
    cases, pilot = load_pilot(pilot_ids, rubric_path)
    rubric = rubric_path.read_text("utf-8")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", model):
        raise ValueError("Invalid Ollama model name.")
    if ":" not in model.rsplit("/", 1)[-1]:
        model += ":latest"
    requests = {case["pmid"]: judge_request(model, rubric, case) for case in cases}
    if client is None:
        from ollama import Client

        client = Client(host=HOST, timeout=180.0, trust_env=False)
    digest = model_digest(client, model)
    version = server_version()
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    journal, state_path = output_dir / f"{slug}.jsonl", output_dir / f"{slug}.manifest.json"
    config = {
        "model": model, "model_digest": digest, "ollama_version": version,
        "options": OPTIONS, "format": "json", "no_think": "think" in next(iter(requests.values())),
        "pilot_ids_sha256": pilot["pilot_ids_sha256"], "pilot_inputs_sha256": pilot["pilot_inputs_sha256"],
        "rubric_sha256": pilot["rubric_sha256"], "script_sha256": text_hash(Path(__file__)),
    }
    with FileLock(str(journal) + ".lock", timeout=0):
        if state_path.exists():
            state = json.loads(state_path.read_text("utf-8"))
            if state["config"] != config:
                raise ValueError("Judge model, digest, settings, rubric, inputs or script changed on resume.")
            state["resume_count"] += 1
        else:
            if journal.exists():
                raise ValueError("Cannot adopt judge outputs without a matching manifest.")
            state = {"config": config, "started_at_utc": now(), "resume_count": 0, "gold_blind": True}
        completed = read_completed(journal, requests)
        state.update(status="running", completed_cases=len(completed))
        atomic_json(state_path, state)
        print(f"{model}: {len(completed)}/20 completed; remaining {20 - len(completed)}", flush=True)
        try:
            with journal.open("a", encoding="utf-8", newline="\n") as handle:
                for case in cases:
                    pmid = case["pmid"]
                    if pmid in completed:
                        continue
                    if model_digest(client, model) != digest:
                        raise ValueError("Judge model digest changed during the run.")
                    started = perf_counter()
                    response = client.chat(**requests[pmid])
                    latency = perf_counter() - started
                    if model_digest(client, model) != digest:
                        raise ValueError("Judge model digest changed during the run.")
                    raw, judgment, error = response.message.content or "", None, None
                    try:
                        judgment = parse_judgment(raw)
                        if not response.done or response.done_reason != "stop":
                            raise ValueError("incomplete_generation")
                        if config["no_think"] and response.message.thinking:
                            raise ValueError("unexpected_thinking")
                    except ValueError as exc:
                        judgment, error = None, str(exc)
                    row = {
                        "pmid": pmid, "request_sha256": request_hash(requests[pmid]),
                        "raw_response": raw, "judgment": judgment, "valid": error is None,
                        "validation_error": error, "latency_seconds": latency,
                        "done_reason": response.done_reason, "recorded_at_utc": now(),
                        "prompt_eval_count": response.prompt_eval_count, "eval_count": response.eval_count,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed[pmid] = row
                    state["completed_cases"] = len(completed)
                    atomic_json(state_path, state)
                    print(f"{len(completed):02d}/20 PMID={pmid} valid={row['valid']}", flush=True)
        except BaseException as exc:
            state.update(status="interrupted", completed_cases=len(completed), last_error_type=type(exc).__name__)
            atomic_json(state_path, state)
            raise
        state.update(status="completed", completed_cases=20, finished_at_utc=now(),
                     invalid_outputs=sum(not row["valid"] for row in completed.values()))
        atomic_json(state_path, state)
    return journal


def main():
    parser = argparse.ArgumentParser(description="Explicitly run one gold-blind local judge pilot (20 cases).")
    parser.add_argument("--model", required=True, help="Installed Ollama model, e.g. qwen3:8b")
    parser.add_argument("--pilot-ids", type=Path, default=ROOT / "results/judge-pilot/pilot_ids.json")
    parser.add_argument("--rubric", type=Path, default=ROOT / "prompts/judge_rubric_v1.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/judge-pilot")
    args = parser.parse_args()
    try:
        print(run_judge(args.model, args.pilot_ids, args.rubric, args.output_dir))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
