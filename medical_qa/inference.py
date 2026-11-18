"""Ollama /api/chat, restricted to the frozen local target model."""

import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass
from time import perf_counter

from .dataset import ROOT, Case
from .validation import RESPONSE_SCHEMA, AnswerValidationError, parse_answer, validate_evidence

MODEL = "qwen3:1.7b"
HOST = "http://127.0.0.1:11434"
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 512}
PROMPT = (ROOT / "medical_qa/prompt.txt").read_text(encoding="utf-8")
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


class InferenceError(RuntimeError):
    """Ollama is unavailable, changed, or did not finish the response."""


def ollama_version() -> str:
    # The frozen ollama 0.6.3 SDK has no public Client.version() method.
    with urllib.request.urlopen(HOST + "/api/version", timeout=5) as response:
        return json.load(response)["version"]


def build_messages(case: Case) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(
            {"question": case.question, "contexts": case.contexts}, ensure_ascii=False,
        )},
    ]


@dataclass(frozen=True)
class Prediction:
    pmid: str
    status: str
    latency_seconds: float
    raw_response: str
    answer: dict | None
    evidence_check: dict | None
    error: str | None
    done_reason: str | None
    prompt_eval_count: int | None
    eval_count: int | None

    def to_dict(self) -> dict:
        return asdict(self)


class OllamaRunner:
    def __init__(self, *, client=None):
        if client is None:
            from ollama import Client

            client = Client(host=HOST, timeout=180.0, trust_env=False)
        self.client = client
        self.lock = json.loads((ROOT / "model.lock.json").read_text("utf-8"))
        self.server_version = None

    def verify_model(self) -> str:
        if self.lock["model"] != MODEL:
            raise InferenceError("The model lock must specify qwen3:1.7b.")
        for model in self.client.list().models:
            if model.model == MODEL:
                if model.digest != self.lock["digest"]:
                    raise InferenceError("Local model digest differs from model.lock.json.")
                if self.server_version is None:
                    self.server_version = ollama_version()
                return model.digest
        raise InferenceError("Target is missing. Run: ollama pull qwen3:1.7b")

    def predict(self, case: Case) -> Prediction:
        # Recheck immediately before every request; never silently substitute a tag.
        self.verify_model()
        started = perf_counter()
        response = self.client.chat(
            model=MODEL,
            messages=build_messages(case),
            format=RESPONSE_SCHEMA,
            think=False,
            stream=False,
            options=OPTIONS,
        )
        elapsed = perf_counter() - started
        if not response.done or response.done_reason != "stop":
            raise InferenceError("Ollama response was incomplete or generation was truncated.")
        if response.message.thinking:
            raise InferenceError("Ollama returned thinking despite think=False.")
        self.verify_model()
        raw = response.message.content
        try:
            answer = parse_answer(raw)
        except AnswerValidationError as exc:
            return Prediction(case.pmid, "invalid_json", elapsed, raw, None, None,
                              str(exc), response.done_reason,
                              response.prompt_eval_count, response.eval_count)
        evidence = validate_evidence(answer.evidence, case.contexts)
        return Prediction(
            case.pmid, "grounded" if evidence.grounded else "ungrounded", elapsed,
            raw, answer.model_dump(), asdict(evidence), None, response.done_reason,
            response.prompt_eval_count, response.eval_count,
        )
