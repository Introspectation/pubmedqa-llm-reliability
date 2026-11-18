"""Strict JSON validation and the specification's deterministic quote check."""

import json
from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["yes", "no", "maybe"]},
        "evidence": {"type": "string"},
    },
    "required": ["label", "evidence"],
    "additionalProperties": False,
}


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    label: Literal["yes", "no", "maybe"]
    evidence: str


class AnswerValidationError(ValueError):
    """The raw model output does not satisfy the exact response schema."""


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise AnswerValidationError("duplicate_json_key")
        obj[key] = value
    return obj


def parse_answer(raw: str) -> Answer:
    """Reject markdown, coercions, trailing text, extra keys, and duplicates."""
    try:
        obj = json.loads(raw, object_pairs_hook=_unique_object)
        return Answer.model_validate(obj)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        # Keep errors compact; do not echo input values into exception logs.
        raise AnswerValidationError("invalid_json_or_schema") from exc


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class EvidenceResult:
    grounded: bool
    reason: str
    normalized_characters: int
    context_index: int | None = None


def validate_evidence(evidence: str, contexts: Sequence[str]) -> EvidenceResult:
    quote = normalize(evidence)
    if len(quote) < 20:
        return EvidenceResult(False, "too_short", len(quote))
    for index, context in enumerate(contexts):
        if quote in normalize(context):
            return EvidenceResult(True, "matched", len(quote), index)
    return EvidenceResult(False, "not_found", len(quote))
