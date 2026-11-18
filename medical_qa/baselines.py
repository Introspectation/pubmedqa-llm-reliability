"""Frozen label-only baselines; no gold labels, text, or model calls are needed."""

from types import MappingProxyType

from .dataset import validate_pmids

LABELS = ("yes", "no", "maybe")
PUBLISHED_PQA_L_PRIORS_PERCENT = MappingProxyType({"yes": 55.2, "no": 33.8, "maybe": 11.0})
PRIORS_CITATION = "Jin et al. (2019), PubMedQA, Table 1, PQA-L (1,000 examples)"
PRIORS_URL = "https://arxiv.org/pdf/1909.06146#page=3"
PUBLISHED_MAJORITY_LABEL = max(
    PUBLISHED_PQA_L_PRIORS_PERCENT, key=PUBLISHED_PQA_L_PRIORS_PERCENT.__getitem__,
)
BASELINE_LABELS = MappingProxyType({
    "always_yes": "yes",
    "published_majority": PUBLISHED_MAJORITY_LABEL,
})


def baseline_definitions() -> dict:
    return {
        "always_yes": {"predicted_label": "yes", "rule": "Always predict yes."},
        "published_majority": {
            "predicted_label": PUBLISHED_MAJORITY_LABEL,
            "rule": "Choose the largest published PQA-L prior; never fit on test labels.",
            "published_priors_percent": dict(PUBLISHED_PQA_L_PRIORS_PERCENT),
            "priors_citation": PRIORS_CITATION,
            "priors_url": PRIORS_URL,
        },
    }


def baseline_predictions(pmids) -> dict[str, dict[str, str]]:
    """Create one prediction per ID for both baselines before opening gold."""
    ids = validate_pmids(pmids)
    return {name: {pmid: label for pmid in ids} for name, label in BASELINE_LABELS.items()}
