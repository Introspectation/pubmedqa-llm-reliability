"""Select the official test split before reading any example text."""

import hashlib
import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_SIZE = 500
SOURCE_SIZE = 1000


class DatasetValidationError(ValueError):
    """A source, split, or example violated the frozen data contract."""


@dataclass(frozen=True)
class Case:
    """The only example fields available to the inference code."""

    pmid: str
    question: str
    contexts: tuple[str, ...]


@dataclass(frozen=True)
class SelectionReport:
    source_cases: int
    source_unique_pmids: int
    official_test_pmids: int
    matched_unique_cases: int
    excluded_cases: int
    test_pmids_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def source_lock() -> dict:
    return json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))


def canonical_pmid(value: int | str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise DatasetValidationError("PMIDs must be positive integers or digit strings.")
    value = str(value)
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise DatasetValidationError("Non-canonical PMID in source.")
    return value


def validate_pmids(values) -> tuple[str, ...]:
    ids = tuple(canonical_pmid(value) for value in values)
    if len(ids) != TEST_SIZE or len(set(ids)) != TEST_SIZE:
        raise DatasetValidationError("Expected exactly 500 unique official test PMIDs.")
    return tuple(sorted(ids, key=int))


def pmid_fingerprint(ids) -> str:
    canonical = "\n".join(sorted(ids, key=int))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise DatasetValidationError("Duplicate key in official test JSON.")
        obj[key] = value
    return obj


def parse_official_test_pmids(raw: bytes) -> tuple[str, ...]:
    """Read keys only. Never return, persist, print, or analyze gold values."""
    obj = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(obj, dict):
        raise DatasetValidationError("Official test JSON must be a PMID-keyed object.")
    return validate_pmids(obj.keys())


def load_test_pmids(*, refresh: bool = False) -> tuple[str, ...]:
    """Use the vendored ID-only official list, optionally reverify its origin."""
    lock = source_lock()
    if refresh:
        with urllib.request.urlopen(lock["official_test_url"], timeout=30) as response:
            ids = parse_official_test_pmids(response.read())
    else:
        ids = validate_pmids(json.loads((ROOT / "data/test_pmids.json").read_text("utf-8")))
    if pmid_fingerprint(ids) != lock["test_pmids_sha256"]:
        raise DatasetValidationError("Official test PMID fingerprint differs from the source lock.")
    return ids


def official_test_indices(source_pmids, official_pmids) -> tuple[tuple[int, ...], SelectionReport]:
    """Validate the join using IDs alone; shared by blind loading and final scoring."""
    wanted = validate_pmids(official_pmids)
    source_ids = tuple(canonical_pmid(value) for value in source_pmids)
    if len(source_ids) != SOURCE_SIZE or len(set(source_ids)) != SOURCE_SIZE:
        raise DatasetValidationError("Expected 1,000 source rows with unique PMIDs.")
    by_id = {pmid: index for index, pmid in enumerate(source_ids)}
    if not set(wanted).issubset(by_id):
        raise DatasetValidationError("Not all 500 official test PMIDs match the HF source.")
    indices = tuple(by_id[pmid] for pmid in wanted)
    report = SelectionReport(
        source_cases=len(source_ids),
        source_unique_pmids=len(set(source_ids)),
        official_test_pmids=len(wanted),
        matched_unique_cases=len(indices),
        excluded_cases=len(source_ids) - len(indices),
        test_pmids_sha256=pmid_fingerprint(wanted),
    )
    return indices, report


def select_official_cases(dataset, official_pmids) -> tuple[tuple[Case, ...], SelectionReport]:
    """Read all IDs, then expose text only for the 500 matched test rows."""
    wanted = validate_pmids(official_pmids)
    indices, report = official_test_indices(dataset["pubid"], wanted)

    # No iteration over the unfiltered records, labels, conclusions, or hints.
    selected = dataset.select(indices)
    selected = selected.flatten().select_columns(["pubid", "question", "context.contexts"])
    cases = []
    for row in selected:
        question, contexts = row["question"], row["context.contexts"]
        if not isinstance(question, str) or not question.strip():
            raise DatasetValidationError("An official test question is empty or malformed.")
        if not isinstance(contexts, list) or not contexts or any(
            not isinstance(context, str) or not context.strip() for context in contexts
        ):
            raise DatasetValidationError("An official test abstract is empty or malformed.")
        cases.append(Case(canonical_pmid(row["pubid"]), question, tuple(contexts)))
    if tuple(case.pmid for case in cases) != wanted:
        raise DatasetValidationError("Selected cases do not exactly match official test IDs.")
    return tuple(cases), report


def resolve_parquet_source(parquet: Path | None = None) -> Path:
    """Resolve and checksum the pinned source without inspecting any fields."""
    from huggingface_hub import hf_hub_download

    lock = source_lock()
    if parquet is None:
        parquet = Path(hf_hub_download(
            repo_id=lock["dataset_repo"],
            repo_type="dataset",
            filename=lock["dataset_file"],
            revision=lock["dataset_revision"],
            cache_dir=ROOT / ".cache/huggingface",
            token=False,
        ))
    if hashlib.sha256(parquet.read_bytes()).hexdigest() != lock["dataset_file_sha256"]:
        raise DatasetValidationError("HF Parquet checksum differs from the pinned source.")
    return parquet


def load_dataset(*, parquet: Path | None = None, refresh_ids: bool = False):
    """Download one pinned HF file, check bytes, then select by official PMIDs.

    A local Parquet file supports offline use but must have the same checksum.
    HF necessarily ships all 1,000 rows in one file; it is a private source cache,
    never an evaluation export. Only selected safe fields become Python records.
    """
    from datasets import Dataset

    ids = load_test_pmids(refresh=refresh_ids)
    parquet = resolve_parquet_source(parquet)
    dataset = Dataset.from_parquet(
        str(parquet), columns=["pubid", "question", "context"],
        cache_dir=str(ROOT / ".cache/datasets"),
    )
    return select_official_cases(dataset, ids)
