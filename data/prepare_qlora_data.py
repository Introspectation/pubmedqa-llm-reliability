"""Prepare frozen PubMedQA train/dev data and a separate artificial supplement.

No training, inference, prompt selection, or evaluation-result access occurs here.
Run from the repository root with the existing venv and Python's -B option.
"""

import argparse
import hashlib
import io
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_qa.dataset import load_test_pmids

OUTPUT = ROOT / "data"
SEED = 42
LABELS = ("yes", "no", "maybe")
REVISION = "323752ad243ce751ba75c9222d34a3130d080030"
ARTIFICIAL_FILE = "pqa_artificial/train-00000-of-00001.parquet"
ARTIFICIAL_SHA256 = "cdee2d3c01369f47d2ea7b89cd32ee64b142cc0debcb863bf39b3a81a3cece05"
ARTIFICIAL_URL = f"https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/{REVISION}/{ARTIFICIAL_FILE}"
OUTPUT_NAMES = (
    "train.jsonl", "dev.jsonl", "artificial.jsonl",
    "train_pmids.txt", "dev_pmids.txt", "artificial_pmids.txt",
    "qlora_dataset_manifest.json", "QLORA_DATA.md",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_pmid(value):
    require(not isinstance(value, bool) and isinstance(value, (int, str)), "Invalid PMID type.")
    value = str(value)
    require(re.fullmatch(r"[1-9][0-9]*", value) is not None, "Invalid PMID format.")
    return value


def select_records(source, pmids, source_name):
    """Materialize only chosen records and only the fields used for supervision."""
    table = pq.read_table(
        source, columns=["pubid", "question", "context", "final_decision"],
        filters=[("pubid", "in", [int(pmid) for pmid in pmids])],
    )
    records = {}
    for row in table.to_pylist():
        pmid = canonical_pmid(row["pubid"])
        require(pmid in pmids and pmid not in records, "Unexpected or duplicate selected PMID.")
        question, contexts, label = row["question"], row["context"]["contexts"], row["final_decision"]
        require(isinstance(question, str) and question.strip(), "Missing question.")
        require(isinstance(contexts, list) and contexts and all(
            isinstance(text, str) and text.strip() for text in contexts
        ), "Missing abstract context.")
        require(label in LABELS, "Unknown label.")
        records[pmid] = {
            "pmid": pmid, "question": question, "contexts": contexts,
            "label": label, "source": source_name,
        }
    require(set(records) == set(pmids), "Selected records do not match requested IDs.")
    return [records[pmid] for pmid in sorted(pmids, key=int)]


def download_artificial():
    """Keep the pinned source in memory so no extra cache/source file is written."""
    print("Downloading pinned PQA-A Parquet (233,411,194 bytes) into memory...", flush=True)
    with urllib.request.urlopen(ARTIFICIAL_URL, timeout=45) as response:
        raw = response.read()
    require(sha256(raw) == ARTIFICIAL_SHA256, "Artificial Parquet checksum mismatch.")
    print("Artificial source SHA-256 verified.", flush=True)
    return raw


def sample_artificial(raw, excluded_pmids):
    """Sample equal classes after exclusions, with stable numeric PMID ordering."""
    metadata = pq.read_table(io.BytesIO(raw), columns=["pubid", "final_decision"]).to_pylist()
    by_id = {}
    for row in metadata:
        pmid = canonical_pmid(row["pubid"])
        require(pmid not in by_id, "Duplicate PMID in artificial source.")
        require(row["final_decision"] in ("yes", "no"), "Unexpected artificial label.")
        by_id[pmid] = row["final_decision"]
    rng = random.Random(SEED)
    selected, available = [], {}
    for label in ("yes", "no"):
        candidates = sorted((pmid for pmid, value in by_id.items()
                             if value == label and pmid not in excluded_pmids), key=int)
        available[label] = len(candidates)
        require(len(candidates) >= 1000, f"Fewer than 1,000 eligible artificial {label} cases.")
        selected.extend(rng.sample(candidates, 1000))
    return select_records(io.BytesIO(raw), set(selected), "pqa_artificial"), {
        "source_rows": len(metadata), "source_unique_pmids": len(by_id),
        "excluded_labeled_pmids": len(set(by_id).intersection(excluded_pmids)),
        "eligible_label_counts": available,
    }


def as_json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-parquet", type=Path, default=ROOT / ".cache/train-00000-of-00001.parquet")
    parser.add_argument("--artificial-parquet", type=Path, help="Optional local copy with the pinned checksum.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="Directory for the new dataset files.")
    args = parser.parse_args()
    output = args.output_dir
    require(all(not (output / name).exists() for name in OUTPUT_NAMES),
            "An output already exists; refusing to overwrite any file.")
    lock = json.loads((ROOT / "sources.lock.json").read_text("utf-8"))
    require(lock["dataset_revision"] == REVISION, "Labeled source revision changed.")
    require(sha256(args.labeled_parquet.read_bytes()) == lock["dataset_file_sha256"],
            "Labeled Parquet checksum mismatch.")
    heldout = set(load_test_pmids())

    # Only IDs are read until the exact 500-row non-held-out complement is known.
    source_ids = [canonical_pmid(value) for value in pq.read_table(
        args.labeled_parquet, columns=["pubid"]
    ).column("pubid").to_pylist()]
    require(len(source_ids) == len(set(source_ids)) == 1000, "Expected 1,000 unique PQA-L PMIDs.")
    require(heldout.issubset(source_ids), "Held-out IDs are missing from PQA-L.")
    remaining = set(source_ids).difference(heldout)
    require(len(remaining) == 500, "Expected exactly 500 non-held-out PQA-L cases.")
    records = select_records(args.labeled_parquet, remaining, "pqa_labeled")
    train, dev = train_test_split(
        records, train_size=450, test_size=50, random_state=SEED,
        stratify=[record["label"] for record in records],
    )
    train, dev = (sorted(part, key=lambda record: int(record["pmid"])) for part in (train, dev))
    artificial_raw = args.artificial_parquet.read_bytes() if args.artificial_parquet else download_artificial()
    require(sha256(artificial_raw) == ARTIFICIAL_SHA256, "Artificial Parquet checksum mismatch.")
    artificial, artificial_info = sample_artificial(artificial_raw, set(source_ids))
    parts = {"train": train, "dev": dev, "artificial": artificial}
    sets = {name: {record["pmid"] for record in part} for name, part in parts.items()}
    require(len(train) == 450 and len(dev) == 50 and len(artificial) == 2000, "Wrong split sizes.")
    require(sets["train"] | sets["dev"] == remaining, "PQA-L split coverage changed.")
    require(not sets["train"].intersection(sets["dev"]), "Train/dev overlap.")
    require(not sets["artificial"].intersection(source_ids), "Artificial/PQA-L overlap.")
    require(all(len(sets[name]) == len(part) and not sets[name].intersection(heldout)
                for name, part in parts.items()), "Duplicate IDs or held-out overlap.")
    require(Counter(record["label"] for record in artificial) == {"yes": 1000, "no": 1000},
            "Artificial labels are not balanced.")

    artifacts = {}
    for name, part in parts.items():
        artifacts[f"{name}.jsonl"] = ("\n".join(json.dumps(record, ensure_ascii=False)
                                                for record in part) + "\n").encode("utf-8")
        artifacts[f"{name}_pmids.txt"] = ("\n".join(record["pmid"] for record in part) + "\n").encode("utf-8")
    checks = {}
    for name, ids in sets.items():
        count = len(ids & heldout)
        report = f"Training PMIDs (unique): {len(ids)}\nHeld-out overlap count: {count}\n"
        checks[name] = {"exit_code": 0, "stdout": report}
        print(f"\n{name}_pmids.txt\n{report}", end="", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    for name, raw in artifacts.items():
        with (output / name).open("xb") as handle:
            handle.write(raw)

    counts = {name: {label: sum(record["label"] == label for record in part)
                     for label in LABELS} for name, part in parts.items()}
    manifest = {
        "stage": "qlora_data_preparation_only", "seed": SEED, "training_started": False,
        "dataset_repo": lock["dataset_repo"], "dataset_revision": REVISION,
        "labeled_source": {"file": lock["dataset_file"], "sha256": lock["dataset_file_sha256"]},
        "artificial_source": {"file": ARTIFICIAL_FILE, "url": ARTIFICIAL_URL,
                              "sha256": ARTIFICIAL_SHA256, **artificial_info},
        "heldout_file": "data/test_pmids.json",
        "heldout_pmids_sha256": lock["test_pmids_sha256"],
        "split_method": "Numerically sort the 500 non-held-out PMIDs; sklearn train_test_split(train_size=450, test_size=50, stratify=labels, random_state=42). Sort exported parts numerically.",
        "artificial_method": "Exclude all 1,000 PQA-L PMIDs; sort candidates numerically per class; one random.Random(42) samples 1,000 yes then 1,000 no without replacement. Export numerically sorted.",
        "schema": ["pmid", "question", "contexts", "label", "source"],
        "label_counts": counts, "overlap_checks": checks,
        "pairwise_split_overlap_counts": {"train_dev": 0, "train_artificial": 0, "dev_artificial": 0},
        "artifact_sha256": {name: sha256(raw) for name, raw in artifacts.items()},
        "builder_sha256": sha256(Path(__file__).read_bytes()),
        "versions": {package: version(package) for package in ("scikit-learn", "pyarrow")},
    }
    documentation = [
        "# PubMedQA QLoRA dataset preparation", "",
        "Data preparation only. Training has not started. Existing prompts, evaluation outputs, and project files were not changed.", "",
        "## Label counts", "", "| Split | yes | no | maybe | Total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, part in parts.items():
        documentation.append(f"| {name} | {counts[name]['yes']} | {counts[name]['no']} | {counts[name]['maybe']} | {len(part)} |")
    documentation.extend([
        "", "## Files and supervision", "",
        "`train.jsonl` and `dev.jsonl` contain the exact 500-record complement of the official test set, split 450/50 with label stratification and seed 42.",
        "`artificial.jsonl` is an optional, separate 2,000-record supplement: 1,000 yes and 1,000 no, sampled with seed 42 after excluding every PQA-L PMID. It is not merged into train or dev.",
        "Every JSONL row has `pmid`, `question`, `contexts` (original abstract sections), `label`, and `source`. Each corresponding `*_pmids.txt` matches the JSONL row order exactly.",
        "These are source records for label supervision. No chat template or tokenization has been applied. Evidence-quote targets have not been invented; `long_answer` and other answer hints are excluded. A later trainer must explicitly choose its output/loss format before fitting the label-plus-evidence task.",
        "Use the same dev set with or without the supplement. Do not train on dev or official held-out records. PQA-A labels are artificial and remain identified by `source`.", "",
        "## Reproducibility", "",
        f"Both sources use [qiaojin/PubMedQA revision {REVISION}](https://huggingface.co/datasets/qiaojin/PubMedQA/tree/{REVISION}). Source checksums, exact sampling methods, dependency versions, output hashes, and overlap-check outputs are recorded in `qlora_dataset_manifest.json`.",
        "`prepare_qlora_data.py` verifies the frozen official IDs in `data/test_pmids.json` and writes to a new output directory. Use `--labeled-parquet` and `--artificial-parquet` to supply pinned local source copies.", "",
        "Run with `.\\venv\\Scripts\\python.exe -B data\\prepare_qlora_data.py --output-dir runs\\qlora-data-rebuilt`.", "",
        "## Held-out checks", "",
        "All three splits were checked against the frozen official IDs before writing. Pairwise PMID overlap between train, dev, and artificial is also zero.",
    ])
    for name in parts:
        documentation.extend(["", f"### {name}_pmids.txt", "", "```text", checks[name]["stdout"].strip(), "```"])
    documentation.extend(["", "Run the held-out checker again on the actual final training PMID collection immediately before any future training starts, including after combining supplements.", ""])
    for name, raw in (("qlora_dataset_manifest.json", as_json_bytes(manifest)),
                      ("QLORA_DATA.md", "\n".join(documentation).encode("utf-8"))):
        with (output / name).open("xb") as handle:
            handle.write(raw)
    print("\nLabel counts: " + json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
