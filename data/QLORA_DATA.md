# PubMedQA QLoRA dataset preparation

Data preparation only. Training has not started. Existing prompts, evaluation outputs, and project files were not changed.

## Label counts

| Split | yes | no | maybe | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 248 | 152 | 50 | 450 |
| dev | 28 | 17 | 5 | 50 |
| artificial | 1000 | 1000 | 0 | 2000 |

## Files and supervision

`train.jsonl` and `dev.jsonl` contain the exact 500-record complement of the official test set, split 450/50 with label stratification and seed 42.
`artificial.jsonl` is an optional, separate 2,000-record supplement: 1,000 yes and 1,000 no, sampled with seed 42 after excluding every PQA-L PMID. It is not merged into train or dev.
Every JSONL row has `pmid`, `question`, `contexts` (original abstract sections), `label`, and `source`. Each corresponding `*_pmids.txt` matches the JSONL row order exactly.
These are source records for label supervision. No chat template or tokenization has been applied. Evidence-quote targets have not been invented; `long_answer` and other answer hints are excluded. A later trainer must explicitly choose its output/loss format before fitting the label-plus-evidence task.
Use the same dev set with or without the supplement. Do not train on dev or official held-out records. PQA-A labels are artificial and remain identified by `source`.

## Reproducibility

Both sources use [qiaojin/PubMedQA revision 323752ad243ce751ba75c9222d34a3130d080030](https://huggingface.co/datasets/qiaojin/PubMedQA/tree/323752ad243ce751ba75c9222d34a3130d080030). Source checksums, exact sampling methods, dependency versions, output hashes, and overlap-check outputs are recorded in `qlora_dataset_manifest.json`.
`prepare_qlora_data.py` can rebuild the splits into a fresh directory. It verifies the official IDs using `data/test_pmids.json` and checks for overlap before writing. It takes the pinned labeled Parquet from `.cache/` by default and downloads the pinned artificial Parquet. Use `--labeled-parquet` and `--artificial-parquet` to supply local copies.

```powershell
.\venv\Scripts\python.exe -B data\prepare_qlora_data.py --output-dir runs\qlora-data-rebuilt
```

## Held-out checks

The existing `check_training_overlap.py` was run on all three PMID files; each exited successfully. Pairwise PMID overlap between train, dev, and artificial is also zero.

### train_pmids.txt

```text
Training PMIDs (unique): 450
Held-out overlap count: 0
```

### dev_pmids.txt

```text
Training PMIDs (unique): 50
Held-out overlap count: 0
```

### artificial_pmids.txt

```text
Training PMIDs (unique): 2000
Held-out overlap count: 0
```

Run the held-out checker again on the actual final training PMID collection immediately before any future training starts, including after combining supplements.
