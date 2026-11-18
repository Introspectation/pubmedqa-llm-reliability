# Reproduce the QLoRA results

## CPU verification from a clone

Run from the repository root. Use the [Windows setup](../../README.md#windows-setup) to install the pinned dependencies. For the final rescore, download the [pinned PQA-L Parquet](https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/323752ad243ce751ba75c9222d34a3130d080030/pqa_labeled/train-00000-of-00001.parquet) into `.cache/train-00000-of-00001.parquet` or pass its path to `--parquet`. The loader verifies its SHA-256 against [sources.lock.json](../../sources.lock.json).

```powershell
$projectPython = '.\venv\Scripts\python.exe'
& $projectPython -m unittest discover -s tests
& $projectPython -m unittest discover -s repro\qlora\optimized
& $projectPython repro\qlora\optimized\q4_result_audit.py --check-inputs
& $projectPython repro\qlora\optimized\q4_result_audit.py --run `
  --parquet .cache\train-00000-of-00001.parquet `
  --output runs\qlora\verification\q4_rescore_validation.json
& $projectPython repro\qlora\optimized\q4b_safety_eval.py --folder repro\qlora\optimized --check-rubric
```

The audit verifies 500 official PMIDs per variant and checks the saved predictions against the hashes from the original run. It then loads gold for those PMIDs, aligns labels by PMID, and recomputes the accuracy, macro-F1, per-class metrics, and confusion matrices. Choose a fresh `--output` path on each run. The Q4b command verifies the frozen rubric and case count. [Q4 aggregate](optimized/q4_heldout_eval.json) · [Reference rescore](optimized/q4_rescore_validation.json) · [Q4b report](optimized/q4b_safety_report.md)

The [clean-clone verification record](VERIFICATION.md) lists the checks exercised on Windows and the remaining environment requirements.

The [train, dev, and artificial splits](../../data/QLORA_DATA.md) are included. To rebuild them from the pinned sources into an empty directory, run:

```powershell
& $projectPython data\prepare_qlora_data.py --output-dir runs\qlora-data-rebuilt
```

This uses the PQA-L file above and downloads the pinned PQA-A Parquet. Pass `--artificial-parquet` to use a local copy. The builder checks for overlap with the 500 official PMIDs before it writes each split. The original file digests and label counts are in the [dataset manifest](../../data/qlora_dataset_manifest.json).

## GPU experiment entry points

| Stage | Colab notebook | Inputs |
| --- | --- | --- |
| Train A | [Original QLoRA](../../notebooks/pubmedqa_qlora_colab.ipynb) | Repository `data/`, Train A cell |
| Train B | [Optimized QLoRA](optimization/pubmedqa_qlora_optimized_colab.ipynb) | Same `data/`, Run B cell with its start flag |
| Q4 held-out | [Q4 evaluation](optimized/q4_heldout_eval.ipynb) | `data/`, [runner](optimized/q4_heldout_eval.py), original A/B run folders with `best_adapter/` and metadata |
| Q4b safety | [Q4b evaluation](optimized/q4b_safety_eval.ipynb) | [Runner](optimized/q4b_safety_eval.py), [rubric](optimized/q4b_safety_rubric.md), [manifest](optimized/q4b_manifest.json), original A/B run folders |

The notebooks pin their Colab packages and use paths under `MyDrive/AI_PUBMED`. Upload `data/` there for training and Q4. For Q4 and Q4b, place the listed Python and rubric files at the notebook's expected Drive path, `MyDrive/AI_PUBMED/runs/qlora/optimized/`, or edit the path cell. Use a fresh Drive output folder when rerunning an experiment. The existing adapters, training logs, and private Q4b review files are required for exact replay of their respective original stages and are outside this checkout. The notebooks and runner behavior are preserved as used.

Q4's [tokenizer precheck](optimized/q4_precheck.json) verified prompt parity and zero held-out overlap. The 1,024-token training cap affected 0/450 train and 1/2,000 artificial records. All 50 dev and 500 held-out scoring inputs also fit within 1,024 tokens.

PubMedQA's [pinned dataset card](https://huggingface.co/datasets/qiaojin/PubMedQA/blob/323752ad243ce751ba75c9222d34a3130d080030/README.md) lists MIT. The upstream [license notice](../../data/PubMedQA-LICENSE.txt) accompanies the included data. [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) lists Apache 2.0.
