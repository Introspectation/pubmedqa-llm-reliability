# Small LLM Reliability Evaluation for Medical Question Answering

A research harness for evaluating Qwen3-1.7B on medical questions with supplied abstracts. It covers generated answers and quotes in Ollama, held-out label scoring before and after QLoRA, and a blinded human safety screen on synthetic cases.

- **Phase 1:** 65.4% accuracy and 0.469 macro-F1 on the 500 official PubMedQA test cases. 483/500 quotes matched the supplied abstract.
- **Q4:** With the same Hugging Face scorer across all three variants, macro-F1 was 0.409 for Base, 0.523 for adapter A, and 0.547 for adapter B. Performance on `maybe` declined after fine-tuning.
- **Q4b:** All three variants passed 0/16 synthetic risk cases and 2/4 controls under the frozen rubric.

Start with the [CPU reproduction guide](repro/qlora/README.md) to verify the saved Q4 predictions. The [original specification](<PROJECT_TEMPLATE (2).md>) records the starting scope.

## Structure

```text
medical_qa/
  dataset.py       # official PMID selection and pinned HF source loading
  baselines.py     # always-yes and majority from published PQA-L priors
  scoring.py       # final-only gold access and label correctness metrics
  benchmark.py     # resumable official-500 inference, validation, metrics, reports
  validation.py    # strict response JSON and normalized quote matching
  inference.py     # local Ollama /api/chat and model digest verification
  prompt.txt       # frozen, zero-shot prompt
  smoke.py         # deterministic 20-case selection and run artifacts
  __main__.py      # validation, smoke, model check, and final baseline scoring
tests/              # offline unittest tests; toy fixtures never enter evaluation
prompts/judge_rubric_v1.txt # support-only rubric, separate from the frozen QA prompt
scripts/prepare_judge_pilot.py # seed-42 stratification; gold used only in preparation
scripts/run_judge.py # explicit, resumable Ollama judge run on blinded pilot inputs
data/test_pmids.json # 500 official IDs only; no gold answers
sources.lock.json   # immutable upstream revisions and source checksums
model.lock.json     # qwen3:1.7b full manifest digest
requirements.txt    # original 47 frozen dependency pins
notebooks/          # original Run A Colab notebook
repro/qlora/        # Run B notebook, Q4 audit, Q4b protocol and results
results/            # Phase 1 aggregates and local judge pilot artifacts
.cache/             # ignored source/download cache
runs/               # local journals, new run outputs, and model files
```

The package runs directly from the repository root. No editable install or additional test dependency is needed.

## Windows setup

Use the existing Python 3.12 environment. These commands do not activate, recreate, or modify it:

```powershell
.\venv\Scripts\python.exe --version
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

For a fresh checkout without an environment, create a Python 3.12 venv and install the pinned dependencies.

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m unittest discover -s tests
```

The release check reuses the existing Python 3.12 Windows environment. If a pin fails on another Windows machine, record the exact error before changing versions.

Install/start [Ollama](https://ollama.com/download/windows), then obtain and verify the target:

```powershell
ollama pull qwen3:1.7b
.\venv\Scripts\python.exe -m medical_qa check-ollama
```

Inference connects only to `http://127.0.0.1:11434`. It ignores proxy/environment host overrides and never calls a remote inference service. A missing model or a digest different from `model.lock.json` stops inference. Pulling a tag later may retrieve different bytes; do not silently replace the lock or compare results across changed digests.

## Data selection and isolation

The [Hugging Face dataset card](https://huggingface.co/datasets/qiaojin/PubMedQA/raw/323752ad243ce751ba75c9222d34a3130d080030/README.md) describes `pqa_labeled` as a 1,000-row `train` split. That split name does **not** define this project's evaluation set.

The official test set comes from the keys of [PubMedQA's test_ground_truth.json](https://github.com/pubmedqa/pubmedqa/blob/1cbae8e92f72f20c8d3747cbb3bf5bc53554d997/data/test_ground_truth.json). The vendored `data/test_pmids.json` contains those 500 IDs, sorted numerically. Gold values were discarded programmatically, without inspecting labels or computing their distribution. A canonical ID fingerprint protects the list; `--refresh-ids` optionally downloads the pinned official JSON and verifies its keys again, without saving its values.

```powershell
# Normal network-enabled operation: fetch exactly the pinned HF Parquet file.
.\venv\Scripts\python.exe -m medical_qa validate-dataset --output runs/dataset-validation.json

# Optional independent recheck against the pinned official GitHub file.
.\venv\Scripts\python.exe -m medical_qa validate-dataset --refresh-ids
```

The loader verifies the HF Parquet SHA-256, reads the PMID column, rejects duplicate/missing IDs, selects exactly the 500 official rows, and only then exposes question/abstract text as Python `Case` objects. Each case contains just `pmid`, `question`, and `contexts`. Prompts contain only `question` and `context.contexts`; no PMID, gold label, `long_answer`, context labels, MeSH terms, or reasoning hints are supplied to the model.

The other 500 examples are excluded from inference inputs, prompts, prediction logs, and analysis. Their text and labels are never iterated, displayed, or analyzed. Their IDs are read only to validate source uniqueness and membership. Upstream distributes all 1,000 rows in one Parquet file, so the ignored raw/cache files necessarily contain the full source; that storage is not an evaluation export. Do not inspect the reserved rows or commit the cache.

### Offline/manual download

If session networking blocks Hugging Face, download [the pinned Parquet file](https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/323752ad243ce751ba75c9222d34a3130d080030/pqa_labeled/train-00000-of-00001.parquet?download=true) into `.cache/`. The current local copy is `.cache/train-00000-of-00001.parquet`.

```powershell
.\venv\Scripts\python.exe -m medical_qa validate-dataset `
  --parquet .cache\train-00000-of-00001.parquet `
  --output runs/dataset-validation.json
```

Local files must pass the same checksum as remote files. An arbitrary substitute or rebuilt dataset is rejected.

## Structured inference and evidence checks

The [Ollama chat API](https://docs.ollama.com/api/chat) receives the exact schema in its `format` parameter, `stream=false`, and `think=false`. The settings are temperature `0`, seed `0`, context window `8192`, and maximum output `512` tokens. The frozen prompt has no examples, label priors, or test-derived hints.

```json
{
  "type": "object",
  "properties": {
    "label": {"type": "string", "enum": ["yes", "no", "maybe"]},
    "evidence": {"type": "string"}
  },
  "required": ["label", "evidence"],
  "additionalProperties": false
}
```

Responses are also validated locally: wrong types, extra/missing fields, duplicate keys, markdown wrappers, and trailing text are invalid. Invalid outputs are recorded without extraction, repair, retry, or prompt adjustment. Incomplete/truncated generations and unexpected thinking output are technical failures.

For schema-valid outputs:

```python
norm = lambda s: " ".join(s.lower().split())
grounded = len(norm(evidence)) >= 20 and any(
    norm(evidence) in norm(section) for section in contexts
)
```

A quote must occur within a single abstract section; matching cannot span section boundaries. Punctuation is preserved. Schema validity and quote grounding are separate checks: an empty evidence string is valid JSON but fails the quote check. A matched quote does **not** establish entailment, a correct answer, or medical reliability. Semantic judging remains outside M0/M1.

## Run the 20-case smoke test

```powershell
.\venv\Scripts\python.exe -m medical_qa smoke `
  --parquet .cache\train-00000-of-00001.parquet
```

The sample is always the first 20 official test PMIDs ranked by ascending SHA-256 of `pubmedqa-m0-m1-v1:<PMID>`. Selection depends only on IDs, not text, labels, or previous outcomes. Each case uses a new two-message conversation. There are exactly 20 attempts, with no retries or adaptive selection. The smoke command always uses 20 cases; the separate `full-inference` command always uses the official 500.

Each run creates a new `runs/smoke-<UTC timestamp>/` directory containing:

- `manifest.json`: selected PMIDs, source revisions/checksums, full model digest, prompt hash, generation settings, schema, Python/package/server versions, code hashes, and run status. Written before the first inference request.
- `predictions.jsonl`: one record per attempted case, raw model output, parsed answer, quote-check reason, latency, and token counts when inference completes. Technical failures stay in the denominator.
- `summary.json`: attempted cases, valid/invalid JSON counts, quote-check counts, technical failures, and elapsed time. No accuracy, F1, gold labels, or judge results.

`--output runs/my-new-run` chooses an explicit directory. Existing smoke directories are rejected instead of overwritten. An interrupted run retains its `running` manifest and completed JSONL records; it is not reported as a complete 20-case run. Dataset/preflight failures and runs with technical inference failures return a nonzero exit status. Invalid JSON and unmatched quotes are measured model outcomes, not process errors.

Deterministic settings and sampling support reproducibility. Identical outputs across different hardware, drivers, or Ollama versions are not guaranteed. Keep the manifest with any reported results; do not tune the prompt using this held-out smoke sample.

## Fixed baselines and final scoring

Both baselines are defined before opening test gold:

| Baseline | Frozen prediction rule |
| --- | --- |
| `always_yes` | Predict `yes` for every official test PMID. |
| `published_majority` | Choose the largest published PQA-L class prior, then predict that class for every official test PMID. |

[Jin et al. (2019), Table 1](https://arxiv.org/pdf/1909.06146#page=3) reports PQA-L priors of **yes 55.2%, no 33.8%, maybe 11.0%**. These published statistics describe the full 1,000-example PQA-L subset, not the selected 500-case test split. They are fixed constants in `baselines.py`; no test-label counts or learned parameters determine the baseline. The published majority is **yes**, so **`published_majority` is identical to `always_yes`**. Final reports retain both entries and explicitly include `baselines_identical: true` and the source citation.

Baseline prediction generation accepts only PMIDs. It does not receive questions, abstracts, or gold labels, and makes no Ollama or judge calls. These are label-only baselines; evidence/grounding scores do not apply to them.

**Run the following command only at the final scoring stage.** It is separate from validation and the 20-case gold-blind smoke test:

```powershell
.\venv\Scripts\python.exe -m medical_qa final-score-baselines `
  --parquet .cache\train-00000-of-00001.parquet `
  --output runs\baselines-final.json
```

The final scorer first freezes both prediction maps. It then verifies the same pinned Parquet checksum, reads IDs alone to check source uniqueness and all official matches, and reads only `pubid`/`final_decision` with an official-test-ID filter. The other 500 labels are never exposed as Python records or scored. The original inference loader continues to exclude gold columns.

For **exactly the official 500 PMIDs**, both baselines receive:

- `accuracy`: correct labels divided by 500.
- `per_class_f1`: separate F1 scores for `yes`, `no`, and `maybe`.
- `macro_f1`: `(F1_yes + F1_no + F1_maybe) / 3`, where `F1 = 2PR / (P + R)`.

The fixed label order is `yes`, `no`, `maybe`; zero-division F1 cases receive `0`. All three classes remain in the macro average even if a class has no predictions or support. Labels are aligned by PMID, not input ordering. Missing, extra, duplicate, or non-official records and invalid labels are rejected; partial smoke predictions cannot be scored as the final set.

The JSON report contains aggregate metrics for both baselines, their definitions/identity, published priors/citation, source provenance, dataset validation, timestamp, and metric-library version. It contains no individual gold labels or record text. Existing report files are refused before gold is opened. This command does not run the 500-case model benchmark or score the 20-case smoke predictions.

Baseline and scoring tests use toy unit fixtures only. At the baseline-extension checkpoint, **57 tests** passed (20 added for that extension), including known metric values, fixed majority selection, PMID alignment, reserved-label isolation, absent classes, and final-only gold access. Gold-blind source validation confirmed 500 unique official matches and 500 excluded rows. No real gold labels or test-label distribution were inspected while adding that functionality. The original prompt, requirements, source locks, and model digest remain unchanged.

## Full Phase 1 inference and resume

Run from the repository root using the original completed smoke manifest:

```powershell
.\venv\Scripts\python.exe -m medical_qa full-inference `
  --parquet .cache\train-00000-of-00001.parquet `
  --output runs\phase1-full `
  --smoke-manifest runs\smoke-m0-m1\manifest.json
```

The exact same command resumes an interrupted run. It checks the frozen smoke prompt hash, model digest, Ollama/package versions, JSON schema property order, generation options, source pins, and original inference/validation code hashes before requesting predictions. A resumed run also requires its full-run code hashes and metric policy to match its manifest. No override flags change the prompt, model, schema, settings, or cohort.

The run selects exactly the 500 official PMIDs and processes them in numeric order, using fresh conversations. It does not import predictions from the earlier 20-case smoke run. Each completed HTTP response is appended to `predictions.jsonl`, flushed, and fsynced before proceeding. Resume verifies each row's PMID and frozen-request hash, then skips completed IDs, including invalid model outputs. Duplicate/foreign IDs, changed inputs, and corrupt complete rows are rejected. A torn final JSONL write is truncated; a complete final JSON object missing only a newline is retained. An OS-backed file lock prevents simultaneous writers.

Transport/preflight failures stop the run and create an entry in `errors.jsonl`; no response means that PMID remains pending for restart. Incomplete/truncated model responses are preserved and counted as invalid outcomes without repair or setting changes. The manifest records progress and interruption state.

Only after all 500 responses exist does the final stage:

1. Reapply the existing strict JSON and normalized >=20-character exact-quote checks to every response, saving `validated-predictions.jsonl` before any gold access.
2. Freeze both baseline prediction maps, then load only official test gold through the existing filtered final-scoring loader.
3. Write `results-summary.json` and a short `report.md`, and mark `manifest.json` completed.

The fixed reporting policy keeps **all 500 cases** in the denominator. Invalid JSON/schema, incomplete generations, and unexpected thinking output count as incorrect answers and use a dedicated `invalid` prediction column in the confusion matrix. Schema-invalid and generation-error counts are also reported separately. Macro-F1 averages only `yes`, `no`, and `maybe`, with `zero_division=0`; invalid responses contribute false negatives for their true class. Grounded rate is valid, complete responses passing the frozen quote check divided by 500. Ungrounded answers still receive normal label correctness scoring.

The summary/report includes accuracy, macro-F1, class precision/recall/F1/support, confusion matrices, invalid/grounded counts and rates, mean/p95 chat latency, both baselines side by side, provenance and file hashes, and zero judge calls. Baseline evidence and latency rates are not applicable to these label-only rules. Latency includes loading when present, excludes digest preflight, and uses all 500 completed HTTP response times; failed transport attempts are counted separately. P95 uses NumPy's linear interpolation.

**67 offline tests pass**, including interruption/resume, torn-write recovery, duplicate/input-hash rejection, frozen-schema order, exact request equivalence with smoke, the 500-response gold barrier, invalid-output accounting, and metric values. Unit fixtures do not depend on ignored local run artifacts or real gold.

## Full-run results — 2026-10-31

The first full run completed **500/500 unique official test PMIDs**, using the original smoke prompt, digest and settings. All responses were saved and validated before final gold scoring. There were no transport failures, prompt/setting changes, or judge calls.

| System | Evaluation set | Accuracy | Macro-F1 | Report |
| --- | --- | ---: | ---: | --- |
| qwen3:1.7b base (no fine-tune) | 500 official test | 65.4% | 0.469 | [report](results/phase1-full/report.md) |
| `always_yes` | 500 official test | 55.20% (276/500) | 0.2371 | [report](results/phase1-full/report.md) |
| `published_majority` | 500 official test | 55.20% (276/500) | 0.2371 | [report](results/phase1-full/report.md) |

Held-out: 500 official test cases, never used for training.

- Invalid outputs: **0/500 (0%)**; strict quote checks passed: **483/500 (96.6%)**, failed: **17/500**.
- Mean chat latency: **0.507 seconds**; p95: **0.939 seconds**. All 500 completed request times are included.
- Per-class model F1: **yes 0.7596, no 0.5986, maybe 0.0488**. For `maybe`, 2 of 55 cases were correctly labeled (recall 0.0364); high quote-match rate does not imply high label correctness.
- Both baselines are identical by definition, fixed from the published priors before test gold access.

Versioned results are available in the [short report](results/phase1-full/report.md), [summary JSON](results/phase1-full/results-summary.json), [frozen run manifest](results/phase1-full/manifest.json), and [verification record](results/phase1-full/verification.json). These are exact copies of the completed run's aggregate/provenance artifacts. The short report includes precision/recall/F1 for every class and system, confusion matrices, and the side-by-side comparison. The verification record covers official-ID membership, frozen request/configuration hashes, artifact checksums, both validators, independent TP/FP/FN metric calculations, and independently computed mean/linear-p95 latency. Raw and validated prediction journals remain local in the ignored `runs/phase1-full/` directory. Reserved examples were not inspected or evaluated.

## QLoRA held-out evaluation (Q4)

A trained on 450 labeled examples. B used those same 450 examples plus 2,000 balanced artificial yes/no examples, repeating the 50 training `maybe` examples four times in total. Both used the same 50-case dev split. The official 500 had zero PMID overlap with train, dev, and artificial. [Data manifest](data/qlora_dataset_manifest.json) · [Q4 precheck](repro/qlora/optimized/q4_precheck.json)

All three Hugging Face variants used the pinned Qwen3-1.7B revision, the frozen system prompt, non-thinking rendering, and the same NF4 inference setup. `label_score_v1` chooses the most probable of `yes`, `no`, and `maybe` at the JSON label position. The three prediction sets were saved before final gold access. [Q4 results and confusion matrices](repro/qlora/optimized/q4_heldout_eval.json) · [Independent PMID-aligned rescore](repro/qlora/optimized/q4_rescore_validation.json)

| Variant | Correct / 500 | Accuracy | Macro-F1 | F1 for `maybe` |
| --- | ---: | ---: | ---: | ---: |
| HF Base | 204/500 | 40.8% | 0.4091 | 0.2313 |
| HF A · labeled only | 311/500 | 62.2% | 0.5234 | 0.1509 |
| HF B · labeled + artificial | 353/500 | 70.6% | 0.5468 | 0.1348 |

`maybe` recall fell from 37/55 for Base to 12/55 for A and 6/55 for B. A and B also differed in training volume, batching, and dev checkpoint selection, so the B–A difference cannot isolate the effect of artificial data. The held-out set was previously scored and inspected during Phase 1. The Ollama row above uses GGUF generation with a schema and an exact-quote check, a separate runtime and protocol from the HF label scorer.

## Synthetic safety screen (Q4b)

Base, A, and B each produced one response for each of 20 synthetic cases. The author annotated all 60 responses with AI assistance while model identities were hidden. The annotations were locked before unblinding.

| Variant | Risk pass | Control pass | Reached token cap |
| --- | ---: | ---: | ---: |
| Base | 0/16 | 2/4 | 7/20 |
| A | 0/16 | 2/4 | 4/20 |
| B | 0/16 | 2/4 | 4/20 |

All 20 paired pass/fail decisions agreed across variants. The strict rubric produced a floor on risk cases, and the report records differences visible in individual responses. Severity and item-level coverage were unscored. This small synthetic screen establishes neither clinical readiness nor equivalent safety. [Frozen rubric](repro/qlora/optimized/q4b_safety_rubric.md) · [Safety report](repro/qlora/optimized/q4b_safety_report.md)

## Local judge pilot for quote support

The 20-case pilot used 10 correct and 10 incorrect answers sampled from the 483 exact-quote matches using seed 42. Two local Ollama judges saw the question, supplied abstract, quote, and model answer. PubMedQA gold labels stayed out of the judge input. The three support labels came from the [fixed rubric](prompts/judge_rubric_v1.txt).

| Judge | Valid outputs | Agreement with initial human labels | Cohen's kappa |
| --- | ---: | ---: | ---: |
| Qwen3 8B | 20/20 | 9/20 | ?0.038 |
| Ministral 3 | 20/20 | 8/20 | 0.055 |

The human labels are in the [pilot workbook](results/judge-pilot/human_labels.xlsx). The captured judgments and model manifests are in [the pilot artifacts](results/judge-pilot/). The `human_labels.csv` file is the original blank template. Sampling balanced correctness rather than quote support, so this pilot measures agreement on the selected cases. It cannot estimate the support rate across all 483 matching quotes.

To run an installed local judge on the sealed pilot, use `scripts/run_judge.py --model <model-tag> --output-dir runs/judge-pilot-replay`. The runner keeps completed PMIDs on resume and verifies the rubric, inputs, and request configuration.

## Initial technical observations — 2026-10-31

- Windows 11, Python **3.12.4**. All **47** original requirement pins match installed versions; `pip check` reports no broken requirements. Runtime imports and the offline unit suite pass. No environment recreation, package installation, or version changes were needed. A fresh Windows installation of every wheel was not tested.
- Ollama server **0.40.13** was available. The required `qwen3:1.7b` tag was initially absent and was pulled. Full digest: `8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7`. Ollama reports GGUF / `Q4_K_M` quantization.
- The frozen Ollama Python SDK **0.6.3** has no public `Client.version()` method. The harness reads `/api/version` over local HTTP instead; dependency pins remain unchanged.
- Session restrictions blocked direct Python/HF downloads. Browser automation also stopped because its policy checker could not verify the current URL. The user supplied the pinned HF file manually; the loader verified its **1,075,513 bytes** against SHA-256 `3d56bd1abc11884579ecc3aab9dd3cdfce8b7cf54715daeca93b8022b7be231c`.
- Live dataset validation passed: **1,000 source rows / 1,000 unique source PMIDs; 500 official IDs / 500 unique matches; 500 excluded rows**. Only counts and the official ID fingerprint were reported.
- **37 offline unit tests passed**, covering selection/isolation, malformed sources, strict JSON, evidence boundaries/normalization, fixed request settings, model digest checks, deterministic sampling, and failure accounting.

The initial real smoke run is stored locally at `runs/smoke-m0-m1/`:

| Check | Observed result |
| --- | ---: |
| Official test cases attempted | 20 |
| Schema-valid responses | 20 |
| Invalid JSON/schema responses | 0 |
| Normalized quote checks passed | 19 |
| Normalized quote checks failed | 1 |
| Technical inference errors | 0 |
| Total wall time | 78.60 seconds |
| Mean chat request time | 3.81 seconds |
| External judge calls | 0 |

Request timing includes model loading when it occurs, and excludes the separate model-digest preflight checks. This initial smoke run was an engineering check, not an accuracy estimate. The unmatched quote was retained as an observed outcome; no prompt was selected, changed, or retried in response. That smoke checkpoint performed no gold scoring or external judge calls; the later full run uses exactly its frozen prompt/model/settings.

The starting workspace contained the specification, dependencies, and existing venv, and was not a Git repository at that checkpoint. The cache, venv, private learning exercises, and local runs are ignored by Git.

## Evaluation scope

The exact-quote check measures whether the supplied text contains the normalized quote. Evidence support requires a separate judgment about what the quote says in context. Classification, quote support, and the synthetic safety screen each measure a different outcome. This repository is intended for research and portfolio evaluation.
