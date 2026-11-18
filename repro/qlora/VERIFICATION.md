# Clean-clone verification

**2026-11-18 · Windows · Python 3.12.4.** The release files were staged in a temporary local repository and checked from a second clone with `core.autocrlf=true`. The clone used the existing Python environment. All 47 installed dependency versions matched their pins and `pip check` passed.

| Check | Observed result |
| --- | --- |
| Offline unit suites | 80 existing tests and 40 Q4/Q4b tests passed |
| Official PubMedQA split | 1,000 unique source PMIDs, exactly 500 official matches, 500 excluded from Phase 1 |
| Rebuild train, dev, and artificial | All six JSONL and PMID files matched the original SHA-256 hashes. Held-out overlaps were 0/0/0. |
| Q4 saved prediction audit | All three sets contained the official 500 PMIDs. Accuracy, macro-F1, per-class metrics, and confusion matrices matched the published Q4 JSON. |
| Q4b rubric check | 20 cases and the frozen SHA-256 matched |
| Notebook and path checks | Four notebooks parsed, all 40 Python cells passed syntax checks after excluding Colab install commands, and README relative links and six original provenance hashes resolved |

The rescore exercised official gold loading from the pinned PQA-L Parquet. The data rebuild also used the pinned PQA-A Parquet. The Q4b check covered the rubric and the saved protocol. A fresh Windows dependency installation, GPU inference, training, and a new human annotation pass were outside this verification. Exact GPU replay uses the original adapters and run metadata described in the [reproduction guide](README.md).
