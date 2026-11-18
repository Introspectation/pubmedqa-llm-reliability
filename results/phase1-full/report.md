# Phase 1: official 500-case results

Frozen smoke prompt, model digest, schema order and generation settings; no judge calls.

| System | N | Accuracy | Macro-F1 | Invalid | Grounded | Mean latency | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3:1.7b | 500 | 65.40% | 0.4690 | 0.00% | 96.60% | 0.507s | 0.939s |
| always_yes | 500 | 55.20% | 0.2371 | N/A | N/A | N/A | N/A |
| published_majority | 500 | 55.20% | 0.2371 | N/A | N/A | N/A | N/A |

Both baselines predict yes and are identical; majority is fixed from published PQA-L priors.
Baseline grounding/latency rates are not applicable to these label-only rules.

## Per-class metrics

| System | Class | Precision | Recall | F1 | Support |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3:1.7b | yes | 0.6810 | 0.8587 | 0.7596 | 276 |
| qwen3:1.7b | no | 0.7040 | 0.5207 | 0.5986 | 169 |
| qwen3:1.7b | maybe | 0.0741 | 0.0364 | 0.0488 | 55 |
| always_yes | yes | 0.5520 | 1.0000 | 0.7113 | 276 |
| always_yes | no | 0.0000 | 0.0000 | 0.0000 | 169 |
| always_yes | maybe | 0.0000 | 0.0000 | 0.0000 | 55 |
| published_majority | yes | 0.5520 | 1.0000 | 0.7113 | 276 |
| published_majority | no | 0.0000 | 0.0000 | 0.0000 | 169 |
| published_majority | maybe | 0.0000 | 0.0000 | 0.0000 | 55 |

## Confusion matrix: qwen3:1.7b

Rows = gold; columns = prediction.

| Gold | yes | no | maybe | invalid |
| --- | ---: | ---: | ---: | ---: |
| yes | 237 | 27 | 12 | 0 |
| no | 68 | 88 | 13 | 0 |
| maybe | 43 | 10 | 2 | 0 |

## Confusion matrix: Both identical baselines

Rows = gold; columns = prediction.

| Gold | yes | no | maybe | invalid |
| --- | ---: | ---: | ---: | ---: |
| yes | 276 | 0 | 0 | 0 |
| no | 169 | 0 | 0 | 0 |
| maybe | 55 | 0 | 0 | 0 |

## Interpretation and provenance

- 0 invalid, 483 grounded, 17 ungrounded out of 500. Invalid outputs count as incorrect.
- Grounded means a normalized >=20-character quote occurs within one supplied section; it does not prove entailment or clinical reliability.
- The official 500 test IDs were selected before reading examples. Reserved records were not inspected or evaluated.
- JSON and exact-quote checks were applied to all 500 completed responses before gold access. No repair, prompt tuning, or judge calls.
- Model digest: `8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7`.
- Prompt SHA-256: `4b10f5318515f435f9f0a47e49baedcb994269174608532959aec6eba5610920`.
- Settings: `{"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 512}`, `think=false`.
- Latency includes model load when present and excludes digest preflight; p95 uses linear interpolation.
- Transport failures recorded across attempts: 0.
- Full precision metrics, hashes, baselines' citation and software provenance are in `results-summary.json` and `manifest.json`.
