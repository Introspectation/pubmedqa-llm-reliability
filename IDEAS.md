# IDEAS.md

Parked ideas for the PubMedQA reliability diagnostic. Not in Phase 1 scope unless moved into PROJECT.md.

## Phase 1 add-ons (Days 8-14, if time allows)

- **Evidence-first output order (parked; prospective experiment).** Pre-register a separate `evidence -> label` variant before running it or inspecting its results: freeze prompt/schema hashes, paired evaluation PMIDs, model digest/options, and accuracy, macro-F1, JSON-validity and strict-groundedness comparisons against the unchanged `label -> evidence` main run; explicitly request and verify emitted key order. Treat any improvement as a hypothesis, report both variants, and keep the main run fixed. Existing main-run smoke results have already been observed; preregistration applies prospectively to the new variant only.
- **Near-quote diagnostic.** Keep strict exact-quote match as the headline groundedness metric. Additionally report a separate "near-quote" bucket (normalized similarity >= 0.90, e.g. difflib ratio or rapidfuzz) for evidence that fails the strict check. Never count near-quotes as grounded. Purpose: separate sloppy copying from real fabrication. Example: PMID 17032327 changed one word ("mesorectal excised") and failed strict.
- **Numbers-only slice.** Jin et al. 2019: 96.5% of PQA-L needs quantitative reasoning; 21.0% of contexts give only numbers with no text interpretation. Tag each case with a keyword heuristic (significant, no difference, improved, reduced, associated, ...) as "text-interpreted" vs "numbers-only" and compare accuracy per group. Expected: small models do worse on numbers-only.
- **"Maybe" class note.** PQA-L priors: yes 55.2 / no 33.8 / maybe 11.0. "Maybe" only survived when both annotators agreed it was ambiguous, so low maybe-F1 is partly a data property. Mention in README.
- **Reference ladder for README.** Majority ~55%, BioBERT multi-phase ~68%, single human (reasoning-required) ~78% (verify in paper Table 5). Frame as a reference ladder, not a leaderboard (different setup, possible contamination).
- **Static results page.** GitHub Pages or Vercel free tier. Results table + confusion matrix + 3 failure examples. No domain purchase.

## Phase 2

- **QLoRA fine-tune** on the 500 reserved PQA-L records (never seen in Phase 1), then before/after regression on the frozen harness.
- **Expanded QLoRA training (parked).** Explore PQA-A warm-up and confidence-filtered pseudo-labeling of PQA-U to extend training beyond the 500 reserved PQA-L records, adapting Jin et al. (2019), [Section 4.3 / Figure 5](https://arxiv.org/pdf/1909.06146#page=6); exclude official test PMIDs from every training and pseudo-labeling stage and keep evaluation on the frozen harness.

## Beyond Phase 2 (benchmark roadmap)

| Benchmark | Strength | For us |
|---|---|---|
| PubMedQA (2019) | yes/no/maybe QA from abstracts | Phase 1 |
| Evidence Inference 2.0 (2020) | Treatment comparison + supporting evidence | Good follow-up |
| MIRAGE / MedRAG (2024) | Retrieval + generation evaluation; includes PubMedQA* | Real RAG phase, harness carries over |
| HealthBench (2025) | Realistic health conversations, clinical safety | Too broad for now |

## Product ideas (separate project, not this diagnostic)

- **Language layer for a medical RAG product:** detect user language, run the model in its strongest language, translate input/output. Prevents Chinese-drift answers. Note: for this diagnostic, Ollama structured outputs already solve drift.
- **Fallback to a cheap API model** (masking + Groq or similar) if local models are too weak.
- **Data sourcing:** before any scraping idea, check the site's terms first. Market Compare was shelved because all three sources forbade automated collection.
