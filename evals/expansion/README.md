# Product evaluation expansion candidates

This directory is intentionally separate from frozen M0--M3 evaluation packs.
Rows here are source-anchored **candidates**, not released gold: every row has
`status: "candidate_review_required"` until a human reviewer confirms the
question wording, source mapping, scope boundary, and data-license record.

`m0_hypertension_candidates_v1.jsonl` is the first expansion batch. It adds
one non-duplicate Chinese query for every current product KnowledgeCard plus
multi-source and corpus-uncovered controls. It can be run as a diagnostic with
the existing M0 evaluator, but its metrics must not be reported as frozen M0
baseline results before promotion into a separately versioned reviewed pack.

External artifacts stay in `artifacts/benchmarks/` and remain separate:

- NFCorpus is an independent biomedical retrieval benchmark with its own qrels.
- MIRAGE and HealthBench require schema and license review before any subset is
  used in a product evaluation protocol.

No file in this directory changes M0, M1, M2, M3 policy decisions, capability
gold, or claim-support gold.
