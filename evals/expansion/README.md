# Product evaluation expansion candidates

This directory is intentionally separate from frozen M0--M3 evaluation packs.
Rows here are source-anchored **candidates**, not released gold: every row has
`status: "candidate_review_required"` until a human reviewer confirms the
question wording, source mapping, scope boundary, and data-license record.

`m0_hypertension_candidates_v1.jsonl` is the first expansion batch. It adds
one non-duplicate Chinese query for every current product KnowledgeCard plus
multi-source and corpus-uncovered controls. The source rows retain their
candidate state for provenance. A separately versioned review manifest freezes
the exact file SHA256 and records an approval; only a manifest-approved hash is
an expansion release. Its metrics remain separate from frozen M0 baseline
results.

`m0_hypertension_expansion_v1.review.json` records the user-approved review of
the current 40-row source file. It is an evaluation-data review, not a clinical
expert validation claim.

M3 expansion files use the same candidate/review-manifest workflow. They do
not alter frozen `m3_capability.jsonl` or `m3_claim_support.jsonl`.

`m3_capability_expansion_v1.review.json` and
`m3_claim_support_expansion_v1.review.json` record the approved hashes for
the first M3 expansion release. The approved files remain physically separate
from the frozen packs, so their live metrics are reported as expansion results
and never overwrite the original M3 closeout.

External artifacts stay in `artifacts/benchmarks/` and remain separate:

- NFCorpus is an independent biomedical retrieval benchmark with its own qrels.
- MIRAGE and HealthBench require schema and license review before any subset is
  used in a product evaluation protocol.

No file in this directory changes M0, M1, M2, M3 policy decisions, capability
gold, or claim-support gold.
