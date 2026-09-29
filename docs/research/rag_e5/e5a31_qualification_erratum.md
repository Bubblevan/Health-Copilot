# E5-A3.1 Qualification Semantics Erratum

- Date: 2026-09-29
- Stage: E5-A3.1
- Base A3 commit: `40c3fddf7994f8ec6401303b97a603b9e479ba45`

## Protocol correction

E5-A3's implementation behaved as specified. The A3 protocol stopped corpus
activation because it made dense source-title `hits@10 == 10/10` a hard
index-ready condition. Post-run review found that this condition conflated two
different properties:

- **Index integrity:** corpus/document identity, complete embedding rows,
  ordering, vector validity, index binding, and deterministic exact search.
- **Retrieval diagnostic:** whether a particular model ranks expected sources
  highly for a small set of source-title probes.

The latter is not an invariant of a correctly built index. The protocol now
uses `rag-e5-index-qualification-v2`: corpus integrity and BM25/dense structural
integrity gate activation; retrieval quality remains a separately reported
diagnostic and is measured in the downstream integration experiment.

This is a disclosed protocol correction after a diagnostic blocker, not a
silent threshold relaxation. No threshold was lowered: semantic top-10 recall
was removed from the definition of structural index integrity.

## Protocol identities

- Previous protocol SHA-256: `3a4e4f3c934f704bef900fdf6b7ae354ff1a4e409b0eb4cff1b679dcfb91d0ad`
- Corrected protocol SHA-256: `49d6edcba8ffc3e4d598b0b0e4f983cbd5f5a060621494c755ad9d88a1502b65`
- Previous qualification semantics: `rag-e5-index-qualification-v1`
- Corrected qualification semantics: `rag-e5-index-qualification-v2`

## Preserved A3 observations

No retrieval output was modified. The frozen title diagnostic remains:

| View | BM25 | Frozen BGE |
| --- | ---: | ---: |
| `PUBLIC_HEALTH_ONLY` | 10/10 | 8/10 |
| `GUIDELINE_ONLY` | 10/10 | 10/10 |
| `PUBLIC_HEALTH_PLUS_GUIDELINE` | 10/10 | 10/10 |

The two public-health dense misses remain `public-health-08` at rank 18 and
`public-health-10` at rank 15. The combined smoke sample does not contain those
queries. They are recorded as `DENSE_RETRIEVAL_DIAGNOSTIC_MISS`, plausibly
contributed to by the frozen English BGE model on Chinese text, with confidence
`PLAUSIBLE_NOT_CAUSALLY_PROVEN`. This limitation is not evidence of a tokenizer,
encoding, corpus, or index-build defect and was not “fixed” by changing the
model, query, cutoff, chunking, or ranking.

## Integrity and promotion

The independent structural audit found the candidate corpus and all three
BM25/dense views intact. The 30 reviewed public-health cards, three owner-
approved guideline sources, 22/22 canonical recommendation sections, 26
guideline chunks, and 56 combined chunks retained their frozen identities.
Every dense row is finite, 1024-dimensional, normalized within `1e-5`, and
nonzero; stored BM25 artifacts bind the exact document IDs and corpus views;
structural search probes return valid IDs with finite scores and deterministic
order. No corpus or index content was rewritten.

The owner's approval is linked to the owner-authored conversation decision
(`“我已经决定好了，和subagent意见一致”`) and the hash-bound record at
[`e5a3_owner_decision.json`](e5a3_owner_decision.json). The approved scopes are
unchanged: physical activity and total-fat guidance are task-authoring eligible;
hypertension pharmacology remains retrieval-context/distractor only.

The unchanged candidate identity
`9b19ad467f47641032277707cb1cfdb1d05c2fb39c558039b180efc7394692bd` was promoted
to active. STANDARD and STRONG were bound to that identity without changing
their frozen config hashes. This establishes E5-A corpus/index readiness only;
it does not claim that retrieval improves end-to-end task outcomes. E5-B was
not started.

Machine-readable evidence:

- [`structural integrity`](../../../runs/rag_e5/e5a31_dense_integrity_report.json)
- [`frozen retrieval diagnostic`](../../../runs/rag_e5/e5a31_retrieval_diagnostic.json)
- [`language inventory`](../../../runs/rag_e5/external_language_inventory.json)
- [`final activation`](../../../runs/rag_e5/e5a_final_activation_report.json)
