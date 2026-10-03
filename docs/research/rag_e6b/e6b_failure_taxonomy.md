# RAG-E6B Failure Taxonomy

## Scope

This taxonomy is computed once from the committed E6B scored rows. Vanilla failure modes are counted only on RAG-required rows where RSEL succeeds; modes are non-exclusive, so counts can overlap. The evaluation uses synthetic structured evidence and the frozen exact `SYNKEY → SYNVAL` relation grammar.

## Vanilla failures repaired by RSEL

| Mode | Count | Interpretation |
|---|---:|---|
| `MULTI_VALUE_DROPOUT` | 192 | Vanilla omitted one or more required values on a multi-value query. |
| `GROUNDING_FAILURE` | 131 | The answer did not use/ground all required resources, despite those resources being retrievable. |
| `VALUE_OMISSION` | 2 | A required value was absent from the generated answer. |
| `CITATION_UTILIZATION_FAILURE` | 1 | Required evidence was available but no required evidence was used. |
| `OUTPUT_CONTRACT_FAILURE` | 1 | The reader did not produce a valid answer/citation contract. |

RSEL had no remaining failure-mode entries in the scored RAG rows. That statement is limited to the generated relation grammar and answerable RAG slice; it is not a claim of zero errors in real clinical language.

## Attribution: evidence retrieval versus evidence use

Every RAG-required row in each reported pool had all required external evidence in the shared top-10. Vanilla's IID external-evidence-use coverage was 92.81%, while RSEL used all required evidence in 100% of the slice. Grounded success increased from 70.59% to 100.00% on IID (102 episodes/60 subjects), and the paired subject-cluster 95% CI for the +29.41 pp delta was [+19.05, +40.78].

Because both arms receive identical retrieval identities and evidence bytes, the experiment does **not** attribute improvement to retrieval, ranking, or candidate recall. The observed failure cluster is downstream: a generative reader can omit or fail to cite one of several already-retrieved facts; a deterministic ledger avoids that failure when the exact relation is present.

## Match, fallback, and execution interruption

- Across all reserved rows, 1,585 matched the exact relation grammar; 1,999 used no-match fallback.
- No-match outputs preserve the paired Vanilla answer and evidence exactly; the invariant passed and the no-match grounded-success delta was 0 pp.
- One call, `U2F-ED426E4F26BD972A|VANILLA_STRONG|reader`, was started but interrupted before a completion record. The runner resumed from checkpoint and honored the call journal's `interrupted_no_retry` rule. It was not submitted again. The episode is OOD_COMPOSITION/RAG/`EXTERNAL_MULTI_SOURCE`; both required external facts were in the shared top-10. Its recorded Vanilla output is empty and fails the answer contract, while RSEL's matched relation ledger succeeds.
- This one interrupted call does not affect the IID result. It can raise the pooled OOD point delta by at most 0.146 pp under the hypothetical that Vanilla would otherwise have succeeded; this is a point-only sensitivity, not a new score or bootstrap interval.

## Sanity slices outside the headline

| Capability slice | Episodes | Vanilla grounded success | RSEL grounded success | Readout |
|---|---:|---:|---:|---|
| `NONE` | 678 | 69.32% | 69.32% | RSEL always falls back; no method delta. |
| `INSUFFICIENT` | 399 | 0.00% | 0.00% | Neither arm solved the sufficiency/abstention task. RSEL is not an abstention policy. |

The second row is a meaningful limit, not a result to hide: the method only addresses explicit, answerable key/value relations. A future system needs a separately evaluated evidence-sufficiency/abstention mechanism, with deterministic Harness ownership of the boundary.

## Composition coverage limit

The reserved plan named four OOD composition themes but defined no per-theme quotas. The frozen adapter did not manipulate generator weights to force them. Post-score structure counts found 52 deep `MEMORY→RAG` serial-chain episodes and 98 three-branch compositional episodes; `multi-source RAG plus memory revision` and `long-history plus versioned external evidence` had zero instances. Do not claim coverage or gains for those absent themes.

## Practical conclusion

The strongest supported diagnosis is: **retrieval saturated on this synthetic RAG slice; the residual Vanilla failures were evidence-use/composition errors that an exact deterministic relation ledger repairs.** The narrow grammar, 100% ceiling, no-match behavior, one OOD interruption, and the `INSUFFICIENT` result constrain the claim. No public-benchmark or natural-clinical-language inference follows.
