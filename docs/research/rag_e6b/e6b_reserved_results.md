# RAG-E6B Reserved TEST/OOD Results

## Decision

The frozen RSEL-v1 method passed the pre-registered IID gate and the OOD support rule. The supported claim is deliberately narrow: on a **project-owned synthetic longitudinal test with explicit structured `SYNKEY → SYNVAL` evidence**, a deterministic, provenance-preserving Harness-side relation ledger improved grounded success over a paired single-pass Vanilla reader while using identical retrieval outputs and no additional RSEL model calls.

This is not a public medical benchmark result, a natural clinical-prose result, or evidence of improved retrieval. The experiment does not establish clinical correctness or general clinical-RAG capability.

## Frozen experiment

- Six reserved pools: `IID_TEST` (512 episodes/256 subjects), four 512/256 OOD pools, and `OOD_COMPOSITION` (1,024/512); 3,584 episodes and 1,792 subjects total.
- Primary population: `IID_TEST` rows whose post-freeze capability oracle labeled them RAG-required: 102 episodes from 60 subjects. No capability or outcome filtering was applied before execution.
- Paired arms: `VANILLA_STRONG` and `RSEL_STRONG` received the same query, pinned LameR-MV retrieval run, ranked top-10, and byte-identical evidence. Vanilla used the Qwen3-8B reader. RSEL deterministically extracts an exact requested key/value relation from visible evidence and preserves the source alias; if no relation matches, it returns the exact paired Vanilla answer/evidence.
- Retrieval: `BM25_QUERY`, `BM25_QUERY_PLUS_BRIDGE`, `BGE_QUERY`, and `BGE_BRIDGE`, fused with RRF `k=20`, weights `[1,2,1,2]`, top-10 evidence. This was frozen and shared by both arms.
- Generator: Qwen3-8B-Q4_K_M, temperature 0, top-p 1, reasoning disabled, 512-token response ceiling; BGE-large and the LameR-MV upstream identity are pinned in [the protocol](e6b_reserved_protocol.md).
- Uncertainty: paired subject-cluster bootstrap, 10,000 resamples, seed `20261002`, percentile 95% CI.

The test data are generated synthetic episodes and synthetic external evidence, not patient records. Each episode has one LameR bridge call and one Vanilla reader call; RSEL adds zero model calls.

## Primary and OOD results

`Grounded task success` requires the expected answer values, required resources used, valid execution/output contracts, and evidence provenance/grounding. For every RAG-required row in each slice below, both paired arms' shared top-10 contained all required external evidence. This makes the observed contrast an evidence-use/composition result, not a candidate-recall gain.

| Pool | Reserved episodes / subjects | RAG episodes / subjects | Vanilla grounded success | RSEL grounded success | Δ pp (95% CI) |
|---|---:|---:|---:|---:|---:|
| IID_TEST | 512 / 256 | 102 / 60 | 70.59% | 100.00% | **+29.41** [+19.05, +40.78] |
| OOD_PATIENT | 512 / 256 | 132 / 77 | 75.76% | 100.00% | +24.24 [+15.56, +33.33] |
| OOD_TASK | 512 / 256 | 92 / 55 | 76.09% | 100.00% | +23.91 [+13.68, +35.23] |
| OOD_TEMPORAL | 512 / 256 | 125 / 70 | 76.80% | 100.00% | +23.20 [+13.93, +33.33] |
| OOD_SOURCE | 512 / 256 | 98 / 55 | 72.45% | 100.00% | +27.55 [+17.17, +38.61] |
| OOD_COMPOSITION | 1,024 / 512 | 237 / 130 | 77.22% | 100.00% | +22.78 [+16.25, +29.49] |
| Pooled OOD RAG slice | 3,072 / 1,536 | 684 / 387 | 76.02% | 100.00% | **+23.98** [+20.06, +28.03] |

The IID gate passed all five pre-registered components: grounded-success gain at least 10 pp; paired CI lower bound above zero; grounding did not degrade by 1 pp or more; full-evidence utilization gain at least 10 pp; and zero extra RSEL calls. OOD support also passed: pooled OOD CI is above zero and every individual OOD pool has a nonnegative (here positive) point estimate. OOD is secondary and did not replace the IID primary.

On IID RAG rows, Vanilla grounding pass was 84.31% versus 100% for RSEL (Δ +15.69 pp; 95% CI [+7.77, +25.00]). Vanilla used 92.81% of required external evidence on average versus 100% for RSEL. Provenance, retrieval-contract, and output-contract pass rates were 100% for both arms on the IID RAG slice. The shared candidate evidence was complete in all 102 IID rows; the missed lift is therefore downstream of retrieval.

## Mechanism diagnostics

- Across all 3,584 rows, the ledger matched in 1,585 and used `FALLBACK_NO_MATCH` in 1,999. The no-match slice had exactly zero success delta and exact answer/evidence parity; the invariant passed.
- In the RAG-only slices, ledger matches were 91/102 IID and 588/684 pooled OOD. The no-match path is intentionally not a second answer generator.
- Across all capability classes, the `RSEL_MATCH` slice had +12.24 pp grounded-success point delta; `RSEL_NO_MATCH` had 0.00 pp. These are diagnostic, not the primary estimand.
- The deterministic operation is exact and intentionally small: extract requested `SYNKEY-XXXXXXXX` tokens, scan only the issued top-10 for exact `SYNKEY-XXXXXXXX to SYNVAL-XXXXXXXXXX` relations, retain document IDs/aliases, and emit the matched values plus citations. It does not infer missing relations or use evaluator labels.

## Execution integrity and the one interrupted call

The execution freeze passed validation and was committed before scoring. The score-start guard records `truth_opened_before_freeze=false`; the score report records `truth_opened_after_execution_freeze=true`, `method_modified_after_freeze=false`, and `one_shot=true`. The scorer verified the committed truth-shard hashes and was run exactly once.

Execution totals: 3,584 episodes, 7,168 generation-call records, 7,168 provider-call accounting units, zero truncations, and one failed call. One Vanilla reader call for `U2F-ED426E4F26BD972A` (OOD_COMPOSITION, RAG, `EXTERNAL_MULTI_SOURCE`) had a `started` record but no completion when the runner stopped. The frozen journal's no-retry rule marked it `interrupted_no_retry`; the call was not repeated. Its RAG candidate top-10 contained both required external documents. Vanilla therefore has a recorded empty-output failure on this one row while the deterministic matched RSEL arm succeeds.

This single OOD row does not affect the IID primary. As a simple worst-direction point sensitivity only, if Vanilla had succeeded on this row, the pooled OOD point delta would be lower by `1/684 = 0.146 pp` (about +23.83 pp instead of +23.98 pp). The registered bootstrap interval remains the one-shot observed interval; no counterfactual CI was recomputed.

## Negative controls, limits, and resume wording

- `NONE`: 678 episodes; both arms had 69.32% grounded success, and RSEL always fell back to Vanilla.
- `INSUFFICIENT`: 399 episodes; both arms had 0% grounded task success. RSEL is not an abstention or evidence-sufficiency policy; this is an important system limitation.
- `OOD_COMPOSITION` had 52 deep `MEMORY→RAG` serial-chain episodes and 98 three-branch compositional episodes. The two other predeclared structural themes had zero generated examples. Composition-specific conclusions are limited to observed structures.
- The candidate manifest contains **zero** rows under the frozen rule `RAG + no relation match + all required evidence in shared top-10 + Vanilla grounded failure`. No SFT, OPD, or GRPO training was started.
- The exact synthetic grammar is unusually favorable to deterministic extraction. These results do not show gains on natural biomedical prose, public clinical QA, R2MED, PubHealthBench, or retrieval metrics; they do not justify claiming RSEL beats a public medical-RAG baseline.

Defensible résumé wording, only with the synthetic scope explicit:

> Built a provenance-preserving deterministic evidence composer for a longitudinal RAG harness; on a frozen project-owned synthetic reserved test, improved grounded success from 70.6% to 100.0% (+29.4 pp; 95% subject-cluster bootstrap CI [+19.0, +40.8]) across 102 IID RAG-required episodes, with shared retrieval/top-10 evidence and zero additional method calls; observed +24.0 pp on pooled synthetic OOD RAG slices.

Do not shorten this to “improved clinical RAG by 29 pp” or “improved retrieval.”

## Frozen artifact references

- Protocol: [e6b_reserved_protocol.md](e6b_reserved_protocol.md)
- One-shot score report: `runs/rag_e6b/reserved_score_report.json`
- Pool/paired bootstrap: `runs/rag_e6b/bootstrap_results.json`
- Failure taxonomy: `runs/rag_e6b/failure_taxonomy.json`
- Post-training selection: `runs/rag_e6b/post_training_candidate_manifest.json`
- Scored rows: `runs/rag_e6b/reserved_scored_episodes.jsonl`
- Score artifact hashes: `runs/rag_e6b/score_artifact_hashes.json`
- Method freeze: commit `2bbf5d475f4421d124e32d8e1a6b574f2020bbcc`
- Materialization freeze: commit `ecf916081337714c3c48b54a251679ce21f5a456`
- Execution freeze: commit `60b2cc809bdbcd207a77d9e81b9a9ab2630b098a`
