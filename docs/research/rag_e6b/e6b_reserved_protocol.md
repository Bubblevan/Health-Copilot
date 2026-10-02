# RAG-E6B Reserved Confirmation Protocol

## Question and estimand

E6B is a one-shot, project-owned synthetic longitudinal confirmation of the already frozen RSEL-v1 transform. It asks whether, with the same Qwen3-8B reader, the same pinned LameR-MV retrieval bridge, the same ranked top-10 and byte-identical evidence, deterministic Harness-side extraction of an exact visible `SYNKEY → SYNVAL` relation improves grounded task success over the paired Vanilla answer.

This is not method development. The RSEL regex, prompt, model, retriever, top-k, evidence order, fallback, and scorer semantics are frozen. An unfavorable IID result closes the headline; OOD is descriptive and cannot rescue IID failure.

## Lineage and method lock

The branch starts from the latest audited `origin/main`, then cherry-picks only the E6A/RSEL implementation, protocol, tests, and frozen artifacts. The RSEL method-critical files must byte-match `runs/rag_e6/frozen_dev/frozen_dev_manifest.json`. Before any reserved row is generated, `runs/rag_e6b/method_freeze.json` is written and committed. The materializer refuses to run unless that lock and every recorded source hash are committed and clean.

Frozen runtime identity:

- Qwen3-8B-Q4_K_M SHA-256: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- BGE-large revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`; weights SHA-256 `45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7`.
- LameR-MV upstream commit `11244a4925a39082967a6c9d38ef01f279c316a5`.
- Four retrieval channels: `BM25_QUERY`, `BM25_QUERY_PLUS_BRIDGE`, `BGE_QUERY`, `BGE_BRIDGE`; RRF `k=20`, weights `[1,2,1,2]`, top-10 evidence.
- Temperature `0`, top-p `1`, reasoning disabled, retry count `0`, response ceiling `512`; effective prompt context stays at the frozen server limit.

## Reserved pools and deterministic adapter

The only seed authority is `runs/integration/u2f-owned-v1-55955b2eff38/reserved_test_plan.json`, schema `u2f-reserved-pools-v1`, SHA-256 `db2c7bf4158374a09502a52e8ccb1671c43332aaec9e09913a9a31d0d70f7d1b`. The plan supplies seed ranges but not row counts. The thin adapter therefore applies the existing U2-F sibling construction: one scenario seed yields one two-episode sibling pair, and each persona seed denotes one subject. This implies 512 episodes/256 subjects for each of the first five pools and 1,024 episodes/512 subjects for `OOD_COMPOSITION`.

Other generator configuration is copied from the frozen U2-F source: unchanged scenario-family weights and TRAIN template family; the frozen TRAIN same-surface-pair fraction is carried forward using nearest-integer pair counts. The seed values themselves are used without offset or regeneration. All runtime rows are materialized; no capability, RSEL-match, or outcome filtering occurs.

The plan lists four `OOD_COMPOSITION` themes but provides no per-theme quotas or assignment function. The adapter does not invent quotas or modify generator semantics to force them. Those themes are retained in the materialization manifest and assessed after truth access using frozen structural metadata. Consequently, composition-specific conclusions are limited to structures actually generated and counted; the pool's name alone is not evidence that every named theme was present.

## Separation and paired execution

Materialization writes separate runtime-visible and evaluator-only trees. The reader runner accepts only runtime episode and runtime corpus paths; it has no truth path argument and never decodes evaluator-only files. Every reserved episode is executed before any capability slice is selected:

1. One LameR bridge call uses the original-query BM25 top-10.
2. Frozen LameR-MV produces the shared ranked top-10.
3. One Vanilla Qwen reader call creates the paired draft.
4. RSEL deterministically transforms that draft/evidence; it makes no model call.

Each call is journaled once, with no retry. Started-but-interrupted calls are not replayed. The call journal, reader output, and paired evidence identity are validated and committed before the scorer can open evaluator truth. The scorer uses a write-once score-start guard and refuses a second run.

## Post-freeze analyses

Primary population is `IID_TEST ∩ capability_requirement_oracle == RAG`, selected only after all executions are frozen. Primary outcome is grounded task success; the pre-registered gate requires `+10 pp`, paired subject-cluster-bootstrap 95% CI lower bound above zero, grounding degradation no worse than `-1 pp`, full-evidence utilization gain of `+10 pp`, and zero extra RSEL calls.

All five OOD pools are reported separately and pooled, always with subject-cluster paired bootstrap (10,000 resamples, seed `20261002`). OOD cannot replace the IID primary. RSEL match/no-match, `NONE`/`INSUFFICIENT`, and composition families are diagnostic. Exact no-match output/evidence parity with Vanilla is an implementation invariant.

No RSEL-solved structured relation cases are routed to post-training. Candidate future OPD/SFT cases are limited to RAG no-match rows where all required external evidence is in the shared top-10 but Vanilla fails. The current sprint only emits a candidate manifest; SFT, OPD, and GRPO are not started.

## Interpretation boundary

Even if the IID gate passes, the strongest supported statement is about a reserved, project-owned, synthetic structured-evidence test and exact explicit relation grammar. It does not establish natural clinical-prose generalization, public medical benchmark gains, improved retrieval, or a general clinical RAG architecture. OOD generalization is supported only if the pooled OOD CI is positive and no individual OOD pool has a negative point estimate.
