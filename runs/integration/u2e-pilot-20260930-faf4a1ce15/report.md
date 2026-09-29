# U2-E Owned Universe Generator Qualification Pilot

- Run ID: `20260930-faf4a1ce15`
- Date: `2026-09-30`
- Status: `QUALIFIED_PILOT` only when every listed gate passes
- Branch: `integration-u2d-dataset-split-license-20260929`
- Base commit: `72d88b241c72f25f8e0f9fca85cb29e68be074be`
- Final commit SHA: resolved from the pushed repository commit at closeout
- Remote SHA: verified against the branch after push at closeout
- Generator version: `u2e-generator-v1.0.1`
- Specification hash: `faf4a1ce15109042a4eb905c331c29f3b995a757756ad19c22526a1b80249017`
- Seed policy: project-frozen disjoint per-split ranges; global seed permutes only within each declared range
- Deterministic regeneration: verified by two isolated runs with byte-identical artifact hashes
- Notice: `SYNTHETIC_RESEARCH_WORLD; NOT_CLINICAL_GUIDANCE`

## Pilot size and splits

- TRAIN_PILOT: 144 episodes, 24 subjects
- DEV_IID: 36 episodes, 6 subjects
- DEV_STRUCTURAL: 36 episodes, 6 subjects
- Total: 216 episodes, 36 disjoint synthetic subjects
- TRAINING_AUTHORIZED: `NO`
- Reserved TEST/OOD pools: manifests only; no rows materialized

## Scenario and capability distribution

- Scenario families: `{"COMPOSITIONAL_MULTI_FACT": 8, "CURRENT_ONLY": 14, "DISTRACTOR_HEAVY": 8, "EXTERNAL_LOOKUP": 17, "EXTERNAL_MULTI_SOURCE": 12, "EXTERNAL_VERSIONED": 26, "INSUFFICIENT_EVIDENCE": 14, "MEMORY_EXTERNAL_CONFLICT": 8, "MEMORY_EXTERNAL_JOIN": 12, "MEMORY_LOOKUP": 23, "MEMORY_MULTI_RECORD": 14, "MEMORY_REVISION": 26, "MEMORY_TEMPORAL_COMPARE": 14, "TEMPORAL_BOUNDARY": 20}`
- Derived capability requirements: `{"INSUFFICIENT": 17, "MEMORY": 94, "MEMORY+RAG": 28, "NONE": 22, "RAG": 55}`
- Architecture supervision: `UNRESOLVED`; architecture labels present: `NO`
- Budget-class supervision: `UNRESOLVED`
- Matched same-surface/different-requirement pairs: 15
- Different-surface/same-latent-graph pairs: 65
- Simple-feature shortcut gate: `PASS` (maximum single-feature accuracy 0.676; threshold 0.95)

## Qualification gates

- Split/sibling leakage: `PASS`
- Temporal and revision semantics: `PASS` (72/72 checks)
- Latent-to-U1.1 counterfactual contract consistency: `PASS` (1728 arms; 0 mismatches)
- Source-independence/import/path audit: `PASS`
- Latent graph ↔ natural-language realization separation: `YES`; evaluation truth is read from latent Layer A only
- Generated-text identifier scan: `PASS`
- Provider calls: `0`; hosted/local LLM generation: `NO`
- Benchmark rows, external medical documents, and gated records used: `NO`
- Real patient data / PHI used: `NO`
- Memory/RAG implementation changed: `NO`
- E2-B/L4 started: `NO`

## U2-D terminology clarification

- U2-D historical decision remains intact: `True`
- WHO/CDC runtime role: `EXTERNAL_RETRIEVAL_RUNTIME`; training: `NO`
- `OWNER_ACCEPTED_SCOPE` does not assert upstream license conflicts were resolved: `True`
- U2-E generated evidence origin: `PROJECT_OWNED_SYNTHETIC`

## Interpretation and routing

This qualifies a deterministic, project-owned synthetic research environment. It does not qualify clinical guidance, synthetic benchmark quality, model performance, or Team advantage. Counterfactual Team arms are contract smoke only. See `counterfactual_contract_report.json`, `split_audit.json`, `shortcut_audit.json`, `matched_pair_diagnostics.json`, `distribution_report.json`, and `source_independence_audit.json` for evidence.

## Next stage

Choose `U2-F — Materialize full owned TRAIN/DEV universe`. The project-owned generator passed split, shortcut, temporal, and counterfactual consistency gates; U2-F can use this frozen specification and keep TRAIN/DEV separate from reserved OOD pools. U2-F is **not** started in this task.
