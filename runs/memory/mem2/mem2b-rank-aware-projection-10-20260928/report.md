# MEM-2B — Rank-Aware Projection Frozen-10 Diagnostic

Gate: `MEM2B_RANK_AWARE_PROJECTION_FROZEN_10_DIAGNOSTIC=YES`

## Protocol

- Single intervention: budgeted ContextManager projection order for the exact frozen M10-Base native top-8.
- No re-ingestion, no `MemoryStore.matches()`, no external baseline reruns, no embeddings, no judge, no hosted calls.
- Frozen reader contract: `57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3`.
- Reader calls: 10/10; all local loopback Qwen3-8B.
- Test access: false; 102 DEV not run.

## Rank Preservation

- Selected-worse-while-better-dropped pairs: 56 → 15 (9/10 → 6/10 affected cases).
- Reader-context order inversions: 11 → 0 (6/10 → 0/10 affected cases).
- Mean top-1 survival: 0.8
- Mean top-3 survival: 0.4666666666666667
- Mean top-5 survival: 0.38

The two inversion counts are distinct. Rank-aware selection makes the order of selected memories monotonic by retrieval rank. The greedy skip rule can still select a later small record after skipping a larger better-ranked record; that is not an item-ID ordering inversion.

## Evidence And Reader Diagnostics

- Mean projected answer-session Recall@5: 0.33999999999999997 (MEM-2A 0.24).
- Mean projected answer-session Recall@8: 0.33999999999999997 (MEM-2A 0.24).
- Mean projected-context MRR: 0.5 (MEM-2A 0.2333333333333333).
- Mean normalized gold-token coverage: 0.39507087115782774 (MEM-2A 0.26412963847746457).
- Mean reader-visible context tokens: 990.5.
- Deterministic token F1 / normalized EM: 0.12453248505880084 / 0.0 (descriptive only).

Answer-session recall is provenance-level retrieval coverage, not answer-bearing evidence recall. Token overlap and exact normalized gold-sequence presence are separate diagnostics. No performance ranking or method-selection claim is made from DEV answer scores.

## Case Review

Human failure-locus and outcome fields remain null. The next-stage failure-surface decision is left for Reflection review.

## Frozen Artifacts

- `call_ledger.jsonl`: `b5ba6ace549fc2e12bac8920ccb18be24c4d37d8b687d1f9fcea8b06d5ee11f7`
- `comparison_mem2a_vs_mem2b.json`: `789150f2ee448623a4cd00bd08b54e2268edc16e3c067a5ff989ab9e2069fb77`
- `context_bundles.jsonl`: `04ecf8f537368e52bc6de93617c4827b804189cf9f73fc504207be53ee0da31e`
- `context_plans.jsonl`: `7829149cf71cb75f4f6951f01e41392eff66d1ce6801029371d2ff3990d4e8df`
- `deterministic_metrics.json`: `4c08da66ebe70e6830d15de82f1b10ca73559ca9e4e11686b4ae289921d29ce7`
- `mem_2b_case_review.json`: `a966dca4a974301ca328bc195657803da71ab1cc229a3c6799a96ec9e1b14393`
- `mem_2b_rank_aware_projection.md`: `215bf66dfd9bac5cff1927783f7d2ccccd9db772579ceab7cc0856ddce0b3399`
- `predictions.jsonl`: `de5d8f637fef278c94fdbcf4d655a9ac69e65734730b686eb9bd2b4bc1037201`
- `projection_artifact_revision_audit.json`: `cf23df0486913f27be67b99a746c1589b6d5f3219f81a4b63015940f8896a891`
- `projection_diagnostics.json`: `6dafb7c51dba5e45b376eb184fb4919f5c14cb44bf7a6557005862922d7d54fb`
- `projection_diagnostics_pre_reader.json`: `a36be3ec78283a7472165d4ed07743d0cf12d89408939faf9d85eac94a1a0c7a`
- `projection_frozen_manifest.json`: `3c5ed255fb88aa14dbdf4436064dd97d80a33e631471b8ed731fff6168a5ec10`
- `rank_aware_projection_contract.json`: `ee16902373d695307db42797f33a1f4a531484096b36c29639b98abfd58d52ed`
- `retrieval_parity.json`: `537bdb5354f23cff386934706f8d109dde29fe19b57f49c70ee5038f3775b30b`

This closeout stops at the frozen-10 counterfactual. It does not run 102 DEV, TEST, MEM-2C, M10-Flat, or RevMem.
