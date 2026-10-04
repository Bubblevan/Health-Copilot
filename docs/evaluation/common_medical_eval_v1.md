# Common Medical Evaluation V1

## Evaluation contract

All system arms use `HealthCopilotHarness.execute(profile, HarnessRequest)`. Gold remains in `EvalCase` and the dataset adapter. The runner creates a request from case ID, question, and answer schema only. Deterministic choice scoring counts parse failures as incorrect and does not call an LLM judge.

The alias registry in `configs/eval/profile_registry.json` defines B0-B3, P0-P3, and R0-R3 as orthogonal model/retrieval/memory/reasoning axes. Profiles do not implement alias-specific runtime behavior. Common medical MCQ profiles set Memory OFF. The product API uses the single `product-adaptive-v1` profile; static Single remains only as a paired evaluation control. See [the pruning decision](../architecture/architecture_pruning_decision.md) for the evidence behind this cut.

## Intended public evaluation sets

| Set | Intended size | Score | H0 state |
|---|---:|---|---|
| DiagnosisArena MCQ | 915 | Exact single-choice accuracy | Snapshot/revision/hash absent; not frozen |
| CMB-Exam full | 11,200 across 28 subcategories | Exact answer-set accuracy and category macro accuracy | Snapshot/revision/hash absent; not qualified |
| CMB-COMMON-1024 | 1,024 stratified public cases | Exact answer-set accuracy and category macro accuracy | IDs/hash cannot be materialized without the source snapshot |

These are `PUBLIC_EXTERNAL_EVAL`, not blind or untouched tests. The Common Eval config records that evaluation freeze preceded post-training as false until the actual snapshots are pinned.

## Deterministic subset protocol

`stratified_case_ids()` sorts category names and case IDs, allocates `floor(1024 / 28)` to each category and distributes the remainder in sorted-category order, then samples without replacement from a category-specific seed derived from the fixed seed. The selected ordered ID list is SHA-256 hashed. The resulting IDs and source snapshot hash must be committed to `configs/eval/cmb_common_1024.json` before training. The current manifest is explicitly `NOT_FROZEN_MISSING_SOURCE_SNAPSHOT` and has no case IDs.

## Runner and checkpointing

```powershell
uv run --project . python tools/eval/run_common_eval.py `
  --dataset diagnosisarena --profile B0 `
  --model-config configs/models/qwen3_8b_base.json `
  --output runs/common_eval/<run-id>
```

Every result is appended to `cases.jsonl` before the next case starts. Resume validates dataset/profile/model identity and every completed case ID. The record contains response and deterministic score, never gold. Traces are metadata-only JSONL. Summary metrics include exact accuracy, parse rate, category macro accuracy, provider calls, tokens, mean/P50/P95 latency, retrieval and MDT metrics where available.

The runner rejects missing or mismatched dataset hashes, incomplete model identity, unready Common KB for RAG profiles, and memory-enabled profiles without a bound provider. Once the KB is qualified, its config must also point to a local `module:function` provider factory that returns a `RetrievalProvider`; a READY flag alone will not create an implicit or richer corpus. H0 does not start the full matrix.

For the remote Qwen3-8B factorial ablation, the same entry point can run all four B profiles in order and emit paired deltas and the RAG×Adaptive interaction:

```powershell
uv run --project . python tools/eval/run_common_eval.py `
  --dataset cmb-common --matrix core `
  --model-config configs/models/qwen3_8b_base.json `
  --output runs/common_eval/<run-id> --resume
```

The matrix maps B0/B1/B2/B3 to Single, Single+RAG, AdaptiveMDT, and AdaptiveMDT+RAG. Each arm has its own checkpoint and trace directory under the output root. `matrix_summary.json` contains descriptive paired accuracy deltas and verifies B1/B3 retrieved-evidence hashes match case by case; per-arm summaries retain call, token, latency, retrieval, and adaptive-routing metrics. Memory stays outside this matrix. All dataset, model, and retrieval gates are checked before the first generation call.

## Common Medical KB gate

B1/B3, P1/P3, and R1/R3 are blocked until `configs/eval/common_medical_kb_v1.json` is fully qualified with source use status, document hashes, corpus/index hashes, BM25 and dense encoder configs, RRF settings, and top-k. The local closed hypertension cards and the R2MED-specific research corpus remain preserved; they do not by themselves qualify a broad Common Medical KB for the public exam matrix.

## H0 execution record

This checkout does not contain the public snapshots or frozen model identity required by these profiles. The matrix runner is wired, but this architecture change runs no benchmark cases and makes no accuracy or system-delta claim. On the remote host, first bind the supplied dataset snapshot/revision/hash and served model SHA, then qualify the shared RAG provider before invoking `--matrix core`.
