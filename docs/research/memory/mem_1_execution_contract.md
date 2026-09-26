# MEM-1 Execution Contract

**Scope:** MEM-1 controlled reproduction of five pinned MemEval baselines only. This document supplements, and does not rewrite, the frozen MEM-0 protocol.

## Frozen boundary

- Dataset input is LongMemEval-S at the revision and SHA in `split_manifest.json`.
- MEM-1 may read the 102 frozen DEV questions only. The 398-question TEST complement and its gold scorer are forbidden until method/configuration lock.
- Systems are FullContext, OpenClaw, Mem0 OSS, SimpleMem and PropMem. No Health-Copilot M10-Flat/RevMem, FAMA, Multi-Agent or training work belongs in this stage.
- Upstream MemEval stays pinned at `807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4`; Health-Copilot-specific changes are carried only by `tools/research/memory/patches/memeval_qwen_main_v1.patch`.

## Roles and provider routing

Every run manifest must contain four distinct role entries:

| Role | Main Track | Upstream-parity sanity |
|---|---|---|
| Reader / answer model | Local Qwen3-8B Q4_K_M, protocol alias `health-memory-qwen3-8b`; temperature 0, seed 42, thinking disabled, output cap 64 | GPT-4.1 with the pinned upstream sampling settings |
| Memory system | The baseline under test: ingestion/materialization, storage, retrieval and context construction | Same five pinned systems |
| Embedding model | OpenAI `text-embedding-3-small`, 1536 dimensions | Same |
| Judge model | OpenAI `gpt-4o`, temperature 0, max 10 output tokens | Same |

The deterministic token-F1 scorer is local code, not a model role. Main reader traffic must use an explicitly configured loopback endpoint. Embedding and judge clients must use their own explicit cloud endpoints and may never inherit the reader URL. A process-wide `OPENAI_BASE_URL` is not a routing control.

## Artifact and resume rules

Each run directory follows this layout:

```text
runs/memory/mem1/<run_id>/
├── run_manifest.json
├── predictions.jsonl
├── predictions.sha256
├── call_ledger.jsonl
├── embeddings_usage.json
├── token_efficiency.json
├── failures.jsonl
├── deterministic_metrics.json
├── judge_results.jsonl
├── judge_metrics.json
└── report.md
```

Generation is completed and `predictions.jsonl` is hashed before deterministic scoring or any judge calls. The prediction sidecar must verify before scoring. Judge outputs are stored separately and never mutate predictions.

Checkpoint at question granularity. A prediction cache identity includes system, question ID, dataset SHA, system-config hash, hashes for all relevant prompts, reader artifact SHA, embedding model, and patch/code hash. Only an `OK` prediction with an exact, internally valid identity can be reused. Infra failures and judge results are not valid prediction cache entries. Any identity change is a cache miss.

Run manifests are immutable after creation, must mark `split=DEV` and `test_access=false`, must keep the four model/system roles separate, and must contain no credentials. Ledger rows contain role, provider, model, token usage where available, latency, retry count, success and truncation status, but never request bodies or keys. Network errors are logged as infrastructure failures with error type and null token usage.

`tools/research/memory/run_mem1.py` is the controlled entrypoint. Generation accepts only question IDs present in frozen DEV or a selection manifest with the frozen dataset SHA; it validates the local dataset bytes and pinned MemEval patch. It checkpoints one LongMemEval question at a time using the cache identity above. Invoke `--stage generate` to produce and freeze predictions; a separate `--stage judge` verifies the frozen prediction SHA before calling GPT-4o. The judge stage cannot mutate prediction artifacts.

## Failure and denominator semantics

- Reader, ingestion, embedding, schema, timeout and library failures are `INFRA_FAILURE`, have null prediction/F1, and are excluded from the quality denominator. They are written to `failures.jsonl`; they are never converted to an empty answer or a zero score.
- GPT-4o judge errors retain deterministic F1, use `judge_status=ERROR` and `longmemeval_correct=null`, retry at most three times, and are excluded from the judge denominator.
- Report both quality and infrastructure counts. A missing judge result is not an incorrect answer.
- For FullContext, every query must carry tokenizer prompt-token count, `max_model_length=131072`, output reserve and an explicit server truncation result. Missing or true truncation status invalidates that run item; it must not be represented as `truncated=false` by assumption.

## Staged execution and stop points

1. **MEM-1A offline readiness:** patch audit, provider boundary, compatibility/dependency audit, failure semantics, telemetry, immutable run manifest, cache/resume and fake-provider tests. No paid API calls.
2. **Cost preflight:** project local GPU work, GPT-4o judging and embedding API separately, including ingestion model calls. Recheck current prices immediately before the first paid request. Do not start 10- or 102-case runs at this point.
3. **MEM-1B one-case Main smoke:** only DEV question `1cea1afa`; run OpenClaw, FullContext, PropMem, Mem0 OSS, SimpleMem in that order, first without judging. Validate routing, full-context length/truncation, failure handling and resume, then judge the five frozen answers. Stop for review.
4. **MEM-1C upstream-parity sanity:** six frozen DEV IDs, five systems, GPT-4.1 reader and GPT-4o judge. Report as directional adapter sanity only; never tune Main Track from it. Stop for review.
5. **MEM-1D diagnostic smoke:** the frozen `main_smoke_10_manifest.json`, all five systems with Qwen3-8B. Scores are diagnostic only. Stop for review before 102 DEV.
6. **MEM-1E full DEV:** all five systems × 102 DEV. Freeze all predictions, calculate token F1, then run native GPT-4o judge. Never access TEST during MEM-1.

At each stage, retain per-question prediction/context artifacts and append-only call/failure ledgers. No later stage begins automatically after a stop point.

## Cost accounting

Report local Qwen GPU wall time/compute separately from cloud charges. Keep GPT-4o judge input/output and `text-embedding-3-small` input-token usage in separate buckets. Count all memory-ingestion and reasoning calls, not only final reader answers. The withdrawn GPT-4.1 Main Track token-cost estimate is not applicable. No fixed embedding dollar estimate is claimed until measured token receipts exist.

## MEM-1A gate status

`MEM1_EXECUTION_READY=YES` is allowed only after the pinned patch clean-applies, offline tests pass, role routing and failure semantics are exercised with fake providers, artifact identity/freeze and DEV-only runner tests pass, and the exact benchmark dependency environment is available. A lockfile resolving successfully is not proof that its packages are installed or importable.
