# MEM-1B Local One-Case Smoke

> Historical R7 adapter record. This run used the MemEval-pinned PyPI SimpleMem artifact, not official `aiming-lab/SimpleMem` v0.1.0. Its measurements remain valid for that exact run but do not represent the State A primary SimpleMem comparator after MEM-1D0.5. See `mem_1d0_5_simplemem_provenance_correction.md`.

**Gate:** `MEM1_LOCAL_ONE_CASE_SMOKE=YES`

**Run:** `runs/memory/mem1/local-one-case-20260927-r7`

**Scope:** one frozen LongMemEval-S DEV question (`1cea1afa`), five baseline systems. This is an adapter/runtime smoke only; it is not a baseline reproduction result and does not set `BASELINE_REPRODUCTION=YES`.

## Protocol

The run is a controlled re-evaluation of public memory architectures under a unified fully-local model stack. It does not claim exact MemEval reproduction, official LongMemEval GPT-4o accuracy reproduction, or comparability with MemEval README GPT-4.1 values.

| Role | Frozen configuration |
|---|---|
| Reader / answer model | Qwen3-8B Q4_K_M, alias `health-memory-qwen3-8b`, SHA256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`; local llama.cpp loopback `127.0.0.1:8081`; shared 256-token answer cap, temperature 0, seed 42, thinking disabled |
| Memory-internal LLM | Same local Qwen3-8B artifact; default 8192-token budget and explicit upstream budgets retained |
| Memory systems | FullContext, OpenClaw, Mem0 OSS, SimpleMem, PropMem |
| Embedding | Local `Qwen/Qwen3-Embedding-0.6B`, Transformers CPU adapter, dimension 1024, L2-normalized; model-tree SHA256 `9d2d790d6448ef2c0911ffeb03f959d035c71ac3d2b14b7d586f2d2b39fb0efa`; weights SHA256 `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| Judge | NONE |
| Hosted API / API key | NONE / NONE |

The reader server used 131,072 context with YaRN scale 4 and `n_gpu_layers=99`. FullContext's exact llama.cpp-tokenized prompt was 108,168 tokens; the reader call recorded `max_model_length=131072` and `truncated=false`. No judge calls occurred. The main-track process ran without `OPENAI_API_KEY`; `OPENAI_BASE_URL` was an unusable loopback sentinel, and the local provider blocks non-loopback requests.

The upstream MemEval checkout was pinned at `807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4`. Health-Copilot adapter patch SHA256: `34c2536a936139b2f0c62cb0520f3fd8b302f58a7da1294685187bc9d99c0815`; patched prompt-source tree SHA256: `c883ec78417224aa326d123817b8ce46a2f994d1eed3abe7bb85cc1ca3144240`.

## Results

Question category: `knowledge-update` (n=1). Deterministic token F1 is the only quality metric; no LLM judge was used.

| System | Status | Prediction | Token F1 |
|---|---|---|---:|
| FullContext | OK | You currently have 600 Instagram followers. | 0.2857 |
| OpenClaw | OK | None. | 0.0000 |
| Mem0 OSS | OK | None | 0.0000 |
| SimpleMem | OK | 500 followers as of 27 May 2023 | 0.0000 |
| PropMem | OK | 600 followers | 0.6667 |

These are single-question smoke outputs, not evidence of comparative performance. The question has two answer-session IDs. Retrieval diagnostics were: OpenClaw Recall@5/10 `0.0/0.0`, MRR `0.0`; PropMem Recall@5/10 `0.5/1.0`, MRR `1.0`. Mem0 and SimpleMem did not expose compatible answer-session provenance for this query, so those metrics are null rather than zero. FullContext has no retrieval stage.

| System | Retrieved context tokens | Reader prompt tokens | Retrieval ms | Ingestion ms |
|---|---:|---:|---:|---:|
| FullContext | 108,168 | 108,168 | - | - |
| OpenClaw | 7,113 | 7,168 | 364 | 321,353 |
| Mem0 OSS | 237 | 289 | 135 | 1,175,229 |
| SimpleMem | 1,575 | Not captured | 29,386 | 5,932,704 |
| PropMem | 2,963 | 3,790 | 289 | 4,720,077 |

SimpleMem's upstream streaming calls did not report prompt/completion usage to the adapter ledger. The generated upstream summary serializes its missing reader-prompt usage as `0`; this must be interpreted as **not captured**, not as a zero-token prompt. Its retrieved-context count and call latency are recorded. SimpleMem ingestion token counts are likewise unavailable from its SSE response path.

## Local Cost And Calls

The call ledger contains 168 local-Qwen requests and 518 local-embedding requests, all marked successful; hosted-provider calls `0`, judge calls `0`, failures `0`. Sum of local-Qwen request latencies was 11,626,064 ms (about 3 h 14 min) on the local GPU-backed llama.cpp server. Sum of local CPU embedding request latencies was 783,428 ms (about 13 min). These are request-latency sums, not a GPU utilization measurement. Hosted reader, embedding, and judge spend was `$0`.

The embedding model and adapter settings, per-system prompt-token counts, FullContext tokenization evidence, and all system predictions are retained in `run_manifest.json`, `call_ledger.jsonl`, `predictions.jsonl`, `deterministic_metrics.json`, and `token_efficiency.json` under the run directory.

## Recovery And Audit

The prediction artifact is frozen with SHA256 `7a056209a4397d4c16a8eb6c2e84d080ff2aef3831b32c23f1d762d6886e3999`. An unchanged resume preserved the same prediction SHA, the same call-ledger SHA, 5 prediction rows, and 686 ledger rows; no new model calls were made.

Earlier attempts are preserved and not scored:

- r4 exposed a SimpleMem planner-role budget error and PropMem's 600-second local request timeout; those were infra failures, not F1 zeroes.
- r5 completed the other systems but PropMem's 64-token final-answer cap yielded an incomplete JSON object.
- r6 used the shared 256-token cap, but Qwen ended the answer after a complete `answer` string while omitting the outer JSON closing brace; strict upstream parsing raised `RuntimeError`.

The active protocol now uses a shared 256-token answer cap. The pinned PropMem compatibility adapter accepts an incomplete outer object only when the JSON string-valued `answer` field itself is complete; partial answer strings still fail. This parser compatibility change does not alter extraction, retrieval, ranking, or evidence selection. r7 is the first complete run after this correction.

## Decision

`MEM1_LOCAL_ONE_CASE_SMOKE=YES`. This confirms only that the five adapters execute under the frozen local stack, FullContext does not truncate, deterministic metrics and failure separation work, predictions freeze, and cache/resume is stable. It does **not** complete MEM-1 baseline reproduction. Stop here: no 10-case, 102-DEV, or TEST run was started.
