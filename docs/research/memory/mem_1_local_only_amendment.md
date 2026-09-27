# MEM-1 Local-Only Protocol Amendment

**Status:** Active amendment to MEM-1, made after the human review of MEM-0. The frozen MEM-0 audit files and their historical claims are not edited by this amendment. This document and `mem_1_local_only_protocol.json` are normative for all subsequent MEM-1 execution.

## Research Position

The Main Track is a **controlled re-evaluation of public memory architectures under a unified fully-local model stack**. It compares memory/context architecture while holding the reader and embedding models fixed. It is not an exact MemEval numerical reproduction, not the official LongMemEval GPT-4o accuracy reproduction, and not directly comparable to the GPT-4.1 numbers in the MemEval README. Published upstream results remain external historical coordinates only.

## Frozen Local Stack

| Role | Frozen configuration |
|---|---|
| Reader / answer model | Existing Qwen3-8B Q4_K_M GGUF, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`; llama.cpp loopback endpoint only |
| Memory-internal LLM | Same exact Qwen3-8B artifact and loopback endpoint as reader |
| Memory system | FullContext, OpenClaw, Mem0 OSS, SimpleMem, PropMem; architecture under comparison, not a model role |
| Embedding model | `Qwen/Qwen3-Embedding-0.6B`, revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, loaded from a verified local path |
| Judge model | None |
| Hosted API / required API key | None / none |

The embedding adapter uses Transformers with `local_files_only=true`; it verifies the frozen weight SHA before loading and has no download or alternate-model fallback. All dense-compatible baselines receive the same local embedding client. The adapter does not change their memory or retrieval algorithms.

The embedding configuration is pinned as follows: 1024-dimensional output, L2 normalization, query-only instruction `Given a conversation-history question, retrieve memory passages that provide evidence needed to answer the question.`, no document instruction, batch size 8, CPU / float32, maximum input 8192 tokens, truncation enabled and measured. Each run manifest records the repository revision, model-tree hash, weight hash, dimension, normalization, both input instructions, batch size, device, dtype, maximum length and observed truncation.

The initial, invalid smoke attempt used a 64-token final-answer cap; it is retained only as an infrastructure diagnostic and is not the effective protocol. The active shared final-answer cap is 256 tokens for every system, as detailed under “Reader Answer Budget Correction” below. Memory-internal generation uses the same Qwen artifact with an explicit default 8192-token output budget; a system's explicit extraction budget takes precedence. Every SDK request receives an explicit role-appropriate output budget, so runtime behavior does not rely on llama.cpp's server default. The first smoke launch omitted that memory-internal default while the server was launched with `n_predict=64`; it was stopped as invalid before any result was frozen.

Runtime roles are kept distinct in configuration and artifacts:

```text
Qwen3-8B llama.cpp loopback endpoint
  ├── reader_answer
  ├── memory_ingest
  └── memory_reasoning

Qwen3-Embedding-0.6B Transformers local runtime
  └── embedding

judge_model = NONE
```

The OpenAI-compatible Python SDK remains only as the client protocol used to speak to the explicitly loopback llama.cpp endpoint and the pinned upstream adapters. The runner does not read `OPENAI_API_KEY`, instantiate an OpenAI-hosted client, consult provider endpoint variables, or permit non-loopback reader requests. Local model files are opened only from the frozen path.

## Cancelled Historical Track

The previously reviewed upstream-parity sanity proposal is retained here for audit and is cancelled, not silently deleted:

```text
status: CANCELLED_BY_LOCAL_ONLY_AMENDMENT
historical reader: GPT-4.1
historical judge: GPT-4o
historical selection: six DEV questions, one per LongMemEval question type
historical IDs: 1cea1afa, 1c549ce4, 778164c6, fca70973, a82c026e, gpt4_e061b84g
execution under this amendment: forbidden
```

The old GPT-4.1/GPT-4o protocol remains only in the pre-amendment audit/history. It is not a runtime track. GPT-4o is not a reader, and no judge model replaces it in this amendment.

## Split and Metrics

The active split remains the human-approved LongMemEval-S split: the fixed MemEval `seed=42` stratified 102 DEV IDs, with the remaining 398 cases as process-level held-out public TEST. The invalid Oracle-derived candidate and `split_parity_audit.json` remain immutable audit records and are never used for tuning or evaluation. MEM-1 reads DEV only; TEST remains inaccessible until a later explicit method lock.

Primary downstream quality is deterministic token F1. No LLM-as-judge runs. Report by each of the six LongMemEval-S categories: deterministic token F1, answer-session Recall@5, answer-session Recall@10, MRR, retrieved/context token counts, retrieval latency and ingestion latency. Deterministic abstention accuracy is reported separately when abstention cases are present. A category with no cases is reported as not present, never imputed as zero.

Future RevMem work may add stale-memory exposure, current-state conflict, revision-resolution and historical/current-state accuracy diagnostics. They are not attributed to these five baselines or treated as MEM-1 results.

## Cost and Claim Accounting

- Reader and memory-internal generation: local Qwen GPU wall time / token throughput only; no per-token hosted reader estimate.
- Embedding: local CPU wall time and token counts; embedding API cost is `NONE`.
- Judge: `NONE`, with zero calls and zero cost.
- Required API key: `NONE`.
- Main Track results must not be described as exact MemEval reproduction or compared numerically to the upstream GPT-4.1 README results.

## Execution Gate

Run only the frozen DEV question `1cea1afa` through all five systems: FullContext, OpenClaw, Mem0 OSS, SimpleMem and PropMem. Verify all reader and memory-internal calls use the frozen loopback Qwen model; every dense call uses the verified local embedding artifact; no hosted request/key/provider fallback is used; FullContext records an actual prompt-token count, 131072 max model length and `truncated=false`; prediction SHA freeze and successful cache/resume work; and infrastructure failures remain separate from quality metrics.

Write `mem_1b_local_one_case_smoke.md` and set `MEM1_LOCAL_ONE_CASE_SMOKE=YES` only if every gate passes. Otherwise record `NO` and the precise blocking/failure reason. Stop immediately after this one-question smoke. Do not launch 10-case, 102-DEV or TEST runs.

## Reader Answer Budget Correction

The initial local smoke used a shared 64-token answer cap. A PropMem-only loopback diagnostic replay of the frozen prompt showed that this cap truncates its required JSON answer, while a 256-token cap returns parseable JSON. The active controlled protocol therefore uses a shared 256-token final-answer cap for every system. This changes only the uniform answer-generation budget; the reader artifact, prompt, temperature, seed, memory algorithms, embedding model, dataset and selected question remain frozen. The initial 64-token attempt is retained as an invalid infrastructure diagnostic and is not scored.

A subsequent exact Qwen response ended after the complete JSON `answer` string but omitted the outer closing brace. The pinned PropMem compatibility adapter now permits only this narrow repair: it extracts a complete JSON string-valued `answer` field from an otherwise unclosed object. It does not repair a partial answer string or change extraction, retrieval, ranking, or evidence selection. The r6 run remains an infrastructure-failure audit; the corrected patch is used only in the new r7 smoke.
