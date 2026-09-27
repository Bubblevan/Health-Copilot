# MEM-1 Execution Contract

**Status:** The local-only amendment is normative. This supplements, but does not rewrite, the frozen MEM-0 audit. The amendment and its protocol JSON record the prior parity proposal as cancelled.

## Stage Boundary

- Research positioning: **controlled re-evaluation of public memory architectures under a unified fully-local model stack**.
- Dataset is the frozen LongMemEval-S dataset. MEM-1 reads frozen DEV only; the 398-question process-level held-out public TEST is inaccessible until a later method lock.
- Systems are FullContext, OpenClaw, Mem0 OSS, SimpleMem and PropMem. PropMem remains the strong primary comparator.
- This stage does not implement M10-Flat, RevMem, FAMA, Multi-Agent or training.
- Upstream MemEval remains pinned at `807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4`; local adapter changes are carried in the pinned patch.

## Local Model Roles

| Role | Main Track |
|---|---|
| Reader / answer model | Existing frozen Qwen3-8B Q4_K_M, alias `health-memory-qwen3-8b`; loopback llama.cpp only; temperature 0, seed 42, thinking disabled, answer cap 256; 99 GPU layers, Flash Attention, Q4_0 GPU KV |
| Memory-internal LLM | The same Qwen3-8B artifact and loopback endpoint; default budget 8192 new tokens, with explicit baseline budgets honored |
| Memory system | Baseline architecture being evaluated: ingestion, storage, retrieval and context construction |
| Embedding model | Local CUDA `Qwen/Qwen3-Embedding-0.6B`, `cuda:0` / float16, frozen revision and hash; Torch `2.10.0+cu128` local overlay |
| Judge model | NONE in Main Track; an opt-in local Qwen semantic judge is separate and post-freeze only |
| Hosted API / required key | NONE / NONE |

The final-runtime pin is `llama-server 10068 (571d0d540)`, binary SHA256
`3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb`; the
launcher fails closed on any version, build or binary-hash mismatch. It binds
only to `127.0.0.1:8081` with 99 GPU layers, Flash Attention, Q4_0 GPU KV and a
131072-token context.

The OpenAI-compatible SDK is a wire-protocol client only. Its transport disables environment proxies and blocks non-loopback hosts. No `OPENAI_API_KEY`, cloud endpoint variable, proprietary fallback, GPT-4.1 reader, GPT-4o judge, or hosted embedding endpoint is used or required. The adapter explicitly caps final reader answers at 256 for every system and applies an 8192-token default to memory-internal calls; an explicit per-system budget takes precedence. The larger shared answer cap is required for structured-answer baselines such as PropMem to finish valid JSON. Local requests use a 3600-second timeout and the SDK's explicit two retries so long structured extraction is not mistaken for a short reader timeout. No server-side generation default is relied upon.

The embedding adapter loads only local Transformers files, verifies the pinned model revision and weights, returns 1024-dimensional L2-normalized vectors, uses the frozen query instruction and no document instruction, batches by 8 / at most 8192 input tokens, and records device, dtype, token limit and actual truncation. Every dense-compatible baseline receives this adapter; baseline memory/retrieval algorithms remain unchanged. The CUDA wheel is an ignored local runtime overlay; MemEval's pinned `uv.lock` remains untouched.

The isolated MemEval venv initially selected a CPU-only PyTorch wheel because its upstream dependency range did not pin a CUDA build. That was an installation-target mistake: system Anaconda already had CUDA-enabled PyTorch. The isolated venv now uses the official `2.10.0+cu128` wheel and sees the RTX 4090. The local-only reinstall script changes only this ignored venv and does not edit `uv.lock` or the system Python installation. The frozen local reader launch and measured trade-off are recorded in `mem_1c_context_isolation_amendment.md` and `runs/memory/mem1c_reader_gpu_q4_115k_verified.json`.

## Answer Tracks

- `native` preserves the upstream answer heads for adapter/runtime audit only.
- `context_controlled` asks each system for its native ContextBundle, then uses one shared Qwen3-8B answer prompt and generation config. Only the bundle content differs. Bundles are persisted and SHA-verified before the reader result can freeze.
- FullContext preflight reserves the protocol's same 256 output tokens used by the actual answer request.
- Abstention questions are identified by `question_id.endswith("_abs")`. A single exact normalized refusal helper defines both empty-gold token F1 and deterministic `abstention_accuracy`.

## Metrics and Claim Limits

- Primary controlled-track metrics: answer-session Recall@5, Recall@10, MRR, `context_reader_tokens`, retrieval/ingestion latency and provenance coverage. Downstream answers report token precision, recall, F1 and normalized exact match.
- `context_reader_tokens` is measured on serialized shared-reader context by the frozen Qwen3-8B llama.cpp `/tokenize` endpoint (`add_special=false`). `context_embedding_tokens` uses the embedding tokenizer and is diagnostic only; tokenizer equivalence is not assumed.
- Retrieval: answer-session Recall@5, Recall@10 and MRR when source-session provenance is available; otherwise report `null`, not a fabricated miss.
- Efficiency: `context_reader_tokens`, reader prompt tokens, retrieval latency, ingestion latency, local reader wall time, and local embedding wall time/token counts.
- Report each metric by the six LongMemEval-S question categories. Report deterministic abstention accuracy separately if abstention cases are present.
- No LLM-as-judge runs in Main Track. An opt-in local Qwen judge reads only frozen predictions, uses separate cache/trace artifacts, and reports `local_qwen_judge_accuracy`; it is not official GPT-4o accuracy and is headline-ineligible until manual 10-case disagreement calibration.
- FullContext requires an actual 131072-token slot. Measure the exact llama.cpp chat-template output with `/apply-template` and `/tokenize`; compare that count to successful server `usage.prompt_tokens`. Record `max_model_length=131072` and set `truncated=false` only when counts agree and prompt plus output reserve fits. Missing or mismatched evidence invalidates the item.
- Do not claim exact MemEval numerical reproduction, official LongMemEval GPT-4o accuracy reproduction, or direct performance comparability with MemEval README GPT-4.1 values. Upstream numbers are external historical coordinates only.

## Failure and Resume Rules

- `failure_attribution_hint` is a lexical heuristic with `failure_attribution_heuristic=true`, not a causal finding. Token-subset overlap cannot prove context sufficiency.
- Reader, ingestion, embedding, schema, timeout and library errors are `INFRA_FAILURE`, have null prediction/F1 and are excluded from the quality denominator. They remain separately logged and retryable; embedding failure is never quality zero.
- PropMem's local response adapter accepts only complete JSON or the one known case where a complete string-valued `answer` is the final member and only the outer brace is missing. Malformed, partial, duplicate, extra-field, or trailing-content forms remain infrastructure failures.
- Missing SimpleMem streaming usage is `null` / `NOT_CAPTURED`, never numeric zero. Do not make total-token Pareto claims involving SimpleMem unless reliable usage becomes available without changing its algorithm.
- Internal query planning/retrieval LLM calls use `memory_reasoning`; final answer-generation calls use the shared 256-token `reader_answer` budget.
- No credential, request body or key is written to manifests or ledgers. Logs contain role, local provider, model, tokens, latency, retry/cache status, success and truncation metadata.
- Cache identity includes system, question ID, dataset SHA, system config and prompt hashes, reader artifact SHA, embedding artifact hash when used, and adapter/patch code hash. Only exact-identity successful rows are reusable.
- On successful generation, write `predictions.sha256` and verify it on resume. Context-controlled runs also freeze `context_bundles.sha256`. A frozen run resumed unchanged must not make new model calls.

## MEM-1C Final-Runtime One-Case Gate

Run only the `context_controlled` answer track for DEV question `1cea1afa` through FullContext, OpenClaw, Mem0 OSS, SimpleMem and PropMem. Do not run the native answer track. Verify:

1. Reader and all memory-internal generation use the frozen local Qwen3-8B artifact.
2. Every dense operation uses the frozen local Qwen3-Embedding-0.6B artifact.
3. No hosted request, OpenAI key, hidden fallback or outbound model call is possible.
4. FullContext has actual prompt-token evidence, `max_model_length=131072`, and verified `truncated=false`.
5. Both context tokenizer counts are explicit; shared reader prompt template and generation are identical, and no native answer head is called.
6. Deterministic metrics, ContextBundle SHA freeze, prediction SHA binding/freeze, cache/resume, and infra-versus-quality separation work.

Write `mem_1c1_context_controlled_one_case.md` with `MEM1_CONTEXT_CONTROLLED_ONE_CASE=YES|NO`. `YES` is allowed only if every gate passes. Then stop. Do not run native, 10-case, 102 DEV, TEST, M10-Flat or RevMem.

## Cancelled Historical Proposal

The earlier draft proposed a six-case GPT-4.1 reader / GPT-4o judge upstream-parity stage, GPT-4o native judge for Main Track, and `text-embedding-3-small` API embedding. Those runtime stages are **`CANCELLED_BY_LOCAL_ONLY_AMENDMENT`** and must never execute under MEM-1. Their historical details remain in `model_protocol.json` and `mem_1_local_only_amendment.md` as audit records, not as active configuration. The first local smoke launch omitted a memory-internal output budget while using a 64-token server default; that attempt was stopped and is invalid, not a scored run.

## Cost Accounting

Report local Qwen GPU wall time/compute and local CUDA embedding wall time/token counts separately. Judge API cost and embedding API cost are both `NONE`; hosted API calls are zero. Do not reuse the prior 55M GPT-4.1-token / hosted-reader estimate for this local Main Track.
