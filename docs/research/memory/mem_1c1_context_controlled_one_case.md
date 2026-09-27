# MEM-1C1 Final-Runtime Context-Controlled One-Case Gate

`MEM1_CONTEXT_CONTROLLED_ONE_CASE=YES`

This is runtime and diagnostic evidence for one frozen public DEV question only. It is not a performance ranking or a quality claim.

## Run Identity

- Base commit: `ffd6f90991b9edc2b3b7740726b7f12d392e407d`
- Run: `mem1c1_context_controlled_one_case_20260927`
- Track: `context_controlled` only; native answer track was not run
- Question: `1cea1afa` (`knowledge-update`); DEV only; TEST access `false`
- Dataset SHA256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- MemEval: pinned at `807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4`; patch SHA256 `99a9f9cb82ffb9420722ee20fcf3101ec2d67a8164e1dbba8567317409d304eb`; patched tree SHA256 verified as `760db4e71f58b840c3ea3413bd4c8d734190fd20`
- External checkout was restored to the pinned commit with a clean worktree after the run.

## Runtime And Isolation

- Reader / memory-internal model: frozen Qwen3-8B Q4_K_M, artifact SHA256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`
- llama.cpp: `10068 (571d0d540)`, executable SHA256 `3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb`; `127.0.0.1:8081`; 99 GPU layers; Flash Attention; Q4_0 GPU KV; 131072 context; YaRN scale 4 / original context 32768
- Embedding: `Qwen/Qwen3-Embedding-0.6B`, revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, model-tree SHA256 `9d2d790d6448ef2c0911ffeb03f959d035c71ac3d2b14b7d586f2d2b39fb0efa`, 1024 dimensions, L2 normalized, CUDA `cuda:0` / FP16
- Judge: none. Hosted calls: `0`. The run process had `OPENAI_API_KEY`, hosted base URL and judge-model variables removed; Transformers/Hugging Face ran offline.
- All 685 recorded calls succeeded: 166 local Qwen calls (5 shared-reader answers, 155 memory-ingest calls, 6 memory-reasoning calls) and 519 local embedding calls. FullContext does not use dense embeddings. No native answer head was invoked: exactly one `reader_answer` call per system.
- All five answer rows share reader-template SHA256 `0ff70b000bd4b43db85ae587731fea35b691febc815cfbb57c2acb1c8b295b2b` and the same temperature 0, seed 42, thinking-disabled, 256-token generation configuration. Actual prompt hashes differ because the ContextBundle is the only system-varying input.
- FullContext actual shared-reader prompt: 108,137 Qwen3-8B tokens. With the 256-token reserve, 108,393 tokens fit within 131,072. The reader call reported `truncated=false`; maximum model length was 131,072.
- Runner/provider infrastructure failures: `0`; failed provider calls: `0`; all five prediction rows are `quality_status=OK`.
- After resume, prediction SHA256 `42d231228db11175e8676ef961dd8fe384cdc03a2712bd2afe128bb5a380cc03`, ContextBundle SHA256 `f3f8ae24cebbeb803371f63bd38a710ffef0448ea3b86e666dc3e428786a91f6`, and call-ledger hash/row count were unchanged (`5` predictions, `5` bundles, `685` calls). Cache/resume made no new model calls.
- Frozen call-ledger SHA256: `478860b63b5700da266084af5f43b9b7ea49ac913c1850a7004455abd0547d2f`.
- The Health-Copilot reader was stopped after verification; GPU usage returned to 57 MiB / 16376 MiB.

## Per-System Diagnostics

`Context reader tokens` uses the loaded frozen Qwen3-8B tokenizer via loopback llama.cpp `/tokenize` with `add_special=false`. `Context embedding tokens` uses the embedding model tokenizer and is diagnostic only; no equivalence is assumed. Context-size comparisons use reader-token counts. Recall metrics are `null` when the system does not expose auditable answer-session provenance, not zero.

| System | Context reader tokens | Context embedding tokens | Recall@5 | Recall@10 | MRR | Token precision | Token recall | Token F1 | Normalized EM | Retrieval ms | Ingestion ms | Failure attribution hint (`heuristic=true`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FullContext | 108064 | 108065 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 | `CONTEXT_HAS_ANSWER_READER_MISSED` |
| OpenClaw | 7267 | 7268 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 84.896 | 7570.097 | `CONTEXT_HAS_ANSWER_READER_MISSED` |
| Mem0 OSS | 452 | 453 | null | null | null | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 59.381 | 548119.296 | `FALSE_ABSTENTION`; `CONTEXT_MISSING_ANSWER` |
| SimpleMem | 1462 | 1463 | null | null | null | 0.1000 | 1.0000 | 0.1818 | 0.0000 | 11681.752 | 515957.952 | none |
| PropMem | 3314 | 3315 | 1.0000 | 1.0000 | 1.0000 | 0.2000 | 1.0000 | 0.3333 | 0.0000 | 130.430 | 726760.601 | none |

The lexical `failure_attribution_hint` is heuristic, not causal evidence. Token-subset overlap cannot prove that context was sufficient or insufficient.

## Observed Baseline Warnings

- Mem0 OSS printed an internal memory-action DELETE handler error (`Error: '14'`) and an `Invalid JSON response` warning during ingestion. These did not fail the local provider call or prevent a ContextBundle/prediction from completing. Treat them as baseline ingestion-quality warnings; they are not hidden by the runner's zero infrastructure-failure count.
- SimpleMem reported that optional FTS index creation was skipped because the `lance` library was unavailable. It continued with its vector-store path and produced a ContextBundle/prediction. This is disclosed as an environment diagnostic; it is not a performance claim.

This one-case gate confirms final-runtime isolation, reader/tokenizer semantics, artifact integrity, and resumability only. It does not authorize 10-case, 102-DEV, TEST, M10-Flat, or RevMem runs.
