# Memora LoCoMo Local-Qwen Protocol

## Objective

Evaluate the architecture transfer of Microsoft's Memora on the complete public LoCoMo dataset under one fully local Qwen model stack. This is not an exact reproduction of the paper's proprietary GPT reader, writer, embedding, or judge configuration, and its scores must not be presented as directly comparable to the paper's published GPT numbers.

## Method Mapping

The pinned upstream Memora implementation is used without tracked source edits. Its LoCoMo path is retained for:

- LLM-based conversation segmentation into topical episodes.
- Memory construction with a primary abstraction (`index`) and specific memory value (`value`). The upstream embedding function indexes the primary abstraction, while the value is retained as the content.
- LLM update-or-create decisions and merge history for similar existing memories.
- Cue-anchor generation and cue-to-primary links, giving a memory multiple retrieval handles.
- Episodic memory storage from original conversation segments, linked to extracted factual entries.
- Local Chroma persistence, BM25 hybrid retrieval, timestamped context projection, and the official answer prompt.
- Both upstream retrieval strategies: one-shot semantic retrieval and the zero-shot prompted policy (`EXPAND`, `RE_QUERY`, `STOP`, maximum four steps).

The run does not train or evaluate GRPO. The two retrieval strategies share the same built memory and the same Qwen model configuration.

## Frozen Data And Models

- Dataset: LoCoMo `locomo10.json`, 10 conversations, 1,986 questions; SHA-256 and upstream revision are frozen in each run manifest.
- Reader / answer model: local Qwen3-8B Q4_K_M through llama.cpp 10068 / build `571d0d540`, loopback `127.0.0.1:8081`, 131,072 context, 99 GPU layers, Flash Attention, Q4_0 GPU KV.
- Memory-internal LLM: the same Qwen3-8B service; local deterministic calls use temperature 0 and seed 42. Answer output is capped at 256 tokens; internal operations at 8,192 tokens.
- Embedding: local `Qwen/Qwen3-Embedding-0.6B`, pinned repository revision and weights hash, CUDA FP16, 1,024-dimensional normalized vectors. Query uses the frozen conversation-memory retrieval instruction; document instruction is empty.
- Judge: the same Qwen3-8B service, using the official Memora LoCoMo `ACCURACY_PROMPT`. Category 5 is excluded only from judge accuracy, as in upstream evaluation; deterministic metrics include all categories.
- Hosted model/embedding APIs and API keys: none. A process-local outbound socket guard rejects non-loopback connections; image URL checks are disabled so the benchmark cannot fetch remote media.

## Metrics

- Upstream Memora token F1: set-overlap F1 after lowercase and its punctuation splitting; this is deliberately named `official_f1` in artifacts.
- Upstream exact match: case-insensitive exact string equality.
- Additional normalized EM: lowercase, remove English articles and punctuation, then normalize whitespace.
- Local Qwen judge accuracy, reported separately and not treated as the paper's GPT-4o-mini judge score.
- Per-category quality, local reader-token estimate over formatted memory lines, retrieval latency, answer latency, local provider token usage when available, and ingestion latency/memory counts.
- Prompted-policy minus semantic paired F1 difference with 10,000 category-stratified bootstrap resamples (seed 42). A CI crossing zero is inconclusive.

Memory-context tokens are counted with the local llama.cpp Qwen tokenizer over formatted memory lines joined with newlines. This is a context-size diagnostic, not the full answer prompt token count. Prompt/completion counts are taken from local llama.cpp usage when present; missing usage is reported as unknown, never imputed as zero.

## Resume And Failure Semantics

Each conversation has an atomic checkpoint containing its memory-build log and timing. An interrupted conversation is rebuilt from its start, while completed conversations remain reusable. Each answer and judge row is append-only and keyed by frozen question ID. Local reader, retrieval, embedding, and judge failures are recorded separately; failed calls cannot silently become a quality score of zero.

## Provenance

- Memora paper snapshot and local PDF checksum: [memora_paper_snapshot.md](memora_paper_snapshot.md).
- Runner and artifact manifests record the upstream commits, model hashes, local runtime, split/question-manifest hash, software versions, and runner SHA-256.
- Raw predictions, judgments, local call ledger, Chroma state, and resumable checkpoints remain under the generated run directory and are not committed as repository source.
