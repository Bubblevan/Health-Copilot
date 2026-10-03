# Memora Local-Qwen Runtime Amendment

Date: 2026-10-03

## Change

The initial full-run adapter serialized all ten LoCoMo conversations and all QA items despite the pinned upstream LoCoMo implementation supporting worker-pool execution. The v3 attempt was stopped before a conversation checkpoint; its partial Chroma writes and call ledger are retained under `runs/memory/memora-locomo-qwen-v3` for audit and are excluded from all results.

The replacement run uses four workers across independent conversations and four workers across independent questions. Each conversation's own sessions remain sequential. Reader, memory-internal LLM, judge, embedding model, prompts, retrieval parameters, scoring, dataset, seeds, and upstream Memora source remain unchanged. This is scheduling-only, not a method change.

The pinned llama.cpp build and Qwen weights are unchanged. The aggregate server context remains 131,072 tokens and is shared across four parallel sequences (32,768 tokens per sequence). Every prompt uses the same local tokenizer and no hosted API or API key.

## Qualification Gate

Before the full run, `parallel-smoke` must complete four conversations (one session each) and eight QA items across both retrieval strategies, with:

- no provider or embedding failures;
- only loopback Qwen calls and local CUDA embeddings;
- exact expected prediction/judge counts;
- valid, unique question rows and parseable call-ledger JSONL;
- successful resume from the smoke artifacts.

The full run has a new runner hash, protocol identity, and artifact root. No partial v3 artifact is imported into it.

The first two-conversation parallel pilot exposed an occasional CUDA embedding encoder race and local-Qwen judge responses that ended with a standalone label instead of JSON. The pilot is retained as a failed diagnostic. Embedding calls are now serialized through a narrow lock; judge parsing accepts only a valid JSON label or an unambiguous standalone final label, and reports the latter as a format fallback. The four-conversation qualification then passed, including resume with no duplicate model calls.
