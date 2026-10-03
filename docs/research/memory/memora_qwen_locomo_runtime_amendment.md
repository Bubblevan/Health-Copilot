# Memora Local-Qwen Runtime Amendment

Date: 2026-10-03

## Change

The initial full-run adapter serialized all ten LoCoMo conversations and all QA items despite the pinned upstream LoCoMo implementation supporting worker-pool execution. The v3 attempt was stopped before a conversation checkpoint; its partial Chroma writes and call ledger are retained under `runs/memory/memora-locomo-qwen-v3` for audit and are excluded from all results.

The v4 shared-database run used four workers, but was interrupted after concurrent embedded Chroma/HNSW operations emitted `Error finding id` and missing-segment-reader failures. Its partial artifacts under `runs/memory/memora-locomo-qwen-v4-parallel` are invalid for scoring and are retained only as failure evidence.

The replacement uses four workers across independent conversations and four workers across conversation groups of QA items. Each conversation's sessions and questions remain sequential. Every conversation is assigned a separate local Chroma directory, matching upstream per-user isolation while avoiding concurrent writes to one embedded HNSW database. Reader, memory-internal LLM, judge, embedding model, prompts, retrieval parameters, scoring, dataset, seeds, and upstream Memora source remain unchanged. This changes storage partitioning and scheduling only, not the memory algorithm.

The v5 stress smoke completed four three-session conversation builds and eight QA items per strategy with zero provider failures. That run predates explicit capture of upstream Chroma query retry logs, so it is retained as preliminary qualification only. The storage-audited v6 smoke also passed with no Chroma retries or terminal failures, but a subsequent source audit found the upstream searcher wrote diagnostic results to one strategy-wide JSON path. v6 is retained as preliminary; v7 assigns one output JSON to each strategy/conversation pair. The storage logger records all retry messages with question context, and any terminal query failure prevents `status=complete`.

The pinned llama.cpp build and Qwen weights are unchanged. The aggregate server context remains 131,072 tokens and is shared across four parallel sequences (32,768 tokens per sequence). Every prompt uses the same local tokenizer and no hosted API or API key.

## Qualification Gate

Before the full run, the final v7 `store-stress-smoke` must complete four conversations (three sessions each) and eight QA items across both retrieval strategies, with:

- no provider or embedding failures;
- only loopback Qwen calls and local CUDA embeddings;
- exact expected prediction/judge counts;
- valid, unique question rows and parseable call-ledger JSONL;
- successful resume from the smoke artifacts.
- zero terminal Chroma query failures, with all retry warnings captured in `storage_events.jsonl`.

The full run uses a new runner hash, protocol identity, and artifact root. No partial v3, v4, or preliminary v5 artifact is imported into it.

The first two-conversation parallel pilot exposed an occasional CUDA embedding encoder race and local-Qwen judge responses that ended with a standalone label instead of JSON. Those pilots are retained as diagnostics. Embedding calls are serialized through a narrow lock; judge parsing accepts only a valid JSON label or an unambiguous standalone final label, and reports the latter as a format fallback. The four-conversation one-session qualification passed, including resume with no duplicate model calls; v5 additionally stress-tests three sequential sessions per conversation under isolated Chroma storage.

The storage-audited v6 preliminary smoke passed on 2026-10-03: four conversations, three sessions each, eight QA items per strategy, semantic and prompted-policy both complete; provider/embedding failures `0`, Chroma retry warnings `0`, terminal Chroma failures `0`, unresolved infrastructure failures `0`. The 1,757-row local call ledger SHA-256 remained `0484e2479050a4ca7e9ab153a8ac00e680d08b0d911f157fe690cd7ed847b22d` across a resume run, confirming no model calls were duplicated. It is not the final parallel-output qualification or benchmark evidence.

The final v7 stress smoke passed on 2026-10-03: four conversations, three sessions each, eight QA items per strategy, semantic and prompted-policy both complete; storage retry/terminal failures `0`, unresolved infrastructure failures `0`, and distinct upstream trace paths are assigned per strategy/conversation. The 1,564-row call ledger SHA-256 remained `f7aed7069bd7e953e25ba41ff1bc7d852a7c00d018b7120b3a16bb4da8310601` across resume, with no model calls duplicated. It qualifies the v7 full-run root; it is not benchmark evidence.
