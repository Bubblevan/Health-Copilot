# MEM-1D2 Reflection And Provenance Review

- Gate: `MEM1D2_REFLECTION_PACKET_READY=YES`
- Scope: frozen MEM-1D1 ten DEV cases only; zero model, embedding, reader, judge, or inference-provider calls.
- Dataset revision/hash: `98d7416c24c778c2fee6e6f3006e7a073259d48f` / `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`; only the ten frozen DEV records were decoded.
- No TEST records were decoded or analyzed; no causal failure labels were assigned.
- The four MEM-1D1 evidence JSONL files remain byte-identical to their frozen SHA sidecars.

## Frozen Evidence

| Artifact | SHA256 |
|---|---|
| `context_bundles.jsonl` | `18f38700f7cc0b3b44ab45c8c29007a21c44b79e600c07254665cebb9bb9e26c` |
| `predictions.jsonl` | `b90d6d6e7b08a720d358b14c114c553b432f4bf488fb2773b8060056cc34125f` |
| `call_ledger.jsonl` | `480ab337bdd02ce8f573d0737d61931224698d338a8bcd39e2b0f3ac319b946b` |
| `baseline_warnings.jsonl` | `838de6f71474fcab98ce88cb06797aa75cfb1a0e9b000529e311fd3f402dfb7e` |

## ContextBundle Status Semantics

`context_bundle_content_valid` means each canonical bundle and prediction binding validates. `context_bundle_hash_frozen` is true only when `context_bundles.sha256` exists and verifies. This post-hoc field cleanup changes reporting semantics only; it does not change any frozen MEM-1D1 evidence row.
For this run: content valid = `True`; hash frozen = `True`. The historical `deterministic_metrics.json` was not overwritten; these unambiguous fields live in the D2 diagnostics artifact.

## Provenance Root Cause And Parity

The historical adapter attached one session ID per logical turn even when a turn rendered to multiple physical Markdown lines. The chunker indexes physical line numbers, so after the first embedded newline later line-to-session lookups could shift. Date/header/blank lines correctly remain null in the repaired map.

| Question | Markdown bytes equal | Chunk count | Ordered text/hash/range digest |
|---|---:|---:|---|
| 1cea1afa | True | 498 | `82caad805c51578a33eddb2811182d9f9d96e67217dde9cffc18b72ebc31bfcf` |
| 1c549ce4 | True | 484 | `c298c0959bd1005460d59d0817ee8d3e4d1f837b232f01059040171fe95f89bc` |
| 778164c6 | True | 486 | `3896f41a301abbca6e100246ff29f7522be40b6ebde137145b7ec3319dfff43c` |
| fca70973 | True | 471 | `0c8c400d34efcbd30e1564277de2196760aff9121d0a1e8c24933f15b57a5062` |
| a82c026e | True | 489 | `4652b2b211a8ef0405f86660fb8a6a1d37195441c587eeae09978fd97f09c898` |
| gpt4_e061b84g | True | 509 | `dd618b04bfc72493ddc20f2cb44a6e3345b7d925ba3d4d7d2426a4c1b77e22cb` |
| gpt4_f420262c | True | 482 | `4e5913af41277394e526098b35206321dd9c9c1cea2b6695d33c8512708b8d4e` |
| 8550ddae | True | 484 | `c24b426d5f92cbfb167767b874dc0965a7b1ca5c438c264740559a59a4806324` |
| 06878be2 | True | 494 | `945f7969c58903a316f45f2480df5e621c24f490962e1940c78a00b4eb0acf5f` |
| c4ea545c | True | 507 | `e13d402f30cf8a775adc0ec0d9a14b175ddb2e4414791dcbef9544568d8fd233` |

All ten old/corrected Markdown byte comparisons and all reconstructed OpenClaw chunk text/hash/order comparisons pass. The overlay contains 260 affected OpenClaw/PropMem fallback items; 0 identical-text matches require conservative provenance intersection.

OpenClaw retrieved chunks and PropMem fallback chunks are matched by exact text and SHA256. PropMem proposition session IDs are retained directly and are not rewritten. FullContext remains trivial full-history coverage; Mem0 and SimpleMem provenance remains unavailable.

## Corrected Retrieval Diagnostics

Original OpenClaw/PropMem MEM-1D1 recall/MRR values are `SUPERSEDED_FOR_DIAGNOSTIC_INTERPRETATION`; original artifacts remain unchanged. These corrected values are session-level diagnostics, not answer-quality claims.

| System | Validity | Recall@5 | Recall@10 | MRR | Gold token coverage | Context reader tokens |
|---|---|---:|---:|---:|---:|---:|
| fullcontext | TRIVIAL_FULL_HISTORY_COVERAGE | 1.000 | 1.000 | 1.000 | 0.996 | 106485.100 |
| openclaw | CORRECTED_CHUNK_PROVENANCE | 0.843 | 0.930 | 0.825 | 0.876 | 7173.900 |
| mem0 | UNAVAILABLE | - | - | - | 0.396 | 442.600 |
| simplemem | UNAVAILABLE | - | - | - | 0.535 | 2255.900 |
| propmem | DIRECT_PROPOSITION_PLUS_CORRECTED_FALLBACK_PROVENANCE | 0.693 | 0.747 | 0.668 | 0.846 | 3530.700 |

## Category Separation

| Category | LongMemEval type(s) | Questions | Interpretation boundary |
|---|---|---:|---|
| Single-session factual recall | single-session-user, single-session-assistant | 3 | Direct fact surfacing; not preference abstraction. |
| Multi-session | multi-session | 1 | Cross-session composition; not reducible to temporal ordering. |
| Temporal reasoning | temporal-reasoning | 2 | Event ordering/relations. |
| Knowledge update | knowledge-update | 2 | Current versus prior state. |
| Preference | single-session-preference | 2 | Synthesis/abstraction; not evidence for temporal revision by itself. |

The fixed sample is descriptive and sparse by category; no rebalancing or causal classification was performed.

## Comparator Cost

| System | Avg ingestion ms | Avg retrieval ms | Avg context reader tokens | Memory-internal LLM calls | Embedding calls/tokens | Warning rate |
|---|---:|---:|---:|---:|---:|---:|
| fullcontext | 0.000 | 0.000 | 106485.100 | 0 | 0/0 | 0.00% |
| openclaw | 9717.299 | 265.119 | 7173.900 | 0 | 628/1679989 | 0.00% |
| mem0 | 542570.956 | 149.743 | 442.600 | 908 | 1851/66107 | 50.00% |
| simplemem | 1020887.784 | 17715.961 | 2255.900 | 214 | 87/93290 | 70.00% |
| propmem | 1005175.959 | 281.886 | 3530.700 | 477 | 2322/2049048 | 0.00% |

This is descriptive cost accounting only. No quality/cost winner is computed.

## Reflection Packet

The packet contains 50 rows with frozen answers, compact context items, answer-session evidence excerpts, corrected provenance, context diagnostics, warning types, and heuristic lexical hints. Long FullContext items are represented by hashes and local evidence windows, never copied wholesale.
No Reflection-causal labels (for example `RETRIEVAL_EVIDENCE_MISS` or `TEMPORAL_ORDERING_FAIL`) were assigned.

## M10 And RevMem Boundary

The current M10 substrate already has typed `MemoryRecord`, ADD/UPDATE/DELETE/NOOP, versions, SUPERSEDED status, `supersedes_id`, `valid_from`/`valid_until`/expiry, an active-state materialized view, scope/objective/intent filtering, and deterministic lexical retrieval (see `src/health_ai_copilot/runtime/memory.py` and `src/health_ai_copilot/runtime/context_manager.py`).
Retire the provisional name `M10-Flat`; the future public-benchmark adapter is `M10-Base`. No M10-Base code is implemented in MEM-1D2.
Any earlier locked protocol occurrence of `M10-Flat` is retained as historical terminology only and is superseded by this naming decision.

Candidate hypotheses only: H1 compact context may reduce attention dilution; H2 session hits may miss answer-bearing items; H3 knowledge updates may need explicit current/history resolution; H4 CURRENT/AS_OF/CHANGE may be distinct intents; H5 temporal revision alone may not solve multi-session composition; H6 preference may need separate synthesis policy. All remain untested until Reflection review.

## Artifacts

- Provenance overlay: `docs/research/memory/mem_1d2_provenance_overlay.json`
- Retrieval/context metrics: `docs/research/memory/mem_1d2_retrieval_metrics.json`
- Reflection packet: `docs/research/memory/mem_1d2_reflection_packet.json`

STOP: no M10-Base, RevMem, 102 DEV, TEST, judge, embedding, SFT, or RL stage was started.
