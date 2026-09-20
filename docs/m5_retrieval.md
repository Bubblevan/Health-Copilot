# M5 — Hybrid Retrieval & Evaluation Scale

M5 keeps the upper contract unchanged: `Retriever.search(query, top_k) -> list[Evidence]`. Agent, capability policy,
search tool, M3 claim-first handling and M4 record/replay do not inspect which retrieval implementation is active.

## Contracts

`KnowledgeCard` remains a reviewed product-knowledge contract. `RetrievalDocument` is a generic external-corpus unit
and deliberately has no reviewer, publisher, or clinical-provenance fields. The NFCorpus adapter uses only
`RetrievalDocument`; it does not fabricate product-card provenance.

The frozen `BM25Retriever` remains k1=1.5/b=0.75. `DenseRetriever` receives vectors through a provider-independent
embedding backend, normalizes them, uses cosine ranking and breaks ties by document ID. Its persisted index manifest
binds embedding identity/dimension, normalization, corpus hash, product pack version, document count and build commit.
Manifest mismatch fails rather than silently reusing stale vectors.

`HybridRetriever` uses reciprocal-rank fusion, never direct addition of BM25 and cosine scores: those scores have
different scales. RRF's rank begins at one, has an explicit `rrf_k`, deduplicates document IDs and breaks ties by ID.
`RerankedRetriever` takes a bounded candidate pool and applies an independent reranker component.

## Backends and scope

CI uses deterministic fake and hashing character-ngram vector backends. The hashing backend is a runnable vector
baseline, not a learned semantic embedding. Optional `.[retrieval]` adds sentence-transformers local learned embedding
and cross-encoder implementations; model weights remain outside Git and the runtime defaults to local-files-only to
avoid implicit network access during an experiment.

The current M5 product retrieval-only artifact is [`runs/m5/20260920T183628+0800`](../runs/m5/20260920T183628+0800). On the frozen
M0 pack's 62 source-labelled cases, learned Hybrid+CrossEncoder has Hit@1 0.9355 and MRR 0.9651, versus BM25 0.9032 and
0.9315; its mean retrieval latency is about 141 ms versus BM25 about 0.08 ms in that local run. This is a reviewed-existing-pack
retrieval diagnostic, not clinical correctness, a generalization claim, or an automatic production-default decision.

The same stack has a six-case, reviewed M3/M4 focused end-to-end diagnostic in
[`runs/m5/20260920T182649+0800`](../runs/m5/20260920T182649+0800). All four arms returned the four expected answers,
had no unexpected abstain, no OOD answer, and no OOD tool execution in that one run. The four answerable cases and two
OOD controls are too small to call those stable rates or to select a product default; the artifact records runtime call,
token, and latency observations for that reason.

## Component-derived product suite and ablation

[`evals/retrieval/m5_product_retrieval_v1.jsonl`](../evals/retrieval/m5_product_retrieval_v1.jsonl) has 80 rows:
40 mechanically selected source-labelled rows from frozen M0 and all 40 rows from the already review-manifest-approved
M0 expansion. Its manifest binds both component SHA256 values and explicitly records that it asserts no new annotation
or user approval. It has 74 source-anchored rows and six corpus-uncovered controls.

The corresponding four-arm run is [`runs/m5/20260920T184208+0800`](../runs/m5/20260920T184208+0800):

| Retriever | Hit@1 | Recall@5 | MRR | nDCG@5 | Mean retrieval latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.8784 | 0.9640 | 0.9137 | 0.9230 | 0.10 ms |
| Dense | 0.8108 | 0.9212 | 0.8680 | 0.8686 | 12.66 ms |
| Hybrid RRF | 0.9054 | 0.9685 | 0.9369 | 0.9396 | 12.84 ms |
| Hybrid + CrossEncoder | 0.9459 | 0.9977 | 0.9673 | 0.9733 | 137.63 ms |

The deterministic failure table shows BM25 semantic misses on `m5r-002`/`m5r-003`; dense has a jurisdiction
confusion on `m5r-041` and additional lexical/semantic misses; RRF reduces, but does not eliminate, those errors.
Reranking rescues several BM25 failures (`m5r-001`, `m5r-002`, `m5r-003`, `m5r-033`, `m5r-052`, `m5r-054`,
`m5r-061`, `m5r-073`) while retaining a source-overlap failure on `m5r-034`. The six corpus-uncovered controls are
recorded as retrieval diagnostics, not as a threshold-based claim that a returned document is an answer; M3 capability
policy remains responsible for tool authorization and answerability.

The external NFCorpus hashing-baseline artifact is distinct: BM25 MRR 0.5182/nDCG@10 0.3110; hashing dense and hybrid
are lower. External benchmark figures must not be presented as Health-Copilot product performance.

## Status

M5 is **frozen** at implementation/evaluation checkpoint `main@a69801bde6826daaf02aa933c2cdf2a05697d42a`.
`DEFAULT_RETRIEVER` remains **BM25**: Hybrid+CrossEncoder wins both reviewed retrieval diagnostics, but costs roughly
three orders of magnitude more local retrieval latency, the end-to-end pack is only six cases, and the closed corpus is
small. This is a conservative production-default decision, not a claim that BM25 wins every retrieval metric. M6 is not
started.
