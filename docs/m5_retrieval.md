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

The external NFCorpus hashing-baseline artifact is distinct: BM25 MRR 0.5182/nDCG@10 0.3110; hashing dense and hybrid
are lower. External benchmark figures must not be presented as Health-Copilot product performance.

## Status

M5 is **not frozen**. The existing 40-case approved M0 expansion is useful evidence but does not by itself satisfy the
new, category-balanced 60–100-case M5 retrieval release requirement. The four-arm focused end-to-end ablation is now
complete, but the required expanded product retrieval release and its failure review remain outstanding. Therefore BM25
remains the product default and M6 is not started.
