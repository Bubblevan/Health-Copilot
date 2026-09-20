# M5 closeout

The M5 retrieval stack is frozen at implementation/evaluation checkpoint `main@a69801bde6826daaf02aa933c2cdf2a05697d42a`.

- The product suite has 80 component-derived rows (74 source-anchored, six corpus-uncovered controls), with component
  review provenance in `evals/retrieval/m5_product_retrieval_v1.manifest.json`.
- Hybrid+CrossEncoder leads this suite's retrieval metrics: Hit@1 0.9459, Recall@5 0.9977, MRR 0.9673, nDCG@5 0.9733.
- BM25 is retained as `DEFAULT_RETRIEVER`: it is about 0.10 ms mean locally versus about 137.63 ms for the reranked
  hybrid, and the end-to-end comparison is a six-case focused diagnostic rather than a generalization estimate.
- The four-arm M3/M4 focused run had 4/4 expected answers and 0/2 OOD answers/tool executions in every arm. It does not
  prove stable rates or remove the need for M3 CapabilityPolicy.
- NFCorpus hashing-baseline results remain external-only: BM25 has the best observed MRR/nDCG@10 among those hashing
  arms. They are not Health-Copilot product-performance claims.

M6, SFT, DPO, GSPO, and RL were not started.
