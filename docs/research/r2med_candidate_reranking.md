# R2MED Complementary Candidate Fusion and Reranking

## Frozen scope

This is the final R2MED method-family attempt. It reuses the pinned DEV split
(PMC-Treatment 150, PMC-Clinical 114, IIYi-Clinical 129; 393 queries), existing
LameR-MV and compact CRB-Q top-100 rankings, and the existing corpus/query
source manifest. Existing generated text and rankings remain read-only. TEST is
not opened during DEV analysis or model selection.

The machine-readable protocol was frozen before any reranker inference in
`runs/rag_r2med_rerank/protocol.json`. The baseline BGE cross-encoder is the
public `BAAI/bge-reranker-v2-m3` model, pinned to revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. It is a public component, not an
original contribution. ReasonRank is optional and will be skipped if its
official weights/runtime do not fit the existing 16-GB GPU environment.

## Phase A: zero-model candidate complementarity

Phase A ran only on the existing 393 DEV rankings. Candidate lists were
deduplicated by document ID. Qrels were accessed only after ranking/fusion, for
evaluation. The raw union candidate-pool coverage is the relevance ceiling; its
ordering is never chosen with labels. Ranked union cutoffs use equal-weight RRF
with k=60 and deterministic rank/document-ID tie-breaks.

| DEV subset | LameR R@100 | CRB R@100 | raw pool union R@100 | RRF union R@100 | top-100 Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: |
| PMC-Treatment | 0.7970 | 0.7388 | 0.8282 | 0.7877 | 0.4279 |
| PMC-Clinical | 0.7032 | 0.7171 | 0.7953 | 0.7354 | 0.5009 |
| IIYi-Clinical | 0.6225 | 0.6757 | 0.7306 | 0.6690 | 0.4767 |
| Equal-subset macro | 0.7076 | 0.7105 | **0.7847** | **0.7307** | 0.4685 |

The raw pool union is +0.0742 Recall@100 over the strongest single source,
clearing the frozen +0.005 candidate-complementarity threshold. The RRF-ranked
union has Recall@10/20/50/100 of 0.3804 / 0.4696 / 0.6221 / 0.7307. Mean
relevant-document overlap Jaccard is 0.7181; ordinary top-100 candidate-set
Jaccard is 0.4685.

Across all subsets, positive qrel document counts were 79 found only by
LameR, 86 found only by CRB, 617 found by both, and 238 found by neither. This
confirms real, if asymmetric, complementarity; it does not establish improved
top-rank quality.

Full per-subset counts, input hashes, and metric definitions are in
`runs/rag_r2med_rerank/candidate_analysis.json`.

## Frozen DEV comparison

The baseline matrix retains original BM25, original BGE-large, equal-weight
BM25+BGE RRF (k=60), LameR single-view BGE, LameR-MV, and compact CRB-Q. Rerank
LameR-MV and compact CRB-Q at K=20/30/50. Since Phase A passed, also evaluate
DualSource-RRF with lambda in {0.5, 1.0, 2.0}, k=60, crossed with the same K
grid. The dual-source score is
`1/(60+lamer_rank) + lambda/(60+crb_rank)`; candidates are deduplicated by
document ID and source ranks are retained.

The cross-encoder sees only the original benchmark query and full corpus text,
truncated deterministically to 512 pair tokens. It does not receive CRB text,
answers, relevance labels, or qrels. Exactly the first K items are reranked;
the tail preserves its original order. Pair scores are cached on E: using
query/document/model identities so repeated configurations do not re-run
inference.

Primary selection is equal-subset macro nDCG@10, with MRR@10, Recall@10, smaller
K, then lambda closer to 1 as tie-breakers. The conservative DEV gate compares
the best OURS configuration against the strongest frozen non-OURS arm and
requires at least +0.005 macro nDCG@10 plus non-inferiority on at least two of
three subsets. TEST is allowed only after this gate and a committed final method
lock. AAR is eligible only if DualSource-RRF+BGE strictly beats LameR-MV+BGE.

If the gate fails, report the result and stop R2MED method development: no new
prompt/schema, retriever family, reranker, or post-hoc grid expansion.
