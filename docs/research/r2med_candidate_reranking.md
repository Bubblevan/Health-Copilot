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
grid. B2 is truncated to the fused top 100. The pinned per-subset SHA256
identities of the B0/B1/B3 inputs, as well as the LameR/CRB candidate rankings,
are recorded in the protocol. The dual-source score is
`1/(60+lamer_rank) + lambda/(60+crb_rank)`; candidates are deduplicated by
document ID and source ranks are retained.

The cross-encoder sees only the original benchmark query and full corpus text,
truncated deterministically to 512 pair tokens. It does not receive CRB text,
answers, relevance labels, or qrels. Exactly the first K items are reranked;
the tail preserves its original order. Pair scores are cached on E using
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

## Phase B/C: frozen DEV result and stop decision

The pinned reranker was downloaded and checked at revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Inference ran on the RTX 4090
Laptop in CUDA FP16 and scored 28,130 unique query-document pairs. No existing
generation or retrieval artifacts were modified. The complete machine-readable
report, including per-subset metrics, model-file hashes, and cache identity, is
`runs/rag_r2med_rerank/dev_report.json`.

| DEV arm | Macro nDCG@10 | Macro MRR@10 | Macro R@10 | Macro R@100 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.1913 | 0.2341 | 0.2419 | 0.5215 |
| BGE-large | 0.1874 | 0.2362 | 0.2407 | 0.5559 |
| BM25 + BGE RRF | 0.2169 | 0.2634 | 0.2927 | 0.6270 |
| LameR single-view BGE | 0.2892 | 0.3613 | 0.3349 | 0.6508 |
| LameR-MV | **0.2998** | 0.3546 | 0.3795 | 0.7076 |
| Compact CRB-Q | 0.2921 | 0.3493 | 0.3683 | 0.7105 |
| DualSource-RRF, lambda=0.5 | 0.3019 | **0.3630** | **0.3806** | 0.7195 |
| LameR-MV + BGE rerank, K=20 | 0.2009 | 0.2293 | 0.2985 | 0.7076 |
| Compact CRB-Q + BGE rerank, K=20 | 0.1961 | 0.2232 | 0.3008 | 0.7105 |
| Best DualSource + BGE rerank, lambda=2, K=20 | 0.1953 | 0.2214 | 0.2992 | 0.7230 |

K=30 and K=50 were also evaluated for every predeclared reranking source and
DualSource weight; both were worse than K=20. The unreranked DualSource-RRF
lambda=0.5 arm is the best OURS arm, at 0.301918, just +0.002117 over the
strongest non-OURS baseline, LameR-MV at 0.299800. Its subset nDCG@10 deltas
versus LameR-MV are:

| DEV subset | LameR-MV | DualSource-RRF lambda=0.5 | Delta |
| --- | ---: | ---: | ---: |
| PMC-Treatment | 0.4571 | 0.4414 | -0.0157 |
| PMC-Clinical | 0.2627 | 0.2627 | -0.0001 |
| IIYi-Clinical | 0.1796 | 0.2017 | +0.0221 |

Thus only 1/3 subsets is non-inferior, and the overall gain is below the frozen
+0.005 gate. DualSource+BGE does not beat LameR+BGE, so AAR was not run. The
DEV gate is **NEGATIVE**; no final method lock was created, and TEST was not
opened. Per the stop rule, R2MED method development stops here.

The unexpectedly weak BGE reranking result was checked against the pinned
model's official Transformers example: the implementation uses the documented
pair tokenizer, 512-token truncation, and raw sequence-classification logit.
A synthetic sanity check returned `[-8.1797, 5.2617]` for the model card's
irrelevant/relevant panda pair (its example reports `[-8.1875, 5.2617]`),
confirming score direction and model loading. This supports reporting the
clinical DEV result as a negative transfer result for this frozen setup; it
does not authorize post-hoc reranker changes. See the
[pinned BGE reranker model card](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e)
and [FlagEmbedding's inference documentation](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/inference/reranker/README.md).

The optional ReasonRank arm was skipped on resource grounds. The official
`reasonrank-7B` Hugging Face card identifies an 8B-parameter BF16 checkpoint,
and the authors' inference script requests four GPUs, so it is not a viable
official-recipe inference on this single 16-GB GPU without quantization or a
new runtime. Neither was authorized by the frozen protocol. See the
[official ReasonRank repository](https://github.com/8421BCD/ReasonRank) and
[checkpoint card](https://huggingface.co/liuwenhan/reasonrank-7B).

```text
RERANKER_BASELINE_INFERENCE = COMPLETED
CANDIDATE_COMPLEMENTARITY = POSITIVE
DUALSOURCE_SIGNAL = NEGATIVE_BY_FROZEN_GATE
AAR_ELIGIBLE = NO
DEV_SIGNAL = NEGATIVE
STRONG_DEV_SIGNAL = NO
FINAL_METHOD_FROZEN = NO
TEST_RUN = NO
PUBLIC_BENCHMARK_IMPROVEMENT = NO
RESUME_HEADLINE_READY = NO
REASONRANK_SKIPPED_RESOURCE_CONSTRAINT = YES
```
