# RAG Line Closeout

> Scope: resume evidence engineering, not clinical validation. MIRAGE answer evaluation is exposed exploratory; the learned router uses OOF on that exposed case set.

## External baseline reproduction

### NFCorpus

On NFCorpus test (323 queries; 3,633 documents), BM25 Recall@10/MRR/nDCG@10 = 0.152/0.518/0.311; BGE dense = 0.103/0.409/0.236; hybrid RRF nDCG@10 = 0.309; cross-encoder hybrid nDCG@10 = 0.309. Dense underperformed BM25 on this corpus; the hybrid did not improve nDCG over BM25.

### R2MED

The pinned R2MED TEST contains 303 queries across three subsets. DEV selected BGE dense as the strongest fixed baseline. The RRF + MedCPT rerank method scored macro nDCG@10 0.117 vs BGE dense 0.152 (Δ -0.035; paired 95% CI [-0.060, -0.010]); the frozen headline gate failed. BM25/MedCPT/BGE parity and canonical-vs-R2MED MedCPT wiring are documented in `docs/research/e1_2_r2med_parity_audit.md`.

## Medical QA — MIRAGE (exposed exploratory, n=5,235)

| Arm | Accuracy | Retrieval calls | Input tokens |
| --- | ---: | ---: | ---: |
| Closed book | 61.99% | 0 | 1,127,913 |
| Random context | 60.52% | 5,235 | 6,667,133 |
| BM25 | 62.02% | 5,235 | 6,642,084 |
| MedCPT | 62.45% | 5,235 | 6,167,268 |
| Cheap router | 62.23% | 2,311 | 3,640,366 |
| Jev router | 62.35% | 4,908 | 5,876,822 |

The fixed-arm accuracy range is narrow: closed book 61.99%, BM25 62.02%, MedCPT 62.45%. Random context scored 60.52%; this does not support a generic claim that adding context helps. MedCPT is the best fixed arm, but its gain over closed book is only 0.46 pp while using 447% more answer input tokens.

## Adaptive retrieval and question-only OOF router

| Policy | Accuracy | Retrieval calls | Input tokens |
| --- | ---: | ---: | ---: |
| Cheap router | 62.23% | 2,311 | 3,640,366 |
| Jev router | 62.35% | 4,908 | 5,876,822 |
| TF-IDF direct | 62.04% | 108 | 1,237,336 |
| TF-IDF hierarchical | 61.80% | 1,345 | 2,493,763 |
| TF-IDF + always MedCPT | 61.89% | 1,345 | 2,420,647 |
| BGE hierarchical | 61.87% | 2,493 | 3,626,709 |
| BGE + always MedCPT | 61.99% | 2,493 | 3,535,311 |
| Cost oracle v2 | 67.56% | 292 | 1,425,551 |

The q-only post-hoc cost oracle achieves 67.56% at 292 calls. This is an exposed upper bound, not an executable router. The best learned OOF policy is `tfidf_direct` at 62.04%; its recovered oracle headroom is 1.0%.
Oracle-v2 is +5.578 pp over Closed and +5.119 pp over the best fixed arm. The best learned policy recovers 1.0% of Closed-to-oracle headroom and -3.6% of Cheap-to-oracle headroom.
Observed Pareto frontier (including the non-executable oracle): Closed only, Cost oracle v2, TF-IDF direct.

## Negative and limiting results retained

- Dense retrieval is not universally better: on NFCorpus BM25 beats BGE dense; on R2MED, BGE dense beats BM25, while MedCPT dense is weak on the pinned R2MED wiring.
- Canonical MedCPT and R2MED ARTICLE_ARTICLE use different query/document wiring; the parity audit explains the gap rather than treating these as interchangeable.
- On MIRAGE, always-RAG accuracy gains over closed-book are small relative to retrieval calls and answer-token cost; random context can reduce accuracy.
- Jev routing spends almost as many retrieval calls as always-RAG for little accuracy gain; no Jev API was called in this sprint.
- The learned question-only routers and their exact OOF deltas are reported as exploratory. The post-hoc oracle headroom is not evidence that a deployable router can recover it.
- Historical PubMedQA/BioASQ results and the earlier Harness RAG abstention-heavy result remain historical exposed artifacts, not rerun or mixed into the 5,235-case primary comparison.

## Resume evidence candidates

- No accuracy/cost headline is supported by the frozen gates. A factual engineering-only line may describe the reproducible R2MED parity audit and the question-only OOF evaluation harness, but should not imply a measured positive routing gain.
- The TF-IDF-direct point estimate is a cost/quality tradeoff, not a Pareto win: 95.3% fewer calls and 66.0% fewer measured input tokens vs Cheap, for -0.191 pp accuracy (best-learned paired exploratory 95% CI [-0.669, +0.287] pp). Treat as a follow-up hypothesis only; the interval spans zero and there is no external validation.

Do not describe these results as clinical accuracy, clinical validation, an untouched holdout, or state of the art. Do not train SFT/GRPO from this sprint; use a future separately frozen, genuinely unseen evaluation set if pursuing post-training.

## Final status

- `RAG_BASELINES_COMPLETE = YES`
- `OOF_PROTOCOL_VALID = YES`
- `COST_ORACLE_V2_READY = YES`
- `TFIDF_SIGNAL_FOUND = YES`
- `BGE_SIGNAL_FOUND = YES`
- `BGE_BEATS_TFIDF = NO`
- `LEARNED_ROUTER_PARETO_DOMINATES_CHEAP = NO`
- `STRONG_CAPABILITY_ROUTING_SIGNAL = NO`
- `SFT_CANDIDATE = NO`
- `RAG_CLOSEOUT = YES`
