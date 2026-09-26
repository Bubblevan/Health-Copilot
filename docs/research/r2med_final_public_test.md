# R2MED Frozen Public TEST Evaluation

## Purpose and claim boundary

This is a frozen public TEST evaluation of six retrieval pipelines. The dataset has previously been accessed for earlier baselines, so the split is explicitly `PUBLIC_BENCHMARK_REUSED`; it is not an untouched holdout or confirmatory test. The earlier DEV gate for a new-method claim was negative: DualSource-RRF λ=.5 reached 0.301918 macro nDCG@10 versus 0.299800 for LameR-MV, a +0.002117 difference. The TEST run therefore asks only whether the frozen pipelines improve over basic retrieval baselines. It does not assume that DualSource is a novel method or that it beats the strongest GAR method.

## Benchmark and split

The source is R2MED/R2MED pinned at commit `11244a4925a39082967a6c9d38ef01f279c316a5`. DEV contains PMC-Treatment (150), PMC-Clinical (114), and IIYi-Clinical (129). This final evaluation uses MedQA-Diag (118), MedXpertQA-Exam (97), and Medical-Sciences (88), 303 queries total. Source revisions and query/corpus SHA-256 identities are recorded in `runs/rag_r2med_final_test/final_eval_lock.json`.

## Frozen systems and attribution

| Arm | Description | Attribution |
| --- | --- | --- |
| BM25 | Original query; Lucene BM25, k1=.9, b=.4 | R2MED-compatible sparse baseline |
| BGE-large | Original query; pinned `BAAI/bge-large-en-v1.5` | Public embedding model |
| BM25+BGE RRF | Top-100 from each; RRF k=60, weights [1,1] | Standard rank fusion |
| LameR-MV | Top-10 BM25 feedback to the pinned upstream LameR prompt; four-view RRF k=20, weights [1,2,1,2] | Public generation-augmented retrieval adapted to this runtime; not Health-Copilot-invented |
| Compact CRB-Q | Frozen compact q/t/e bridge; four-view RRF k=20, weights [1,1,1,1] | Health-Copilot experimental structured query bridge |
| DualSource-RRF λ=.5 | Frozen fusion of LameR-MV and Compact CRB-Q rankings, k=60 | Health-Copilot adaptation combining two frozen ranking sources |

The generator is local Qwen3-8B Q4_K_M, one call per query per generated arm, temperature 0, reasoning disabled, and 256 output tokens. Invalid compact output uses the frozen original-query fallback without retry. Expected generation volume is 303 LameR plus 303 compact CRB calls; paid API calls are zero. No reranker is in scope.

## Data boundary and execution

Generation and retrieval receive only native query text, corpus text, and (for LameR) the same query's top-10 BM25 passages. They do not read qrels, relevance labels, gold documents, or answer keys. The six complete rankings are persisted to the E: artifact root and SHA-256 frozen before evaluation. `eval/r2med_crb_evaluator.py` remains the only qrels/relevance consumer. Infrastructure interruption may resume unfinished generation query IDs; a completed query ID is never generated again.

### Pre-scoring validation erratum

The first locked runner pass completed all six ranking artifacts and froze their hashes, then stopped before the evaluator opened qrels. Its final generation-budget guard compared bare `query_id` values globally across all three subsets. R2MED query IDs are numeric and subset-local: each artifact had the exact locked subset count, unique IDs within that subset, matching query order, and matching generation/ranking hashes, but nine IDs legitimately appeared in more than one subset. The failed guard therefore rejected complete artifacts due to an identity-scope bug, not missing generations.

The scoring continuation records this erratum in `runs/rag_r2med_final_test/evaluation_preflight_erratum.json`. It revalidates the committed lock, generation manifests/artifact hashes, the six frozen ranking hashes, and exact IDs/order per subset before scoring. It applies the uniqueness invariant to `(subset, query_id)` and calls the original locked scoring/evaluator code; no generation, retrieval, ranking, model, metric, or method setting is changed. The erratum is committed before qrels are opened. The first pass did not reach qrels access.

## Metrics and uncertainty

Primary metric is equal-subset macro nDCG@10: compute mean nDCG@10 within each TEST subset, then take the unweighted mean of the three subset means. Secondary metrics are MRR@10 and Recall@5/10/50/100. Paired bootstrap uses 10,000 resamples, stratified by subset, seed 20260926. Intervals are exploratory because this is public/reused TEST; they are not confirmatory significance tests, and no multiplicity claim is made.

Comparisons are DualSource versus BM25, ordinary BM25+BGE RRF, and LameR-MV. The LameR-MV versus ordinary RRF comparison separates GAR gains from DualSource's incremental contribution. Candidate complementarity reports each source's Recall@100, raw union-pool Recall@100, ranked DualSource Recall@100, and relevant documents unique to either source. Raw union Recall@100 is a candidate-pool ceiling, not a final ranking metric, and is post-hoc interpretation only.

## Results

Results will be filled from the committed `runs/rag_r2med_final_test/test_report.json` and `candidate_analysis.json` after the frozen TEST ranking evaluation. No TEST-derived choice of prompt, schema, weight, lambda, fusion, retriever, or reranker is permitted.

| Method | MedQA-Diag nDCG@10 | MedXpertQA-Exam nDCG@10 | Medical-Sciences nDCG@10 | Macro nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | pending | pending | pending | pending |
| BGE-large | pending | pending | pending | pending |
| BM25+BGE RRF | pending | pending | pending | pending |
| LameR-MV | pending | pending | pending | pending |
| Compact CRB-Q | pending | pending | pending | pending |
| DualSource-RRF λ=.5 | pending | pending | pending | pending |

## Interpretation and resume claims

The report applies three distinct gates rather than collapsing them into a single success claim:

- Basic public-baseline improvement requires DualSource to exceed both BM25 and ordinary hybrid RRF, with the exploratory paired 95% interval lower bounds above zero for both comparisons.
- Strong basic-baseline improvement additionally requires at least +0.020 absolute macro nDCG@10 over ordinary RRF. This is an internal resume gate, not a field-wide threshold.
- Strongest-GAR improvement requires DualSource to exceed LameR-MV by at least +0.005, with a paired 95% interval lower bound above zero.

Only claims supported by the corresponding gate may be used. Even a positive result does not establish SOTA, clinical superiority, clinical validation, or performance on an unseen test. If a basic gate passes, the resume bullet may say that the frozen pipeline improved R2MED public TEST retrieval over the named basic baseline, reporting the exact split and absolute delta. If the strong-method gate fails, do not claim that DualSource beat the strongest reproduced GAR method. The full mixed/negative result remains in the project record.

## Closeout

After the one frozen public TEST evaluation and report commit, all R2MED method development stops. No lambda, prompt, schema, RRF, embedding, retriever, or reranker follow-up is authorized by this sprint.
