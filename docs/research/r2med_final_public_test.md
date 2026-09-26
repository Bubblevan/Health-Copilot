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

The exact aggregate report is `runs/rag_r2med_final_test/test_report.json`; post-hoc pool diagnostics are in `runs/rag_r2med_final_test/candidate_analysis.json`. All values below are equal-weight means over the three subsets, not query-count-weighted micro averages.

| Method | MedQA-Diag nDCG@10 | MedXpertQA-Exam nDCG@10 | Medical-Sciences nDCG@10 | Macro nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.0255 | 0.0066 | 0.1968 | 0.0763 |
| BGE-large | 0.0833 | 0.0410 | 0.2781 | 0.1341 |
| BM25+BGE RRF | 0.0811 | 0.0257 | 0.3109 | 0.1392 |
| LameR-MV | **0.1655** | **0.0980** | **0.4039** | **0.2225** |
| Compact CRB-Q | 0.1376 | 0.0766 | 0.3841 | 0.1995 |
| DualSource-RRF λ=.5 | 0.1510 | 0.0894 | 0.4023 | 0.2142 |

| Method | MRR@10 | Recall@5 | Recall@10 | Recall@50 | Recall@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.0798 | 0.0714 | 0.1137 | 0.2442 | 0.2973 |
| BGE-large | 0.1347 | 0.1441 | 0.2029 | 0.3390 | 0.4275 |
| BM25+BGE RRF | 0.1472 | 0.1392 | 0.2010 | 0.3530 | 0.4157 |
| LameR-MV | **0.2406** | **0.2209** | **0.2971** | **0.4971** | 0.5699 |
| Compact CRB-Q | 0.2205 | 0.1980 | 0.2752 | 0.4517 | 0.5340 |
| DualSource-RRF λ=.5 | 0.2328 | 0.2064 | 0.2904 | 0.4785 | **0.5791** |

### Paired comparisons

Intervals are the pre-specified 10,000-resample, subset-stratified paired bootstrap with seed 20260926. They are exploratory uncertainty intervals on a public/reused TEST split, not confirmatory significance tests.

| Comparison | Δ macro nDCG@10 | Exploratory 95% CI | Relative change |
| --- | ---: | ---: | ---: |
| DualSource − BM25 | +0.13794 | [ +0.11446, +0.16316 ] | +180.76% |
| DualSource − BM25+BGE RRF | +0.07502 | [ +0.05825, +0.09232 ] | +53.88% |
| DualSource − LameR-MV | −0.00823 | [ −0.01974, +0.00318 ] | −3.70% |
| LameR-MV − BM25+BGE RRF | +0.08324 | [ +0.06397, +0.10305 ] | +59.79% |

### Generation audit

| Frozen generator arm | Calls | Valid | Fallback | Truncated | Paid API |
| --- | ---: | ---: | ---: | ---: | ---: |
| LameR | 303 | 303 (100%) | 0 | 16 | 0 |
| Compact CRB-Q | 303 | 296 (97.69%) | 7 | 0 | 0 |

Each call was local, one per query, with no retries. The LameR manifest's `completed_count` is 287 because 16 outputs reached the 256-token limit; those rows remain valid generated artifacts and were used as-is. Compact CRB's seven invalid structured outputs followed the frozen original-query fallback. No repair was made on TEST.

### Candidate complementarity

| Candidate/ranking view | Equal-subset Recall@100 |
| --- | ---: |
| LameR-MV candidates | 0.56987 |
| Compact CRB-Q candidates | 0.53399 |
| Raw union of both top-100 pools | 0.62045 |
| Final DualSource ranked top-100 | 0.57911 |

Across query-relevant document pairs, 95 were found only in LameR's pool, 62 only in CRB's pool, 430 in both, and 471 in neither. Thus CRB supplies real complementary candidates, and DualSource raises ranked Recall@100 by 0.00924 over LameR, but that complementarity did not translate into higher nDCG@10. The raw union figure is a pool ceiling, not a final ranking score.

## Interpretation and resume claims

The report applies three distinct gates rather than collapsing them into a single success claim:

- Basic public-baseline improvement requires DualSource to exceed both BM25 and ordinary hybrid RRF, with the exploratory paired 95% interval lower bounds above zero for both comparisons.
- Strong basic-baseline improvement additionally requires at least +0.020 absolute macro nDCG@10 over ordinary RRF. This is an internal resume gate, not a field-wide threshold.
- Strongest-GAR improvement requires DualSource to exceed LameR-MV by at least +0.005, with a paired 95% interval lower bound above zero.

Only claims supported by the corresponding gate may be used. Even a positive result does not establish SOTA, clinical superiority, clinical validation, or performance on an unseen test. If a basic gate passes, the resume bullet may say that the frozen pipeline improved R2MED public TEST retrieval over the named basic baseline, reporting the exact split and absolute delta. If the strong-method gate fails, do not claim that DualSource beat the strongest reproduced GAR method. The full mixed/negative result remains in the project record.

| Final gate | Result |
| --- | --- |
| FINAL_EVAL_LOCKED | YES |
| TEST_EXECUTED_ONCE | YES — one generation/retrieval run and one qrels scoring pass from frozen rankings |
| TEST_CONFIG_DRIFT | NO |
| PUBLIC_BASIC_BASELINE_IMPROVEMENT | YES |
| STRONG_BASIC_BASELINE_IMPROVEMENT | YES |
| POINT_IMPROVEMENT_OVER_LAMER | NO |
| STRONG_BASELINE_IMPROVEMENT | NO |
| PUBLIC_R2MED_TEST_EVIDENCE_READY | YES |
| RESUME_BASIC_BASELINE_HEADLINE_READY | YES |
| RESUME_STRONG_METHOD_HEADLINE_READY | NO |
| R2MED_FINAL_CLOSEOUT | YES |

### Resume evidence candidate

> On the reused public R2MED TEST split (303 queries), a frozen generation-augmented retrieval pipeline improved equal-subset macro nDCG@10 from 0.1392 with BM25+BGE RRF to 0.2142 with a fixed LameR/Compact-CRB DualSource fusion (+0.0750 absolute; exploratory paired 95% CI [+0.0582, +0.0923]). It did not outperform LameR-MV (0.2225).

This is a factual candidate sentence, not an automatic résumé edit. It supports the basic-baseline claim only; do not describe the fusion as a novel method or as outperforming strongest GAR.

The local scoring environment used the existing project `.venv` with Pyserini 0.44.0, PyJNIus 1.7.0, and Gensim 4.4.0. These dependencies were installed after the initial locked run stopped at BM25 setup; no generation or ranking had occurred before that setup was corrected.

## Closeout

After the one frozen public TEST evaluation and report commit, all R2MED method development stops. No lambda, prompt, schema, RRF, embedding, retriever, or reranker follow-up is authorized by this sprint.
