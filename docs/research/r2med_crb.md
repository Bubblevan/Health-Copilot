# R2MED Same-Generator GAR + Clinical Reasoning Bridge

## Objective and evidence policy

This sprint evaluates whether a generated clinical-reasoning bridge improves
retrieval on the public R2MED benchmark. The only tuning partition is the frozen
DEV set (PMC-Treatment 150, PMC-Clinical 114, IIYi-Clinical 129). The public
TEST set (MedQA-Diag 118, MedXpertQA-Exam 97, Medical-Sciences 88) has already
been used in earlier work and is labeled `PUBLIC_BENCHMARK_REUSED`; it is not an
untouched confirmatory set. TEST is permitted once only after the DEV gate and
a committed method lock.

The pinned upstream identity is `R2MED/R2MED` commit
`11244a4925a39082967a6c9d38ef01f279c316a5`. The source manifest contains the
SHA-256 identities of the five upstream prompt/retrieval files. Prompt-family
name adaptation is documented in `r2med_gar_prompt_mapping.md`; this work is
not described as byte-for-byte official reproduction.

## Gold boundary

Generation and ranking load only native query text and corpus passages. They
do not load qrels, relevance labels, gold document IDs, answer keys, or a
separate answer field. LameR and CRB-PRF see only the top ten passages from the
original-query upstream-compatible BM25 ranking. The new evaluator is the only
module that opens `qrels.jsonl`.

Corpus duplicate-ID behavior follows the pinned upstream loaders: BM25 indexes
all corpus rows and collapses the scored results by ID, while dense retrieval
uses the unique-ID corpus view. Identical repeated rows are retained for BM25;
conflicting text under one ID is rejected. Source JSONL bytes are never changed.

## Frozen models and generation

- Generator: `Qwen/Qwen3-8B-GGUF`, revision
  `6a569868d07d3bd59e8b97fb001bf8c0b254bb20`,
  `Qwen3-8B-Q4_K_M.gguf`, SHA-256
  `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- Dense retriever: `BAAI/bge-large-en-v1.5`, revision
  `d4aa6901d3a41ba39fb536a557fa166f842b0e09`, weights SHA-256
  `45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7`.
- Generation uses one local llama.cpp call per query/method, temperature 0,
  reasoning disabled, and a 256-token output limit. llama.cpp binds to
  `127.0.0.1` only. Invalid CRB JSON falls back to the original query and is
  counted; there is no retry.
- Query2Doc retains the pinned upstream few-shot examples.

## Retrieval arms

Single-view core matrix: original/BM25, original/BGE-large, HyDE/BM25,
Query2Doc/BM25, LameR/BM25, HyDE/BGE-large, Query2Doc/BGE-large, and
LameR/BGE-large. HyDE dense and LameR dense average the original and generated
query vectors; Query2Doc dense embeds `query[SEP]generated passage`.

Cost-matched multi-view GAR gives HyDE, Query2Doc, LameR, CRB-Q, and CRB-PRF the
same four top-100 channels and the same ten predeclared weighted-RRF configs:

1. BM25(original query)
2. BM25(method-specific expanded query)
3. BGE-large(original query)
4. BGE-large(generated passage)

The grid is `k ∈ {20, 60}` crossed with weights `[1,1,1,1]`, `[2,1,2,1]`,
`[1,2,1,2]`, `[2,2,1,1]`, and `[1,1,2,2]`. Each multi-view method selects
its own DEV configuration by equal-subset macro nDCG@10, then MRR@10,
Recall@10, and the declared simplicity tie-break.

CRB-Q takes the query only. CRB-PRF also receives the same BM25 top-ten
feedback passages used by LameR. Its structured bridge has exactly four
fields: `canonical_query`, up to five `key_concepts`, up to five
`disambiguating_terms`, and a `pseudo_evidence` passage of at most 160 words.
Neither CRB variant answers the benchmark question or receives gold data.

## Metrics and stop rules

Primary metric is the arithmetic mean of the three subset-level nDCG@10 scores
(equal subset weight). Secondary metrics are MRR@10 and Recall@5/10/50/100.
Candidate recall is retained to distinguish candidate-generation problems
from ranking problems.

The DEV gate compares the better of CRB-Q/CRB-PRF with the strongest
cost-matched GAR method. It passes only with delta at least +0.005 macro
nDCG@10 and positive delta in at least two of three subsets. If it fails, run
only the predeclared DEV ablation, do not run TEST, and stop this method family.

If DEV passes, freeze the selected variant/config, commit the lock, then run
TEST once with all fusion settings selected on DEV. Compare CRB primarily with
the strongest frozen cost-matched GAR baseline. The resume-ready gate requires
delta at least +0.010, a positive 95% subset-stratified paired-bootstrap lower
bound (10,000 resamples), and a point score above that baseline. A positive
point estimate with a confidence interval crossing zero is reported as
inconclusive, not as a significant win.

The experiment runner writes large models, embeddings, rankings, and generated
views under `E:\Health-Copilot-RAG`; the repository stores small manifests,
code, and reports that are required to reproduce decisions.
