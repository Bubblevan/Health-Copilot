# E1.2 baseline-source audit

Audit date: 2026-09-24. No upstream source or dataset was copied into the Git
repository. The independent source checkouts and any model/cache payloads belong
on the dedicated external scratch volume described in
[`storage_plan.json`](../../runs/e1_2/storage_plan.json).

## Identity and selection decisions

| Source | Pinned identity checked | License / data boundary | E1.2 disposition |
| --- | --- | --- | --- |
| [Medical MIRAGE](https://github.com/gzxiong/MIRAGE) | `392943af99cd94cafd50a0de2e7fca24bbf65494`; E0 raw SHA `6f7f08c64cd2efe02a5d0c247229813c90db345d9dd6e3a451b5d24146d0f8fa`; normalized SHA `3a31d3e7fbe6b1184afdbd46925db28747e2ec60d3e53948d8b7ce8ecd614923`; 7,663 cases | E0 admissibility approved; component dataset terms remain distinct and question text/gold will not be committed | Primary external QA benchmark. Correctly identified as **Medical Information Retrieval-Augmented Generation Evaluation**, not the unrelated metric-intensive or misinformation-defense projects also called MIRAGE. |
| [MedRAG](https://github.com/gzxiong/MedRAG) | `7599a728a28789fd601728c08d313b1148051f41` | Repository `LICENSE` returned an NCBI public-domain notice; this is not treated as blanket permission for every corpus. Existing local Textbooks corpus is reused without copying. | Reuse as corpus/retriever reference; no large MedCorp/PubMed download. |
| [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) | `e82f680b5a3fce1349bf818c249857d14b83516a` | Repository states MIT; individual dataset terms still apply | Framework/baseline reference only; its broad dependency stack is not installed for this sprint. |
| [Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG) | `0c88670af8707667eb5c1163151bb5ce61b14acb` | Repository states Apache-2.0; its benchmark/data terms remain separate | Conceptual reference for no-retrieval vs retrieval selection. The original runner expects Elasticsearch and non-medical datasets, so it is not used as the medical baseline. |
| [Clinical RAG Retrieval Benchmark](https://github.com/yngvemikkelsen/clinical-rag-retrieval-benchmark) | `7654241a1056e1da3baa076524cbbc9130ff28cf` | Code MIT. README assigns separate terms to MTSamples and PMC-Patients; only its synthetic corpus is clearly generated for the study. | **Rejected as a primary retrieval score source** after code audit (details below). No corpus or result file is downloaded. |
| [Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) | Official Qwen publisher; revision `7c41481f57cb95916b40956ab2f0b139b296d974`; file `Qwen3-8B-Q4_K_M.gguf`, 5,027,783,488 bytes, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`; model card states Apache-2.0 | Model only; no question data sent to a hosted generator if selected | Candidate local answer model for closed-book DEV selection. Fetch only this pinned file to the external scratch root. |

The user-provided `data/mirage/benchmark.json` has SHA-256
`9f07ffab4822513906d55ec8833607c94cbf86dd6f0810a20c12923f1a51c1e1`, which does
not match the pinned E0 MIRAGE source. It is not used. The E0 raw and normalized
artifacts are already present locally and their identities match the frozen
manifest, so they are reused in place.

Before any answer generation, the local llama.cpp `GET /v1/models` preflight
reported the served model ID as the full path
`F:\Health-Copilot-E1.2\models\Qwen3-8B-Q4_K_M.gguf`, while the API request name
remains `Qwen3-8B-Q4_K_M.gguf`. The frozen protocol records both identities and
checks the server-reported ID on every answer response; this is an identifier
alignment only, not a candidate or weight change. The pinned SHA-256 remains
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.

## Clinical retrieval benchmark protocol audit

The pinned benchmark's README describes three 500-document corpora, two query
formats, multiple embedding configurations and reported MRR@10. Its checked-in
`clinical_rag_benchmark_v3.py` does not provide independent query/document
relevance labels or a train/dev/test split. Instead, it builds each query from
the same note at index `i`, compares it against the full note matrix, and treats
document `i` as the single relevant item (`scores[i]`). PMC and synthetic query
text is extracted directly from that target note; the MTSamples query fields are
also built from the target row. This is a self-retrieval proxy, not an
independently judged clinical retrieval benchmark, and can inflate or distort
MRR/Recall. The README's stated `13 × 3 × 2 × 4 = 294` condition count is also
not arithmetically consistent (the product is 312), while the inspected runner
does not implement a four-way chunking loop. These are material protocol gaps,
not tuning opportunities. Therefore its published scores will not be used as an
E1.2 baseline or resume claim.

The existing E1.1 NFCorpus result remains a valid graded-qrels retrieval result,
but its 323-query TEST aggregate and per-query artifacts have already been
inspected. It is historical context, not a pristine E1.2 holdout and will not be
retuned. E1.2's clean QA holdout will instead be constructed only from MIRAGE
subsets whose cases were not included in the prior 1,118-case PubMedQA/BioASQ
experiments; those two previously evaluated subsets will be labeled as exposed
development/history and shown separately in the five-subdataset report.

## Track A: independent retrieval benchmark

The [R2MED benchmark](https://github.com/R2MED/R2MED) provides independently
judged query-to-document relevance labels and reports nDCG@10. Its Hugging Face
dataset cards label the packaged subsets CC-BY-4.0. Separately, the R2MED paper's
[Table 8](https://openreview.net/pdf/b1cd41af38a3494706a923e6de10affdcc496356.pdf)
lists source-query terms: PMC-Clinical is CC-BY-NC-SA 4.0, while PMC-Treatment
and IIYi-Clinical are not specified in that table. The project records this
provenance discrepancy rather than treating package-level metadata as overriding
source terms. The user approved proceeding with this disclosure on 2026-09-24.
Raw benchmark content and derived indexes, embeddings, and ranked lists remain
outside Git; only aggregate metrics and attribution are intended for the project
record. Exact source revisions and per-file SHA-256 identities are recorded in
[`r2med_source_manifest.json`](../../runs/e1_2/r2med_source_manifest.json).
The small public benchmark comprises multiple distinct query/corpus tasks, so
we use dataset-level separation rather than mixing records from each task:

- DEV only: `PMC-Treatment`, `PMC-Clinical`, and `IIYi-Clinical` (393 queries).
- Frozen TEST only: `MedQA-Diag`, `MedXpertQA-Exam`, and `Medical-Sciences`
  (303 queries). Their query/qrels records were not opened, printed, or used to
  select methods; only file hashes, sizes, and line counts were checked.
- `Biology` and `Bioinformatics` are not used in this sprint.

Correction to the exposure log: while checking JSONL schemas, one first-record
query/qrels/corpus excerpt from each of the three DEV subsets was emitted in the
working context. This is acceptable only for DEV and is why those subsets are
never counted as the confirmatory holdout. The three TEST subsets remain
uninspected at the record level. No raw R2MED queries, qrels, or document text
are committed to the repository.

The retrieval baselines are fixed as SQLite FTS5 BM25, BAAI BGE-base-en-v1.5,
and MedCPT bi-encoder. The proposed method is rank fusion of BM25 and MedCPT
top-100 lists followed by the MedCPT cross-encoder. BGE's published retrieval
query prefix is applied only to the BGE query encoder; all document strings
remain unprefixed. This local FTS5 implementation is explicitly not described
as an exact reproduction of R2MED's Pyserini BM25 implementation. The strongest
fixed baseline is selected on DEV; TEST compares that frozen baseline with all
predeclared arms exactly once, using macro nDCG@10 across the three subsets as
the primary score and paired subset-stratified bootstrap intervals. No positive
headline is permitted unless the custom method beats the DEV-selected fixed
baseline on TEST and the paired 95% interval excludes zero.

The DEV source audit found 145 repeated rows across 142 IDs in
`PMC-Treatment`; every repeated ID had exactly identical retrieval text. The
runner retains the first copy, reports raw/unique/removed counts, and still
aborts if any repeated ID has different text. The other two DEV corpora had no
duplicate IDs. This deduplication affects only identical source rows and is
part of the pinned runner implementation.

The original `hfd.sh` attempt could not retrieve the Hugging Face metadata API
(HTTP status `000`); it did not download dataset payloads. The selected public
files were instead fetched from the same pinned Hugging Face Git/LFS revisions.
The BGE weight SHA-256 matches the official model file page. Its official model
card identifies MIT licensing and recommends the retrieval query prefix used
here.

## Baseline interpretation

- `closed_book`, BM25 `RAG_STANDARD`, and MedCPT `RAG_STRONG` use the same pinned
  generator and question-only input. Answer options and gold are never supplied
  to retrieval or routing.
- E1.1's NFCorpus improvements remain modest and its cross-encoder arm did not
  improve its frozen test metric. They are not presented as an E1.2 win.
- The candidate's DEV-only model choice and routing thresholds will be frozen
  before touching the new MIRAGE holdout. All TEST data remain one-shot and
  oracle routing remains analysis-only.
