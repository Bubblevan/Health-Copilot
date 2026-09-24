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

## Baseline interpretation

- `closed_book`, BM25 `RAG_STANDARD`, and MedCPT `RAG_STRONG` use the same pinned
  generator and question-only input. Answer options and gold are never supplied
  to retrieval or routing.
- E1.1's NFCorpus improvements remain modest and its cross-encoder arm did not
  improve its frozen test metric. They are not presented as an E1.2 win.
- The candidate's DEV-only model choice and routing thresholds will be frozen
  before touching the new MIRAGE holdout. All TEST data remain one-shot and
  oracle routing remains analysis-only.
