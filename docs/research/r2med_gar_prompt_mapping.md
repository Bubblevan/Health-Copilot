# R2MED GAR Prompt-Family Mapping

This sprint uses the prompt families from the pinned `R2MED/R2MED` checkout at
`11244a4925a39082967a6c9d38ef01f279c316a5`. Dataset labels in the local R2MED
split are not identical to the prompt keys in that upstream checkout, so the
mapping below is an explicit naming adaptation based on the dataset/task family
and the upstream prompt examples/instructions.

This is **not a byte-for-byte official reproduction**. The retrieval protocol,
generator, output budget, and isolation boundaries are specified separately in
`r2med_crb.md`; the map here selects the corresponding upstream prompt family.

| Local R2MED subset | Pinned upstream prompt key | Mapping rationale |
| --- | --- | --- |
| `PMC-Treatment` | `PMC-Treat` | PMC clinical treatment questions use the upstream clinical-case-to-relevant-passage family. |
| `PMC-Clinical` | `PMCPatients` | Patient/case retrieval uses the upstream similar-clinical-case family. |
| `IIYi-Clinical` | `IIYiPatients-EN` | English IIYi patient cases use the upstream English patient-case family. |
| `MedQA-Diag` | `MedQA-Diag` | Exact task-family name match. |
| `MedXpertQA-Exam` | `MedXpertQA-Exam` | Exact task-family name match. |
| `Medical-Sciences` | `Stack-Medical` | Medical-science posts use the upstream medical-science post family. |

The upstream prompt source and few-shot examples are loaded only from the
verified pinned checkout. The run manifest records the commit and SHA-256 of
`src/eval_BM25.py`, `src/eval_retrieval.py`,
`src/generate_hypothetical_doc.py`, `src/instrcution.py`, and
`src/example.py`. Any commit or file-hash mismatch must stop the run before
generation. Query2Doc retains the upstream examples; they are not dropped or
replaced.

The input is the R2MED `query.jsonl` native `text` field. We do not separately
load answer keys, answer options, qrels, relevance labels, or gold document IDs
for prompt construction. If the public query text itself contains options,
those remain part of the native query text.

## Corpus ID behavior verified against upstream

The pinned dense loader materializes `corpus.jsonl` as a dictionary keyed by
document ID, so repeated IDs collapse to one entry (the last text value, while
the original insertion position is retained). The pinned BM25 loader keeps all
corpus rows while building its index, then stores similarity scores in a
document-ID-keyed dictionary; repeated IDs therefore update the score but do
not create duplicate ranked IDs. The sprint loader preserves every source row
for BM25 and exposes a first-occurrence unique-ID view for dense encoding. It
rejects repeated IDs with conflicting text instead of silently changing the
upstream semantics. The frozen DEV PMC-Treatment file has 28,954 rows and
28,809 unique IDs; all 145 repeats have identical text.
