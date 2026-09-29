# RAG-E5 External Corpus Eligibility Plan

## Current state

The active reviewed source catalog contains 30 source records, all mapped to
`public_health`. Its current catalog identity is
`7cef21fceb5c04577fed2541dbefd2798900299825b711ab06c8c14e41b29c19`. The
existing E2 capability contract makes `PUBLIC_HEALTH` eligible, while
`GUIDELINE` is `NOT_YET_ELIGIBLE` because no reviewed guideline/recommendation
corpus is connected to the active retriever. `LITERATURE` remains out of scope.

Broad patient-education pages that mention lifestyle advice are not reclassified
as clinical guidelines. The current
[`guideline_source_manifest.json`](guideline_source_manifest.json) contains
three hash-pinned WHO candidates, all `READY_FOR_OWNER_REVIEW` with owner status
`PENDING`. They are not active guideline evidence and are not retrievable.

## Admission checklist

Before a source can become `reviewed_guideline`, the E5 corpus owner must:

1. Select an official public guideline/recommendation source and record the
   publisher, jurisdiction, intended population, title, publication/effective
   date, and version.
2. Preserve the retrieved source bytes and compute a SHA-256 document hash;
   record retrieval timestamp and stable source URL/identifier.
3. Complete human review for authority, scope, currency, and allowed use; store
   reviewer identity/status and review date in the manifest.
4. After owner approval, ingest the approved document into the active retrieval
   index, with frozen extraction/chunking metadata and a source-family mapping
   of `reviewed_guideline`.
5. Verify retrieval provenance and family filtering using synthetic tests, then
   freeze the corpus, chunks, source catalog, BM25/BGE index identities, and
   RRF profiles **before** any counterfactual run.

No random web page, search snippet, LLM-generated text, unreviewed aggregator,
or raw paper corpus qualifies. E5-A2 has downloaded and hash-checked candidate
WHO PDFs, but has not admitted, chunked, or indexed them. See the
[`E5-A2 review packet`](e5a2_guideline_review_packet.md) and
[`corpus qualification`](e5a2_guideline_corpus.md).

## Gate consequence

`GUIDELINE_CAPABILITY_ELIGIBLE = NO` blocks the T2 integration study and all
E5-B outcomes/oracle calculations. After the owner records explicit decisions,
rerun source-hash validation, recommendation retention, deterministic chunking,
provenance smoke retrieval, index binding, and capability audits. Only then may
the corpus identity be bound and E5-A be re-frozen. Do not patch a missing
source into a running or scored experiment.
