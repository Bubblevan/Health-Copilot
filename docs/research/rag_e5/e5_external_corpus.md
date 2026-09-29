# RAG-E5 External Corpus Eligibility Plan

## Current state

The active reviewed source catalog contains 30 source records, all mapped to
`public_health`. Its current catalog identity is
`7cef21fceb5c04577fed2541dbefd2798900299825b711ab06c8c14e41b29c19`. The
existing E2 capability contract makes `PUBLIC_HEALTH` eligible, while
`GUIDELINE` is `NOT_YET_ELIGIBLE` because no reviewed guideline/recommendation
corpus is connected to the active retriever. `LITERATURE` remains out of scope.

Broad patient-education pages that mention lifestyle advice are not reclassified
as clinical guidelines. The empty
[`guideline_source_manifest.json`](guideline_source_manifest.json) is an
eligibility template, not evidence that a guideline corpus exists.

## Admission checklist

Before a source can become `reviewed_guideline`, the E5 corpus owner must:

1. Select an official public guideline/recommendation source and record the
   publisher, jurisdiction, intended population, title, publication/effective
   date, and version.
2. Preserve the retrieved source bytes and compute a SHA-256 document hash;
   record retrieval timestamp and stable source URL/identifier.
3. Complete human review for authority, scope, currency, and allowed use; store
   reviewer identity/status and review date in the manifest.
4. Ingest the approved document into the active retrieval index, with frozen
   extraction/chunking metadata and a source-family mapping of
   `reviewed_guideline`.
5. Verify retrieval provenance and family filtering using synthetic tests, then
   freeze the corpus, chunks, source catalog, BM25/BGE index identities, and
   RRF profiles **before** any counterfactual run.

No random web page, search snippet, LLM-generated text, unreviewed aggregator,
or raw paper corpus qualifies. E5-A does not download or ingest sources.

## Gate consequence

`GUIDELINE_CAPABILITY_ELIGIBLE = NO` blocks the T2 integration study and all
E5-B outcomes/oracle calculations. Once an eligible corpus has been assembled,
the proper next action is to rerun the eligibility and source-family audits,
update this manifest through human review, and re-freeze E5-A. Do not patch a
missing source into a running or scored experiment.
