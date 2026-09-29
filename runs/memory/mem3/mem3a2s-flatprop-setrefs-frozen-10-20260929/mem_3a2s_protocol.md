# MEM-3A.2S Evidence-Ref Set Canonicalization

Status: protocol amendment and frozen-ten diagnostic only.

## Scope

MEM-3A.2S continues the MEM-3A.2R 16K FlatProp extraction from the benchmark-routing history at `5f666b241fefc0dbcd7524bd4ea6c85bbeaa6bb9` while preserving the later mechanism commit `73332535a7aef74d6a93d39092f5f596d00ec767` as an ancestor. Historical artifacts remain immutable.

The only validator relaxation is exact duplicate `evidence_refs` within one proposition. The field is logically a set, transported as a JSON array. Every raw ref must first resolve to the current session's frozen RawSpan catalog; unknown, malformed, cross-session, empty, hash-mismatched, offset-mismatched, missing-field, and unexpected-field cases remain fatal. Exact duplicates are then collapsed and ordered by catalog ordinal. No fuzzy matching, model repair, retry, cross-proposition deduplication, or semantic inference is permitted.

The frozen writer prompt and v3 schema remain unchanged, including `uniqueItems: true`. The new packet validator identity binds the v3 extractor contract, this normalization contract, validator source, and RawSpan catalog contract. That identity participates in the normalized-packet cache key.

## Historical Response Recovery

The exact 102 captured MEM-3A.2R outcomes are revalidated from local write-ahead envelopes. They are not replayed through a provider. Session identity, source session, request SHA, envelope SHA, assistant-content SHA, prompt SHA, v3 contract SHA, 16K cap, and finish reason are checked. The known session-102 duplicate response must match its frozen envelope SHA and normalize the repeated `S0070` to one ref. Any additional structural issue stops the stage.

Only after all 102 outcomes pass may the run issue one local Qwen writer request for each remaining source identity, up to 375 calls, with the unchanged 16K cap and zero retries. A `finish_reason=length` stops with `WRITER_UNBOUNDED_OUTPUT_FAILURE`.

## Downstream Diagnostic

After 477 normalized packets freeze, the existing pipeline materializes ADD-only `SESSION_NOTE` / `SESSION_DERIVED` / ACTIVE version-1 memories, uses proposition-text-only local Qwen3-Embedding dense top-8 retrieval, the existing 1024-token rank-aware projection, and exactly ten calls to the frozen shared reader. No judge, hosted provider, 102-case DEV, TEST, MedMemoryBench scoring, ESL ingestion, or revision semantics are used.

Raw/canonical duplicate counts are evaluator metadata only. They do not enter reader-visible MemoryRecord values, embedding documents, or prompts. The frozen-ten results remain diagnostic and stop before MEM-3B0.

## Gates

The benchmark-routing gate remains `MEDMEMORYBENCH_COMPATIBILITY_AUDITED=YES`; the LongMem memory mechanism gate remains `LONGMEM_MEMORY_MECHANISM_FROZEN=NO`. This stage does not change MedMemoryBench audit artifacts or initiate MED-M0.
