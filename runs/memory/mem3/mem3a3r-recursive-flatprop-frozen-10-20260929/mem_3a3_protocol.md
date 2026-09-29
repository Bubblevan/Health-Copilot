# MEM-3A.3 Deterministic Chunked FlatProp Writer

## Scope

MEM-3A.3 closes the monolithic-session writer after the observed 4,096- and 16,384-token truncation failures. It keeps the frozen v3 proposition semantics and set-semantic evidence-ref validator, changing only writer execution granularity. The method identity is `flat-proposition-extractor-v4-chunked`.

The full 477-session plan is generated and SHA-frozen before any writer call. Primary chunks greedily preserve chronological complete turns under the 6,144-token request-budget ceiling. A turn that cannot fit alone is partitioned only at existing RawSpan boundaries. One complete neighboring turn may overlap under the frozen preceding-first policy. Every RawSpan has exactly one primary assignment.

The request-budget tokenizer view consists of the frozen llama.cpp rendered chat prompt, the exact prefix `\n\n<FLATPROP_DYNAMIC_SCHEMA>\n`, the canonical JSON bytes of the exact dynamic response schema, and the exact suffix `\n</FLATPROP_DYNAMIC_SCHEMA>`. Both rendered chat tokens and schema tokens are recorded separately. The combined budget view is tokenized by the running frozen Qwen3-8B llama.cpp `/tokenize` endpoint with `add_special=false` and must not exceed 6,144 tokens.

## Writer and Aggregation

Each chunk uses the unchanged v3 extraction prompt and proposition fields, `temperature=0`, `seed=42`, thinking disabled, and `max_tokens=8192`. Execute every chunk of `de43030f_1` first, then continue in frozen source-session/chunk order only after all of that session's chunks succeed. This changes execution order only; chunk/request identities remain bound to the canonical plan. Each chunk is called at most once through the local loopback endpoint. A length stop at the cap, malformed response, transport failure, or provenance violation stops the stage; no retry, cap increase, repair call, or hosted fallback is allowed.

After a session's chunk outputs freeze, exact cross-chunk identity is SHA256 over canonical JSON of stripped proposition text and catalog-ordered canonical evidence refs. Only exact identity matches from different chunks collapse. Repeated outputs within a single chunk, paraphrases, and distinct evidence sets remain separate. Every surviving proposition retains source chunk lineage and the original Harness-reconstructed evidence. A casefolded `SequenceMatcher` ratio of at least `0.90` is recorded as a heuristic near-duplicate diagnostic only; it never merges propositions or gates the writer.

## Diagnostic Boundary

The downstream path reuses ADD-only FlatProp materialization, local Qwen3-Embedding-0.6B Dense top-8, the frozen 1,024-token rank-aware projection, and exactly ten calls to the frozen shared reader. Every proposition remains `ADD / SESSION_NOTE / SESSION_DERIVED / ACTIVE / version=1 / supersedes_id=null`; set `MEM3A3_FLAT_NO_REVISION=YES` only when this invariant passes. Labels are loaded only after writer chunks, session aggregates, inventory, retrieval, plans, bundles, predictions, and the reader ledger freeze. No judge, hosted call, 102 DEV, TEST, MedMemoryBench scoring, or revision semantics are in scope. The results remain frozen-ten mechanism diagnostics only.

The MEM-3A.2 and MEM-3A.2S whole-session outputs remain historical diagnostics and are not imported as MEM-3A.3 chunk results.
