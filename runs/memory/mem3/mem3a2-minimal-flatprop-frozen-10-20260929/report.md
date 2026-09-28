# MEM-3A.2 Minimal FlatProp v3 Frozen-Ten Diagnostic

Completion gate: `MEM3A2_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC=NO`.

## Stop Reason

The 477/477 request preflight passed: the longest rendered prompt was 19,862 tokens, the 4,096-token output reserve fit within the 131,072-token context limit, and no prompt was truncated. The frozen local llama.cpp/Qwen3-8B runtime was used.

Writer extraction stopped at unique source session 55 of 477. The HTTP response was successful (`200`), but llama.cpp returned `finish_reason=length` at the frozen 4,096 completion-token cap. The harness recorded `COMPLETION_TRUNCATED`; it did not validate or accept that packet and did not retry it.

The preceding 54 structurally valid outcomes (567 propositions) remain in the ignored local write-ahead journals and are also summarized in `session_extractions.partial.jsonl` and `session_extraction_manifest.partial.json`. The 55th response remains captured locally for audit. No raw HTTP response is included in committed artifacts.

## Scope Not Run

- FlatProp materialization, Dense embeddings/retrieval, ContextPlans, ContextBundles, and reader calls were not started.
- Benchmark labels were not loaded; the 102-question DEV set and TEST split were not accessed.
- Judge calls and hosted calls: zero.
- No RawSpan-Dense versus FlatProp-Dense comparison or performance claim is available from this stopped run.

The historical `NO` and `NOT_RUN` manifests remain unchanged. This is a transport/runtime output-cap failure, not evidence about FlatProp semantic quality. Any continuation requires a reviewed protocol decision; this run will not replay session 55.
