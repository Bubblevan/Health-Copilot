# MEM-3A Frozen Ten-Case Diagnostic - Halt Report

**Status:** FAILED at writer extraction; `MEM3A_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NO`.

The frozen writer prompt and contracts passed hash verification. Tokenizer preflight covered 477 unique source-session identities; maximum prompt length was 6429 tokens, every prompt plus the 4096-token reserve fit within 131072, and all were marked `truncated=false`.

The local Qwen3-8B runtime was verified on loopback. Three initial `memory_write_extract` calls were issued: 2 succeeded and 1 failed exact output validation. The failed call ended with `finish_reason=stop` (4625 prompt tokens, 1103 completion tokens); it was recorded as `EXTRACTION_FAILURE` and was not retried. Its source-session identity and raw response SHA are in `writer_failure_diagnostic.json`; the raw response was not retained, and the available error subtype is only `ValueError`. The remaining 474 sessions were not called.

Because the writer failure gate fired, execution stopped before proposition materialization, embedding, Dense retrieval, context projection, reader calls, and benchmark-label access. Metrics, candidate revision groups, case review, and comparison are unavailable. No result or performance claim is made. The GPU reader server was already active before this run and was left untouched.
