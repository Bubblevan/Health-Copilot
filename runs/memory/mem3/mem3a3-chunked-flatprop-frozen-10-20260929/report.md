# MEM-3A.3 - Deterministic Chunked FlatProp Diagnostic

Completion gate: `MEM3A3_CHUNKED_FLATPROP_FROZEN_10_DIAGNOSTIC=NO`.

Stopped before closeout: `RuntimeError: MEM3A3_CHUNKED_FLATPROP_FROZEN_10_DIAGNOSTIC=NO; stopped on exact failed chunk 104d270a9c92d9d464dc5eb07adf1eac6881d5edd6a1d437fc040ec814508bc9`.

No benchmark labels were loaded. See `writer_failure.json` for the exact chunk when a writer request failed.

## Writer Gate Failure

- Frozen plan: 477 sessions, 785 chunks; execution stopped at chunk 148 after 147 successful chunks.
- Failed source session: `sharegpt_vbNrVtS_151`; primary turns `0-7`; 110 RawSpans.
- Failed chunk: `104d270a9c92d9d464dc5eb07adf1eac6881d5edd6a1d437fc040ec814508bc9`.
- Prompt budget: 6,016 tokens in the frozen rendered request; llama.cpp reported 5,254 prompt tokens.
- Outcome: `finish_reason=length` at 8,192 completion tokens after 154,246.814 ms. The single-call policy stopped immediately; no retry, cap increase, or model repair was attempted.
- Calls: 148 local writer calls; 0 retries; 0 hosted calls. Failure attribution: writer capacity/truncation, not a downstream memory-quality result.
- Downstream retrieval, projection, 10 reader calls, labels, and metrics were not run. No FlatProp inventory was materialized.
- `MEM3A3_CHUNKED_FLATPROP_FROZEN_10_DIAGNOSTIC=NO`; do not merge this stage to `main` and do not start MEM-3B0.
- Exact frozen request, chunk row, failed ledger row, raw response, assistant content, response metadata, and cache journal are preserved under `reflection_failed_chunk/` for Reflection.

## Verification

- Ruff: passed for the MEM-3A.3 implementation and focused tests.
- `py_compile`: passed for the changed implementation and test modules.
- Focused tests: 29 passed.
- All run-artifact SHA sidecars: valid.
- `git diff --cached --check`: passed before final report update; no implementation code changed during this run.
