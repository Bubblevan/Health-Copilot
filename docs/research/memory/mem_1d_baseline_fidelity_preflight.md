# MEM-1D Baseline Fidelity Preflight

`MEM1D_BASELINE_FIDELITY_READY=NO`

`SIMPLEMEM_HYBRID_FIDELITY=NO`

`MEM1_FROZEN_10_CASE_DIAGNOSTIC=NOT_RUN_BY_GATE`

## Decision

The 10-case run is blocked before D1. The upstream-supported SimpleMem FTS
dependency was restored and its keyword API passes a deterministic lexical
test, but the installed SimpleMem 0.1.0 `HybridRetriever.retrieve()` execution
path never calls that API or the structured-search API. Calling either path
would require changing SimpleMem's retrieval algorithm, which this stage
forbids. SimpleMem must therefore be treated as a **degraded semantic-only
adaptation** unless human review changes the frozen baseline policy.

No benchmark question was run in MEM-1D. The frozen 10-case manifest, DEV/TEST
split, LongMemEval dataset, prediction artifacts, and prior MEM-1C1 artifacts
were not modified. No native answer track, 102-DEV, TEST, local judge, M10-Flat,
RevMem, or Memora/FAMA run occurred.

## SimpleMem 0.1.0 Audit

- Base Health-Copilot commit: `97c5851d2afd533220d71509a777c2420673876e`.
- MemEval source checkout: pinned commit `807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4`; tracked source remains clean.
- Runtime: isolated `external/memory/MemEval/.venv`, Python `3.11.15`.
- Installed `simplemem==0.1.0`, from the pinned PyPI registry source in MemEval's `uv.lock`. The lock records the wheel SHA256 as `e6754a09f2f8474103d454b2e815daa4a4a1be1cd16b329fe2815322b9aef6b6` and sdist SHA256 as `ae70c518205ba8dccc51f8731007fb1b86fb8d33254b32e192a829e0a6a86196`. All 22 installed RECORD entries verified; installed source-tree SHA256: `cf7741e29ce5290407d2b24e716b5c7db751f51c0ecbbb085e699742147a6472`.
- Installed `lancedb==0.27.1`; SimpleMem's METADATA requires `tantivy>=0.20.0`, installed version `0.25.1`.
- FTS implementation: local LanceDB Tantivy index on `lossless_restatement`, with tokenizer `en_stem`; `VectorStore.keyword_search()` joins keyword terms and calls LanceDB's text query API. SimpleMem metadata lists Tantivy as a dependency. LanceDB `0.27.1` exposes the upstream optional `pylance` extra (`pylance>=1.0.0b14`).

### Isolated Runtime Overlay

The original environment lacked `pylance`, so LanceDB FTS index creation was
skipped. Installed only the declared FTS extra into the isolated MemEval venv:
`lancedb[pylance]==0.27.1`. This left the project `uv.lock` and system Python
unchanged. Resolved overlay versions are `pylance==12.0.0`,
`lance-namespace==0.11.1`, and `lance-namespace-urllib3-client==0.11.1`;
`lancedb` remains `0.27.1`. All 63 installed Pylance RECORD entries verified;
the installed Lance source-tree SHA256 is
`bc1f92f1073b86444fe60e8ff94f10603ae88a3c14876618a7218aae0d6d8986`. The exact
downloaded Pylance wheel hash was not recoverable from the local uv cache.

With this overlay, LanceDB printed `FTS index created (Tantivy mode)` and
`_fts_initialized=True`.

### Deterministic Synthetic Fidelity Test

Two synthetic memories were inserted. The fake embedding deliberately ranks a
nonmatching classical-music distractor above the lexical target about
metformin:

| Path | Top result |
|---|---|
| Semantic vector search | `distractor` |
| Direct `keyword_search(["metformin"])` | `target` |

This proves the restored Tantivy keyword API itself works and can recover a
result when lexical signal is stronger than semantic similarity.

### Actual HybridRetriever Path

Call tracing was run against the installed package with planning both disabled
and enabled. Results were identical:

| Planning | Semantic calls | Keyword calls | Structured calls |
|---|---:|---:|---:|
| Disabled | 1 | 0 | 0 |
| Enabled | 1 | 0 | 0 |

`HybridRetriever.retrieve()` falls back to semantic search when planning is
disabled. With planning enabled, planned queries and reflection searches also
use semantic search. `_keyword_search()` and `_structured_search()` exist as
helpers, but the retrieval/planning/reflection call graph has no call sites for
them. The FTS index and direct keyword API are operational, but they are not
part of the executed SimpleMem retrieval algorithm.

**Fidelity finding:** labeling the current system output as SimpleMem's
hybrid retrieval would be inaccurate. Wiring these helpers into retrieval
would change the algorithm, so the stage's no-algorithm-patch rule prevents a
local fix. SimpleMem is a degraded semantic-only adaptation for this harness.

## Remaining D0 Items

- D0.1: audited; SimpleMem hybrid fidelity gate failed as described above.
- D0.2: structured `baseline_warnings.jsonl` telemetry was not added. The prior MEM-1C1 Mem0 DELETE-handler and invalid-JSON warnings remain disclosed in its report; this telemetry must be implemented before any later multi-case run.
- D0.3: this report uses `full-context reference / context-coverage upper bound` wording; no answer-quality upper-bound claim is made.
- D0.4: final runtime revalidation was not started because the mandatory D0.1 gate stopped the stage. The prior MEM-1C1 report records the earlier runtime pins, but they are not treated as a fresh MEM-1D preflight.

## Human Review Required

Before any 10-case execution, choose whether to:

1. Keep SimpleMem in the frozen five-system matrix but explicitly treat and report this exact adapter as degraded semantic-only; or
2. Revise the baseline matrix/protocol and freeze a replacement or exclusion before running cases.

Do not run D1 until that decision is recorded and the remaining D0 items are
completed. This report is a fidelity gate failure, not a benchmark result.
