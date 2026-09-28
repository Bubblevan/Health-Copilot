# MEM-3A - M10-FlatProp Protocol and Run Status

Status: **HALTED - extraction gate failed**

Completion gate: `MEM3A_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NO`.

## Frozen Protocol

M10-FlatProp is the write-time semantic proposition ablation before revision materialization. The intended comparison is MEM-2D RawSpan + Dense versus proposition extraction + Dense, on the same frozen ten DEV questions. It is diagnostic only and makes no ranking or public performance claim.

The local Qwen3-8B Q4_K_M writer receives one source session at a time, with only its date and ordered turns. It emits atomic propositions with diagnostic candidate keys and exact-turn quotes. The harness validates the strict schema, source roles, turn references, and quote substrings. Each valid proposition is materialized as one independent `ADD`, `SESSION_NOTE`, `SESSION_DERIVED`, `ACTIVE`, version-1 record with no expiry, end validity, or supersession. Candidate keys do not participate in retrieval or projection.

Dense retrieval embeds only the exact `proposition_text` with the frozen local Qwen3-Embedding-0.6B adapter. M10 scope/time eligibility, cosine rank, memory-ID tie-break, native top-8, rank-aware projection, 1024-token Memory budget, and the final reader contract remain frozen. No revision/current-state behavior, judge, hosted API, 102-case DEV, or TEST is in scope.

## Execution

- Base commit: `14d8c3ff9bf127d10ecb25cf5bdd30d949d7a5f4`.
- Required MEM-1D4, MEM-2B, MEM-2C, and MEM-2D gates were verified before preflight.
- Writer prompt/request preflight covered 477 unique source-session identities; all 477 fit the 131072-token context with a 4096-token output reserve and `truncated=false`.
- Frozen local runtime: Qwen3-8B Q4_K_M, SHA `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`; llama.cpp `10068 / 571d0d540`; loopback endpoint only.
- Three initial writer calls were made: two validated successfully, then the third stopped with `EXTRACTION_FAILURE`. No retry was made. The remaining 474 sessions were not called.
- The failed response had `finish_reason=stop`, 4625 prompt tokens, and 1103 completion tokens. The call ledger retained the response SHA256 but not raw response text; the captured exception is only `ValueError`, so a narrower validation subtype cannot be established from this run.
- Hosted calls: 0. Judge calls: 0.

## Outcome and Boundary

The extraction failure policy requires stopping before materialization, retrieval, projection, reader calls, and label join. Therefore no complete proposition inventory, candidate revision-group audit, case review, deterministic QA metrics, or RawSpan comparison exists for this run. No quality, efficiency, or performance conclusion is supported.

`MEM3A_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NO`.

Do not retry the failed session under this run identity. Any extractor/prompt change requires an explicitly versioned new stage and human review. MEM-3B / revision materialization remains unstarted.
