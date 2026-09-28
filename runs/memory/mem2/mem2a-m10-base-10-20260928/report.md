# MEM-2A M10-Base Frozen Ten-Case Diagnostic

- Gate: `MEM2A_M10_BASE_FROZEN_10_DIAGNOSTIC=YES`
- Purpose: diagnostic evidence about the existing M10 Memory substrate; no ranking or performance claim.
- Protocol: controlled evaluation of M10-Base-RawTurn on the frozen public DEV-10 selection.
- Cases: exactly ten frozen DEV IDs; TEST access: `false`; no 102-case DEV run.
- Reader: local frozen Qwen3-8B Q4_K_M; final contract SHA256 `57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3`.
- FullContext prompt-fit/truncation gate from frozen MEM-1D4 v3: `True`; FullContext was not rerun in MEM-2A.
- Calls: reader `10/10`, memory-internal LLM `0`, embedding `0`, judge `0`, hosted `0`.
- Runtime: `version: 10068 (571d0d540)
built with Clang 20.1.8 for Windows x86_64`; loopback `http://127.0.0.1:8081/v1`; model SHA256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- All generated memory operations ADD-only: `True`; logical inventory and replay deterministic: `True`.

## Descriptive Answer Diagnostics

The following six-system table is descriptive only. The first five rows are read from hash-verified frozen MEM-1D4 v3 artifacts and were not rerun. Metrics are deterministic lexical F1 and normalized EM under the same final reader contract; no official LLM judge was used.

| System | Questions | Mean token precision | Mean token recall | Mean token F1 | Normalized EM | Source |
|---|---:|---:|---:|---:|---:|---|
| FullContext | 10 | 0.0598 | 0.5539 | 0.0956 | 0.0000 | frozen MEM-1D4 v3; not rerun |
| OpenClaw | 10 | 0.1685 | 0.3000 | 0.2017 | 0.1000 | frozen MEM-1D4 v3; not rerun |
| Mem0 OSS | 10 | 0.0755 | 0.1289 | 0.0726 | 0.0000 | frozen MEM-1D4 v3; not rerun |
| SimpleMem | 10 | 0.0111 | 0.0143 | 0.0125 | 0.0000 | frozen MEM-1D4 v3; not rerun |
| PropMem | 10 | 0.1249 | 0.4508 | 0.1937 | 0.0000 | frozen MEM-1D4 v3; not rerun |
| M10-Base | 10 | 0.0396 | 0.1200 | 0.0595 | 0.0000 | MEM-2A local run; descriptive diagnostic only |

## Memory Diagnostics

- Mean native answer-session Recall@5: `0.4600`; Recall@8: `0.4800`; MRR: `0.4500`.
- Recall@10 is reported only as a compatibility alias equal to Recall@8 (`NATIVE_TOP_K_8`); M10-Base never retrieves ten candidates.
- Mean ContextManager-projected answer-session Recall@8: `0.2400`.
- Mean reader-tokenized memory context: `1012.6000`; mean M10 estimated memory tokens: `891.4000`. These are distinct token measures.
- Abstention cases: `0`; deterministic abstention accuracy: `NA`.
- Per-category token metrics are in `deterministic_metrics.json`; per-question retrieval, projection, token coverage, gold-sequence presence, and provenance are in the reflection packet.

## Efficiency

- SQLite ADD operations: `4854`; cumulative SQLite size: `11431936` bytes.
- Ingestion: `961.7596` ms total; lexical retrieval: `394.2353` ms total; ContextManager: `20.5831` ms total.
- Reader: `11160` prompt tokens, `50` completion tokens, p50 `468.6325` ms, p95 `745.0780` ms.
- Local reader wall time is reported; GPU-only kernel time and monetary cost were not measured. There was no hosted reader API or embedding cost.

## Interpretation

M10 core supports ADD/UPDATE/DELETE/NOOP, versions, SUPERSEDED and temporal validity primitives, but this run deliberately emitted unique-key ADD only. It does not evaluate semantic revision resolution, stale-fact detection, or CURRENT/AS_OF/CHANGE reasoning. Failures in those tasks do not imply that unused revision primitives failed.

STOP: no external system rerun, 102-case DEV, TEST, M10-Flat, RevMem, proposition extraction, revision materialization, temporal routing, or RL was performed.

Artifacts: `runs/memory/mem2/mem2a-m10-base-10-20260928`; report generated 2026-09-28T07:33:36.512449+00:00.
