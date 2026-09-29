# Memory Benchmark Roadmap

Frozen 2026-09-29. This route preserves historical LongMemEval evidence in its original scope; it does not reinterpret prior partial, failed, or diagnostic runs.

## Benchmark roles

| Benchmark | Role |
| --- | --- |
| LongMemEval | General-memory mechanism debugging and eventual one-shot general-memory closeout |
| MedMemoryBench | Medical-domain external-transfer evaluation of personal longitudinal memory |
| ESL-Bench | Longitudinal integration/reasoning environment candidate, not a memory-mechanism benchmark |

All personal-history methods, including upstream-labelled BM25/Embedding/Graph RAG, route to MEMORY_READ. External clinical knowledge retrieval remains EXTERNAL_RETRIEVAL. TEAM_ORCHESTRATION is a separate execution-policy axis. See benchmark_routing_contract.md.

## Phase I: LongMemEval mechanism debugging

Completed evidence remains the historical sequence:

RawTurn → rank-aware projection → RawSpan granularity → semantic retrieval → provenance-by-reference → FlatProp.

The currently authorized mechanism checkpoint was the frozen-ten MEM-3A.2R diagnostic. Its terminal checkout was commit 73332535a7aef74d6a93d39092f5f596d00ec767 on branch mem3a2r-writer-capacity-20260929; its run manifest records MEM3A2R_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC=NO at session 102 after a duplicate evidence reference (S0070). This is a structurally informative provenance failure, not a context-capacity diagnosis. Its checkout and artifacts remain untouched.

The authorized unresolved chain is limited to the frozen-ten diagnostic:

FlatProp completion → revision identity → deterministic revision materialization → CURRENT / AS_OF / CHANGE semantics.

At audit time, the separate MEM-3A.2R checkout was terminal with a structurally informative NO at session 102 (duplicate evidence reference); it is not a context-capacity diagnosis. Keep its contracts and artifacts untouched. Do not start 102-case DEV, MEM-3B0, MEM-3B, or new retrieval tuning until the frozen-ten mechanism chain is completed and reviewed.

### Mechanism-freeze gate

LONGMEM_MEMORY_MECHANISM_FROZEN=YES only after:

1. FlatProp representation/writer has a terminal usable design.
2. Revision identity has a deterministic frozen contract.
3. Revision materialization is implemented and diagnosed.
4. CURRENT, AS_OF, and CHANGE semantics are frozen.
5. Frozen-ten artifacts have no unresolved infrastructure or provenance defect.
6. Reader, retrieval, and projection contracts are pinned.
7. No method choice is still being selected from LongMemEval outcomes.

This freezes architecture, not a performance claim.

After the gate, the 102-case DEV may be used at most once as ONE_SHOT_GENERAL_MEMORY_CLOSEOUT with the exact frozen architecture and no tuning. It is optional for the main Health-Copilot route. A disappointing score does not reopen method choice unless a demonstrable infrastructure or benchmark-contract defect is found.

## Phase II: MedMemoryBench transfer

Start only after both MEDMEMORYBENCH_COMPATIBILITY_AUDITED=YES and LONGMEM_MEMORY_MECHANISM_FROZEN=YES. The next named stage is MED-M0 Adapter Fidelity, with new run identities separate from MEM/LongMemEval.

Order:

adapter fidelity → frozen transfer → medical revision, temporal, and noise analysis.

The evaluation is PUBLIC_EXTERNAL_TRANSFER because labels/provenance are public; it is not a blind test. It asks whether mechanisms discovered on a general-memory benchmark transfer to longitudinal healthcare dialogue. Outcomes are TRANSFER_SUPPORTED, TRANSFER_PARTIAL, or TRANSFER_NOT_OBSERVED. First attribute a failure to adapter, language, domain semantics, noise, medical-state representation, or reader reasoning before reopening the general mechanism.

No MedMemoryBench tuning or benchmark execution is authorized by this roadmap/audit.

## Phase III: ESL integration

ESL-Bench remains an INTEGRATION_ENVIRONMENT_CANDIDATE. After Memory and External Retrieval transfer characterization:

ESL longitudinal universe → MEMORY-only → EXTERNAL_RETRIEVAL-only where externally dependent → MEMORY + EXTERNAL_RETRIEVAL → TEAM combinations → execution-policy episodes.

Later, appropriately governed episodes may inform SFT, GRPO, or OPD research. This is a future possibility, not current authorization or evidence.
