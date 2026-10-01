# MEM-3B0Q Candidate-Bounded Local Smoke Protocol v1

Status: `FROZEN_OFFLINE_PROTOCOL`; this freezes the control pack, prompt/request
contract, validator, and exact acceptance rule only. It is not authorization for
inference, a POST, or B1.

## Purpose

This protocol tests whether the candidate-ID proposal contract can produce one
source-grounded atom with the local Qwen runtime while ignoring an unsupported
object in the same proposition. It addresses the consumed R4-P01 free-form
span failure with a new method and new source ID. The prompt itself names
"tablet" and "basket" as examples of unlisted nouns, so this is a **prompted
compliance smoke**, not an unbiased unknown-object generalization test. It does
not qualify the extractor, implement revision materialization, or establish
benchmark quality.

The complete draft control pack is
`mem3b0q_r4_candidate_control_pack_v1.json`: six non-overlapping synthetic
cases covering all nine declared typed slot pairs, nine wrong-value controls,
and two unknown-object distractors. Only `R4C-05` is proposed for the first
bounded smoke. The other five cases remain unrun and are not silently included
in the pilot result.

## Roles and Runtime

| Role | Frozen target |
|---|---|
| Reader / proposal model | Local Qwen3-8B Q4_K_M GGUF, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` |
| Memory system | Candidate-bounded source-span proposal + deterministic v3 Harness guard; no MemoryStore |
| Embedding model | `NONE` |
| Judge model | `NONE` |
| Hosted API / key | `NONE` / `NONE` |
| Endpoint | `http://127.0.0.1:8081/v1`, verified loopback only |
| llama.cpp | server build `b10068-571d0d540`, version `10068`, binary SHA-256 `3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb` |
| llama.cpp CLI | version `10068 (571d0d540)`, SHA-256 `48566cf6e2969464b799dbcac7393b3549f9efd6884688074dc803125dbafa85` |
| Server execution | context `131072`; 99 GPU layers; Flash Attention on; Q4_0 K/V cache; one slot |
| Request decoding | temperature `0`; seed `42`; max completion `256`; stream off; thinking off |

The request builder uses a fixed system prompt, deterministic candidates from
the case text and frozen alias map, and a per-case strict JSON Schema. The model
selects owner/object/attribute candidate IDs; values remain exact source spans.
The request projection must contain only `source_id`, `proposition_text`, and
the typed candidate table. `expected_atoms`, wrong-value controls, and the
control-pack oracle must never enter the prompt.

## Proposed Single-Case Gate

Case `R4C-05` reads:

> My personal laptop runs Linux as its operating system. A tablet uses Windows as its operating system.

The only expected atom is `SELF / PERSONAL_LAPTOP / OPERATING_SYSTEM / Linux`.
The unregistered `tablet` fact must not be attached to the laptop candidate.
The prompt and schema permit supported atoms while the v3 Harness rejects a
typed proposal that crosses the sentence boundary. The v3 boundary and
determiner+noun checks are conservative lexical heuristics over this frozen
fixture, not a general sentence or entity parser; unrecognized noun phrases
without a detected determiner can evade the latter check.

Before any future POST, the runner must:

- verify the pack, protocol, prompt, schema, request-builder, candidate-builder,
  v2/v3 guard, runtime, model, and server hashes from a frozen run lock;
- pass the existing GET-only R4 runtime preflight and verify one idle slot;
- confirm the output directory for this exact run ID does not already exist;
- make one completion POST to the exact loopback path with environment proxy
  inheritance disabled, no retry, no fallback, and no service restart;
- re-check process identity and idle service state after the response;
- preserve request, raw response, normalized proposal, validation result, and
  SHA-256 hashes as append-never/overwrite-never artifacts.

Port `8092`, embedding services, external services, MemoryStore, Evidence,
citations, and safety policy are outside this pilot. There is no state mutation.

## Outcomes

`INFRA_FAILURE` covers failed runtime preflight, identity/hash mismatch,
transport failure, non-success HTTP response, timeout, or missing response
envelope before model content exists. It is not a quality score and is never
converted to zero. There is no retry.

`QUALITY_FAILURE` covers returned model content that is malformed, violates
the strict schema/contract, fails candidate resolution or v3 validation, omits
the expected atom, adds an unsupported atom, or binds `Windows` to the laptop.

The pilot passes only if exactly one POST returns HTTP 200 and the validated
normalized atom **multiset is exactly equal** to the oracle's single normalized
atom signature. Equality includes cardinality, scope, typed IDs, exact witness
text and codepoint offsets for owner/object/attribute/value, and exact value;
duplicates, omissions, extra atoms, or witness drift fail. Deterministic
candidate resolution and v3 validation must pass, pre/post runtime witnesses
must match, and hosted calls/retries must both be zero. A pass is recorded only as
`R4C_CANDIDATE_LOCAL_SMOKE=YES`; it does not imply
`MEM3B0Q_CANDIDATE_BOUNDED_EXTRACTOR_READY=YES`.

Any result is diagnostic runtime evidence only. No performance comparison,
LongMemEval claim, medical-transfer claim, or revision correctness claim may be
made from this one synthetic case.

## Full-Pack Qualification Gate

The other five cases are not part of this smoke. Any later full-pack request
requires a separate reviewed lock and bounded-call authorization. The full
pack gate requires exact expected atom-set match in all six cases, zero invalid
cross-binding admissions across all nine typed-slot pairs, both unknown-object
cross-bindings rejected, and no infrastructure failure. If any case fails,
record the error taxonomy and stop; do not edit prompts, aliases, controls, or
the split after seeing its output. A repaired method requires a new version and
new non-overlapping controls.

## Current Offline Evidence

- Frozen R4 v1 pack and artifacts remain unchanged.
- Candidate control pack has six unique IDs not present in frozen R4 v1 or the
  maintained non-frozen inventory in
  `mem3b0q_r4_development_control_manifest_v1.json`; tests enforce the full
  inventory ID set and a case-folded, whitespace-normalized text disjointness
  audit against both sources.
- Offline tests verify all gold atoms pass the v3 guard, all nine wrong-value
  proposals fail closed, both unknown-object cross-bindings fail closed, all
  six generated schemas pass Draft 2020-12 validation, exact normalized atom
  multiset matching rejects duplicate/extra/drifted atoms, and oracle data is
  absent from the model-facing request.
- Candidate request + R4 v2/v3 + builder/locality targeted suite: `54 passed`.
- R4/factorized/pairwise regression suite: `145 passed`.
- Both runs emitted only a pytest-cache permission warning; test execution was
  unaffected.
- No model was loaded or queried for this draft. Grammar conversion and
  candidate-proposal sampler acceptance remain unverified.

The offline protocol/request builder may be frozen after review. A future
single POST still requires a separately reviewed hash-locked one-shot runner
and run lock, plus verified schema grammar/sampler handling. This review grants
no inference authorization.

`MEM3B0Q_CANDIDATE_BOUNDED_EXTRACTOR_READY=NO`
`MEM3B0Q_MEM3B1_READY=NO`
`MEMORY_PUBLIC_CLOSEOUT=NO`
