# MEM-3B0P — Harness-Grounded Pairwise Revision Admission

**Canonical base:** `f356c540fa5aa6075e9da2acbedad67ea0e46d0e`  
**Branch:** `mem3b0p-pairwise-revision-admission-20260930`  
**Purpose:** replace B0S's group-level semantic verifier with grounded proposition pairs and a deterministic all-pairs admission rule. This is a research adaptation, not a new memory algorithm.

## Frozen Inputs and Scope

- FlatProp: `runs/memory/mem3/mem3a3r-recursive-flatprop-frozen-10-20260929/flatprop_inventory.jsonl`, 8,112 rows, SHA-256 `250a24a10b532cb01bd841b965511df019fc0e597ec1e85ee12d7830c166c5fe`.
- B0R identity proposals: `runs/memory/mem3/mem3b0r-revision-identity-keyed-map-20260930/revision_identity_records.jsonl`, SHA-256 `49af354bb535909a572df580cf02483ab56022dcf31873266154caf0086b4f75`.
- B0R `revision_kind`, `subject_key`, `attribute_key`, and `value_text` are `SEMANTIC_CANDIDATE_HINTS`, not a state schema. No B0R artifact is modified; the 8,112-row identity proposer is not rerun.
- This stage writes research artifacts only: no MemoryStore mutation, retrieval, embedding, reader answer, judge, QA benchmark, or MedMemoryBench run.

## Pre-Registered Scorecard

The final memory scorecard is frozen in `memory_module_scorecard_contract.md` and its JSON companion. Commit that contract before any B0P pairwise model request. The 102-case LongMemEval closeout is a one-shot general-memory closeout after method freeze, not a tuning set. No method changes follow that result.

## Deterministic Candidate Construction

The runner projects only these FlatProp fields: `memory_id`, `scope_id`, `source_authority`, and `proposition_text`. It projects only `memory_id`, `revision_kind`, `subject_key`, `attribute_key`, `value_text`, and `identity_origin` from B0R. It never materializes the timestamp-bearing source record. No time field is loaded, displayed, grouped on, or used anywhere in B0P.

Authority is derived only from FlatProp provenance: `user -> USER_ASSERTED`, `mixed -> USER_MIXED`, `assistant -> ASSISTANT_ORIGIN`; missing/other values become `HARNESS_UNKNOWN_FALLBACK`. Only user and mixed authority can proceed. Assistant-origin history remains historical data and cannot mutate user state.

Only `SINGLETON_STATE` and `MODEL_VALIDATED` identity hints can proceed. Fallback identity, unknown kind, assistant-owned subject namespace, and ineligible authority are blocked deterministically. Grounding then applies the frozen rule below. `value_text` is diagnostic only and is never used as a persisted state value.

Grounding normalization is Unicode NFKC, lowercase, `_` to space, then `[a-z0-9]+` tokenization. The frozen stopword set is: `a`, `an`, `and`, `are`, `as`, `at`, `be`, `been`, `by`, `for`, `from`, `had`, `has`, `have`, `in`, `is`, `it`, `of`, `on`, `or`, `the`, `to`, `was`, `were`, `with`. Block when any numeric token in `value_text` is absent from proposition tokens (`IDENTITY_HINT_BLOCKED_VALUE_NUMERIC_MISMATCH`), or when both sides have informative tokens but their intersection is empty (`IDENTITY_HINT_BLOCKED_VALUE_UNGROUNDED`). Do not fuzzy-match or rewrite values.

Passing rows form exact candidate groups by `(scope_id, subject_key, attribute_key)`; no key normalization, fuzzy merge, aliasing, embeddings, or special cases. Groups of size one are `NO_REVISION_HISTORY`; they trigger no model call and create no revision slot.

## Pairwise Verifier

For every repeated group, enumerate every unordered pair in stable sorted-memory-ID order. Pair identity is `SHA256(min_memory_id || max_memory_id || pairwise_contract_sha256)` over UTF-8 bytes, with no delimiter. The model request contains exactly `proposition_a` and `proposition_b`; no key, identity hint, value hint, ID, authority, timestamp, session, question, answer, gold, or retrieval field is sent.

The only task is: “Do these two propositions describe observations of the same real-world subject's same mutable, single-valued attribute, such that different values could represent versions of one state slot?” The only valid verdict enum is `SAME_MUTABLE_SLOT`, `COEXISTING_FACTS`, `UNRELATED`, or `UNKNOWN`. No explanation, generated key/value, clustering, transitivity, or state transition is requested.

Use the frozen local Qwen3-8B Q4_K_M artifact (SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`) through the pinned llama.cpp `10068 / 571d0d540` binary (SHA-256 `3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb`) at loopback only. Generation is temperature 0, seed 42, thinking disabled, max 32 tokens. No hosted fallback. One pair per request.

Malformed JSON, illegal enum, truncated output, or a pair-level transport anomaly becomes `UNKNOWN`; no same-request retry. A pending journal row after interruption becomes terminal `UNKNOWN` and is never resent. Frozen model/binary/endpoint identity failures are global infrastructure failures.

## Harness-Owned Admission and Opaque Slots

Harness admits a repeated group only when **every** pair verdict is `SAME_MUTABLE_SLOT`. Any `COEXISTING_FACTS`, `UNRELATED`, or `UNKNOWN` blocks the whole group as `REVISION_BLOCKED_PAIRWISE_INCONSISTENCY`. No graph component, union-find, or positive-edge transitivity is allowed.

For an admitted group, the opaque ID is `SHA256(scope_id || sorted(member_memory_ids) || admission_contract_sha256)` over UTF-8 bytes. The proposed subject/attribute keys remain human-readable diagnostic labels and do not participate in this identifier. Model output never creates a storage identifier or operation.

## Freeze Boundary

Before a `semantic_freeze.json` marker exists, the runner must not load timestamp-bearing FlatProp objects. B0P uses a strict allowlist projection that discards every non-allowlisted field while decoding. Pairwise calls, pair verdicts, group admission, and opaque slot IDs are frozen and hash-verified before the marker is written. No timestamps are needed after the marker either; any future timestamp-loading helper must refuse to run without verifying the marker first.

The pairwise call ledger is durable before each request. Every eligible repeated-group pair gets exactly one terminal ledger row; zero retries and zero hosted calls are required. `MemoryStore` mutation counters are fixed at `ADD=0`, `UPDATE=0`, `DELETE=0`, `SUPERSEDED=0`.

## Closeout and Readiness

Exhaustively human-review every admitted group using propositions, diagnostic labels, pair verdicts, and source authority only. Re-audit all named B0S false-safe examples, the Instagram 500/600 positive control, and the gym partial-recall control. Do not special-case any example.

`MEM3B0P_PAIRWISE_REVISION_ADMISSION_COMPLETE=YES` is structural and independent of readiness. `MEM3B0P_MEM3B1_READY=YES` requires: Instagram admitted; every historical B0S false-safe group now blocked; no clear false merge in exhaustive admitted-group review; complete all-pairs support; and no timestamp leakage. Gym cross-key recall is not required. If readiness is NO, revise only the pairwise semantic relation task; do not rerun FlatProp or the identity proposer and do not start MEM-3B1.

The frozen scorecard is specified in `memory_module_scorecard_contract.json`.
