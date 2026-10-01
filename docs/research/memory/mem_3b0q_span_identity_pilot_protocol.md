# MEM-3B0Q-R2 Span-Grounded Slot Identity Pilot

Status: protocol freeze for a small, exploratory local-model qualification on the already-reviewed LongMemEval frozen-10 diagnostic material. This is not a public benchmark run, a hidden test, or a final method freeze.

## Research question

Can source-validated principal/object/attribute/value spans produce safer candidate slot identities than free-form identity keys and proposition-pair similarity, while preserving explicit abstention when a proposition does not identify its state dimension?

This stage addresses identity only. It does not decide whether a fact is current, obsolete, updated, or deleted. It does not implement `ADD`, `UPDATE`, `DELETE`, `SUPERSEDED`, `CURRENT`, `AS_OF`, or `CHANGE`.

## Frozen source and diagnostic cases

- Source artifact: `runs/memory/mem3/mem3b0q-factorized-admission-20260930/eligible_records.jsonl`.
- Required source SHA-256: `1ed014923a7213106e4da384929c6d303ee60a4a607e8b2942231dc03718da7c`.
- Case selection lineage: post-freeze B0Q control audit SHA-256 `0f2bb3c86f4666b12032395f7fbfbb96d06fb1e2c50d54f253de69dbea694a64` and user-authorized review decisions SHA-256 `766f602146776e114e6696fa3e404a7954a3ceb3e72c0f3516e51288a8bd5564`.
- Case material comes from the frozen-10 diagnostic and post-freeze controls already reviewed in B0Q. It is intentionally a development challenge set assembled after observing B0Q failures; it is not independent evidence.
- The frozen 19 input records cover Instagram follower count, Node version, wallet color/material, workout-vs-PC key collision, two destination-specific trips, generic numeric `reported_value`, repeated macrame interest, and duplicate completed-purchase event.
- The preparation step records the exact memory IDs and a SHA-256 manifest before any model call. Case labels and expected pair outcomes are stored separately and are never included in model requests.
- No timestamp, source-session date, question, answer, B0Q human decision, candidate source, B0R key/value, or benchmark label enters a proposer request.

## Local-only model protocol

- Proposer: frozen local Qwen3-8B Q4_K_M, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- Runtime: pinned loopback llama.cpp `10068 (571d0d540)`, executable SHA-256 `3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb`.
- Endpoint: `127.0.0.1:8081`; no hosted provider, API key, embedding, judge, reader-answer call, or network fallback.
- Temperature `0`, seed `42`, thinking disabled. Dynamic JSON schema; response budget `8192` tokens for the single pilot batch. One request only: no same-request retry, batch subdivision, or fallback. A transport or output-format failure ends this pilot as `NO`.

## Proposal and deterministic validation

For each proposition, the model may return only:

```json
{
  "principal_key": "normalized label",
  "principal_witness_span": "exact source substring or null",
  "object_key": "normalized label or null",
  "object_witness_span": "exact source substring or null",
  "attribute_key": "normalized label",
  "attribute_witness_span": "exact source substring or null",
  "value_text": "normalized value proposal",
  "value_witness_span": "exact source substring or null",
  "property_kind": "SINGLE_VALUE_STATE|MULTI_VALUE_STATE|EVENT|NON_STATE|UNKNOWN",
  "change_cue_span": "exact source substring or null"
}
```

The model proposes per-record normalized labels, but it has no pairwise-equivalence or revision-decision field. Each non-null witness must occur exactly once in that proposition. Invalid witnesses, missing principal/attribute/value evidence, or an ungrounded label are terminal `UNRESOLVED`; Harness never repairs or guesses them. Key syntax and anchor-support rules are frozen in code. At least one non-generic attribute token must be supported by the attribute witness or the frozen alias lexicon. Generic-only attributes such as `value`, `number`, `information`, `interest`, `preference`, `plan`, `status`, `thing`, `fact`, and `event` are not enough to establish a slot.

Harness normalizes accepted keys with the existing deterministic lower-snake key validator. It verifies every witness as a verbatim, unique substring, then checks key-token support against witness tokens plus a small frozen lexical alias table (initially common color terms and noun inflections). No stemming, embedding similarity, post-hoc LLM alias merge, or unregistered synonym expansion is allowed. A candidate slot key is the exact normalized tuple:

```text
scope_id + principal_key + object_key-or-null + attribute_key
```

Only exact tuple equality creates same-slot candidates. `property_kind` is a veto: only `SINGLE_VALUE_STATE` may proceed to a candidate; `MULTI_VALUE_STATE`, `EVENT`, `NON_STATE`, `UNKNOWN`, and unresolved identity are retained as coexisting history without revision admission. Even a same-slot candidate is not evidence of a revision.

## Pre-registered diagnostic checks

Expected structural/control behavior:

- Instagram 500/600: same grounded slot candidate, independent of the B0Q pair verifier's false negative.
- Node version: same slot candidate; equal values are not a revision.
- Wallet: black/neutral color propositions may share a candidate slot; leather material must not share that slot.
- Seven-minute workout vs gaming PC: must not share a slot even though B0R emitted the same broad key.
- Japan vs India trips: must not share a slot; both plans may coexist.
- Generic numeric `reported_value`: unresolved; no inferred measurement slot.
- Macrame interest mentions: not admitted as a single-value revision slot.
- Completed purchase: event, never a mutable state slot.

Pass requires all model spans to validate and all listed positive/negative control expectations to hold. Any failure is `MEM3B0Q_SPAN_IDENTITY_PILOT=NO`; no prompt tuning against these cases is allowed within this frozen pilot. A future revision must get a new protocol, run ID, and explicit development-only label.

## Scope and stop conditions

- No MemoryStore mutation, retrieval, answer generation, benchmark scoring, LongMemEval 102 DEV/TEST access, MedMemoryBench, or medical evidence/policy interaction.
- Freeze selected inputs, protocol, prompt, schema, validators, code identity, runtime, and response journal before generation.
- After generation, freeze validated proposals before joining the expected diagnostic labels.
- Report all case outcomes, unresolved spans, local calls/tokens/latency, and false/missed identity controls. No accuracy or superiority claim is made from 19 items.
- Stop after this pilot. Do not start B1 or the scorecard. `MEM3B1_READY` remains `NO` unless a later separately reviewed admission gate passes.
