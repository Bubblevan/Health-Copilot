# MEM-3B0Q R4 Candidate-Bounded Design v1

Status: `PROPOSAL_ONLY`; not frozen, not integrated, and not authorization for inference.

## Motivation

The frozen R4-P01 attempt returned `owner_span="My workout plan"`, `object_span="activity"`, and `attribute_span="is"`; the frozen oracle expects `My`, `workout plan`, and `activity`. The response is a real contract-quality failure. One sample cannot establish output drift. It does show that the current request asks the model to generate free-form role spans while the deterministic Harness alone knows the typed alias registry. The request contains the system instruction and source proposition, but does not expose the registry as a bounded set of typed choices. “Use an exact registered alias” is therefore not operationalized at generation time.

The next design hypothesis is to reduce generator authority: let the Harness enumerate typed, source-grounded candidates from the already-frozen alias registry; let the model choose candidate IDs; let deterministic code resolve IDs to canonical slot identity. Do not widen aliases to accept the observed wrong spans.

## Proposed Candidate Interface

For each proposition, a deterministic candidate builder reads only the source text and the frozen alias map. It emits, for every exact alias occurrence:

```json
{
  "candidate_id": "owner:SELF:0:2",
  "field": "owner",
  "canonical_id": "SELF",
  "source_span": "My",
  "start": 0,
  "end": 2
}
```

Offsets are Unicode code-point `[start, end)` positions into the unchanged source. Candidate IDs are derived deterministically from field, canonical ID, and offsets. The alias text and offsets are harness-created, not model-authored.

For `R4-P01`, the frozen registry yields the owner candidate `My` / `SELF`, object candidate `workout plan` / `EXERCISE_PLAN`, and attribute candidate `activity` / `ACTIVITY`. It does not offer `My workout plan` as an owner, `activity` as an object, or `is` as an attribute. For `R4-P04`, the maximal owner alias is `My sister` / `SISTER`; its contained `My` match is suppressed for that field. Candidate generation applies longest exact typed alias first, then source-offset order, under an explicit overlap rule.

The model output schema would select only candidate IDs for owner/object/attribute, plus an exact source-grounded `value_span`, cardinality proposal, and abstention reason. The per-request JSON Schema enum for each candidate ID is generated from the frozen candidate table. An empty atom array remains the abstention path; the pinned llama.cpp schema-conversion path must be proven to accept the generated schema, including a field with zero candidates, before the protocol is frozen. If it cannot represent the candidate enums and empty-array path together, stop and redesign the schema before inference; do not silently fall back to free-form spans. Harness validation independently checks every selected ID against the table and reconstructs the source span and canonical ID. Any missing candidate, invalid ID, duplicate field, wrong source, or out-of-table selection fails closed. Value remains open-class, but must occur exactly once in the original source and is never used to create owner/object/attribute identity.

If one required field has no registered candidate, the model can only abstain with the pre-registered reason; an empty candidate list cannot authorize a guessed identity. `UNKNOWN`, events, and ambiguous references still produce no slot. This proposal does not add dates, revision actions, timestamps, or state materialization.

## Offline Qualification Before Any Request

No model request is permitted until a separately reviewed protocol and implementation satisfy all of these checks:

- Candidate generation is deterministic across replay and uses only frozen source text plus the versioned alias map; the request builder takes these inputs only from the hash-verified frozen pack.
- For every expected accepted atom in the frozen 20-case pack, the oracle's owner/object/attribute has exactly one candidate with the expected canonical IDs and source offsets.
- Maximal owner matching distinguishes P01 `My` from P04 `My sister`; typed matching distinguishes `activity` as attribute from `workout plan` as object.
- Every frozen abstention control has a missing or ambiguous candidate condition consistent with the existing reason precedence; no proposed span may create an unsupported canonical identity.
- Duplicate/overlapping aliases, repeated spans, case folding, punctuation, hyphenation, malformed IDs, cross-case candidate reuse, and unknown surface forms have positive and negative fixtures.
- Request construction transmits the exact candidate table but never the expected oracle. Generated schema hash, candidate manifest hash, prompt hash, and request-builder hash are frozen before output collection.
- All original R4 v1 protocol, pack, prompt, schema, artifacts, and failure labels remain byte-identical and historical. Results from the new method receive a new version/marker and are never merged into R4 v1.

The registry is currently a closed synthetic fixture inventory. This proposal would qualify candidate-bounded slot admission on that controlled inventory only; it would not establish open-world entity resolution, general memory extraction, revision accuracy, LongMemEval performance, or medical transfer.

## Stop Conditions

This design note authorizes no service restart, request, retry, R4 replay, B1, benchmark run, or public claim. The previously recorded R4 v1 outcomes stay `sampler=UNVERIFIED`, `quality=QUALITY_FAILURE`, `R4=NO`, and `MEM3B0Q_MEM3B1_READY=NO`. Any transition requires independent review of the concrete candidate builder, strict schema, tests, runtime witness instrumentation, freeze manifest, and a separate bounded-call authorization.
