# MEM-3B0Q R4 Cardinality Authority Devset v2

Status: development-only protocol. The lock pins all six request bodies, model
and server binaries, runtime settings, dependencies, and one-shot accounting.
This protocol was authored after inspecting v1 results; it is not an independent
or held-out evaluation.

## Question

Does removing a deterministic slot-cardinality decision from the model's output
contract improve grounded proposal admission without weakening the existing
locality, typed-anchor, event, or atomwise validators?

## Single Isolated Change

The frozen slot policy maps each known `(object_id, attribute_id)` pair to one
of `SINGLE_VALUE_AT_A_TIME`, `MULTI_VALUE_CONCURRENT`, or
`EVENT_OR_NOT_STATE`. The model schema no longer contains
`cardinality_proposal`. After validating candidate IDs, a deterministic
adapter derives cardinality from that closed table and adds the internal field
needed by the unchanged downstream validators. Unknown pairs fail closed.

No prompt aliases, source sentences, candidate spans, expected atoms, admission
rules, or downstream guards change. The completion budget stays at 256 tokens
to isolate this contract change; a truncation remains a quality failure and
will not be retried in this run.

## Run Design

- Six fresh one-shot local Qwen requests: `R4C-01` through `R4C-06`.
- No response is replayed from v1; each uses the v2 schema and prompt.
- Temperature `0`, seed `42`, thinking disabled, max completion `256`.
- Direct loopback only at the pinned llama.cpp endpoint; one post per case,
  zero retries, stop on infrastructure failure.
- No hosted service, API key, embedding model, judge, MemoryStore mutation, or
  clinical content.
- Raw request and response, per-case score, pre/post runtime checks, and
  append-never manifest are retained.

## Reporting

Report all six cases, exact atom matches, atom precision/recall, quarantined
reasons, truncation, and infrastructure separately. Compare v1 and v2 only as
development diagnostics. These synthetic prompted controls are not a public
benchmark, unbiased generalization estimate, or evidence of temporal-memory
quality. No public LongMemEval, Memora, MedMemoryBench, 10-case, or 102-case run
is authorized by this protocol.
