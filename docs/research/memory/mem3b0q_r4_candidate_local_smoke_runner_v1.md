# MEM-3B0Q R4C-05 Local One-Shot Runner v1

Status: `RUNNER_REVIEW_REQUIRED`; the runner and lock are for independent
review only. This document is not authorization to make a completion POST.

## Scope

This runner consumes only the frozen synthetic case `R4C-05`. It issues one
candidate-bounded proposition-extraction request to the already-running local
Qwen3-8B endpoint on `127.0.0.1:8081`; it does not start, stop, or reconfigure a
service. It does not call an embedding model, judge, MemoryStore, Evidence
store, or hosted API. Its result is a runtime/contract diagnostic, not
extractor qualification, a benchmark result, or evidence for a performance
claim.

The request and response schema are pinned by SHA-256 in
`mem3b0q_r4_candidate_local_smoke_lock_v1.json`. The lock also pins the runner,
all method/runtime dependencies, the frozen R4 pack, the candidate control
pack, the development-control inventory, and the no-model grammar-preflight
report. It records one POST maximum, zero retries, no API key, no proxy use,
and no state mutation. The specific llama.cpp server/model hashes and runtime
settings are checked against the existing R4 read-only process/service
preflight.

## Fail-Closed Boundary

- The run directory is reserved once and may never be reused. Request and
  result artifacts are created exclusively; an existing path blocks execution.
- Only allowlisted GET paths on the frozen loopback host/port may pass. HTTPS,
  other hosts, other paths, methods, request bytes, or headers are rejected
  before forwarding.
- The only permitted POST must byte-match the locked request. Process identity
  and service state are checked at initial preflight, immediately before the
  POST, and after the response. A second POST is rejected.
- `http.client.HTTPConnection` connects directly to the literal loopback
  address; it does not consult environment proxy settings. No credential is
  read or required.
- A response without a valid HTTP 200 JSON completion envelope is an
  `INFRA_FAILURE`. Once model content exists, malformed JSON/schema, candidate
  validation errors, and any non-exact atom multiset are `QUALITY_FAILURE`.
  Infrastructure failures are not converted into quality zeroes.
- A pass requires the complete normalized atom multiset to equal the sole
  expected atom, including scope, typed IDs, cardinality, value, and exact
  owner/object/attribute/value witness text and offsets. It is recorded only
  as `R4C_CANDIDATE_LOCAL_SMOKE=YES`.

## Schema And Runtime Evidence

The candidate-specific grammar preflight is
`mem3b0q_r4_candidate_local_smoke_grammar_preflight_v1.json`:

- Schema SHA-256: `a5ea71399f2787581e53c0c2f669fc0c69b0c0a52184ac8f4b4480f18fd2c695`.
- Pinned CLI: `10068 (571d0d540)`, SHA-256
  `48566cf6e2969464b799dbcac7393b3549f9efd6884688074dc803125dbafa85`.
- Valid schema reached the intentionally missing-model failure without a
  schema/grammar error; malformed schema was rejected before that failure.
- Model loaded: `false`; inference: `false`; endpoint called: `false`;
  hosted API called: `false`.
- Sampler initialization: `NOT_VERIFIED`.

This establishes the pinned CLI's pre-model-load schema/grammar conversion
path for the exact candidate schema. It does not prove server-side sampler
initialization. A future reviewed one-case request is the first endpoint-level
sampler smoke; any failure is captured once, never retried, and never used to
tune this frozen contract.

## Verification

The offline runner tests cover lock/sidecar tampering, exact request binding,
missing `OPENAI_API_KEY`, host/path/method/body/header rejection, HTTPS denial,
single-POST enforcement, process-change blocking, run-directory reuse refusal,
duplicate JSON-key rejection, and exact CLI-flag gating. They use fake local
connections and snapshots; they do not contact the endpoint. The runner suite
passes `18` tests; the combined R4/factorized/pairwise regression set passes
`163` tests. Pytest emitted one cache-directory permission warning; execution
completed successfully.

Current gates remain:

```text
R4C_CANDIDATE_LOCAL_SMOKE=NOT_RUN
MEM3B0Q_CANDIDATE_BOUNDED_EXTRACTOR_READY=NO
MEM3B0Q_MEM3B1_READY=NO
MEMORY_PUBLIC_CLOSEOUT=NO
```

A separate independent review must inspect the final runner, lock, and test
evidence and explicitly authorize at most one `R4C-05` local completion POST.
Until then, do not invoke `--run-r4c-05-once`.
