# MEM-3B0Q Atomwise Candidate Devset Protocol v1

Status: `FROZEN_OFFLINE_DIAGNOSTIC`; full case requests, atomwise adapter,
runtime, post limits, and scoring are pinned by
`mem3b0q_r4_atomwise_candidate_devset_lock_v1.json`.

## Scope

This is a development diagnostic over the six already-frozen,
project-authored synthetic candidate controls in
`mem3b0q_r4_candidate_control_pack_v1.json`. It preserves the original
one-case smoke protocol and result unchanged. `R4C-05` reuses its previously
recorded response; the five remaining cases each receive at most one local
completion request. The user's instruction to continue authorizes this bounded
development run; it does not turn the old one-case review into broader
benchmark approval.

No public TEST split, LongMemEval, Memora, MedMemoryBench, or clinical content
is involved. The candidate prompt explicitly mentions `tablet` and `basket`,
so this is a prompted contract-compliance diagnostic, not unbiased unknown-
entity generalization evidence.

## Frozen Stack

| Role | Configuration |
|---|---|
| Reader / proposal model | Frozen local Qwen3-8B Q4_K_M GGUF |
| Memory system | Candidate-bounded proposal + frozen per-atom v3 guard + development atomwise quarantine |
| Embedding | None |
| Judge | None |
| Endpoint | Pinned loopback llama.cpp on `127.0.0.1:8081` |
| Hosted API / credential | None / none |
| Generation | temperature `0`, seed `42`, max completion `256`, non-streaming, thinking disabled |
| MemoryStore mutation | None |

The original prompt, source texts, candidate construction, expected outputs,
and per-atom binding validators do not change. Only the handling of independent
semantic validation failures changes: isolate the failed atom and preserve
other atoms that independently pass the same frozen checks. JSON envelope
corruption and inconsistent abstention remain fail-closed. Exact duplicate
admitted atoms are deduplicated.

## Cases and Accounting

The run covers all six case IDs `R4C-01` through `R4C-06` exactly once in the
aggregate. The existing response for `R4C-05` is replayed and must match the
newly built request hash byte-for-byte. No new R4C-05 POST is permitted. The
other five requests are sent sequentially, each with a separate one-shot
loopback guard, exact request-body hash, zero retries, runtime checks before
and after, and append-never output artifacts.

Every completed case reports:

- exact normalized atom-multiset match;
- expected atoms, admitted atoms, quarantined atoms, and per-atom reasons;
- whole-document v3 guard acceptance for comparison;
- prompt/completion tokens and local request latency when supplied by the server;
- HTTP/runtime status separately from proposal quality.

Aggregate atom precision and recall are multiset-based over complete normalized
atom signatures. An infrastructure failure terminates further POSTs and leaves
remaining cases explicitly `NOT_RUN`; a quality failure does not change the
method and does not prevent collection of the other already-locked DEV cases.

## Evidence Boundary

These six authored controls are small engineering diagnostics. Their exact
match, precision, recall, or error counts must not be presented as public
benchmark performance, external generalization, or proof of temporal memory
quality. They only qualify the extractor/admission boundary before any larger
public DEV run. They do not establish a LongMemEval result, revision
materialization, stale-memory reduction, or medical transfer.

No prompt, alias table, control, threshold, expected output, or split may be
changed after observing these six outputs. Any new method requires a distinct
version and fresh non-overlapping diagnostic cases before another comparison.
