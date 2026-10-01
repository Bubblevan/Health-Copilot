# MEM-3B0Q R4 Event-Value Prompt Tuning Smoke v3

Status: one-case development tuning diagnostic. `R4C-04` was already observed
in v1 and v2. This is not independent evidence and cannot be used to claim
generalization.

## Change Under Test

Keep the v2 cardinality-free schema, Harness-owned frozen slot policy,
atomwise quarantine, and all validators byte-for-byte unchanged. Clarify that
for `PURCHASE_EVENT`, the model's `value_span` should name the event/action,
not a phrase that only states when it occurred. Temporal expressions remain
un-normalized by this adapter; no relative-date calculation is added.

## Run

One fresh local Qwen request for `R4C-04`; 256 completion tokens, temperature
0, seed 42, thinking disabled, direct loopback to the pinned llama.cpp
endpoint, no retries. The result is a tuning smoke only. If it passes, the
next evaluation must use a newly authored, non-overlapping event diagnostic
set with the prompt locked before sending any request.

No public benchmark, MemoryStore write, revision materialization, or clinical
workflow is part of this smoke.
