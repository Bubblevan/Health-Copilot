# Event-Value Confirmation v1r2

Status: same locked three-case dataset, prompt, model, request bodies, answer
schema, and scoring as v1. Earlier v1 and v1r1 attempts are preserved as
preflight-only failures with zero completion POSTs.

## Runtime Access

The Windows host denied listener/process queries to the non-elevated runner.
This attempt runs the exact frozen process and service preflight with elevated
OS query permission. The process is still required to pass the same strict
checks: one loopback listener, pinned llama-server path and SHA256, pinned model
path and SHA256, exact command-line options, build `b10068-571d0d540`, served
model, context, sampler defaults, and idle slot. Elevation changes OS query
access only; it does not relax any validator or alter the model request.

The v1r2 lock pins the v1r1 request map and dataset SHA byte-for-byte, as well
as prior zero-POST artifacts. Any preflight mismatch aborts before inference.

## Execution

One local request per confirmation case, max 256 completion tokens, temperature
0, seed 42, thinking disabled, direct loopback only, zero retries. No hosted
API, key, judge, embedding service, MemoryStore write, or clinical content.

Results remain a small synthetic confirmation only, not public benchmark or
generalization evidence.
