# MEM-3B0Q R4 Event-Value Confirmation v1

Status: fresh, project-authored synthetic confirmation set. Pack, dataset hash,
three requests, Qwen model/server hashes, runtime, runner, and one-shot policy
are pinned in `mem3b0q_r4_event_value_confirmation_v1_lock.json` before any
request is sent.

## Purpose

Check whether the already-frozen v3 event-value prompt transfers from the
observed R4C-04 sentence to three non-overlapping event descriptions with
different owner mentions and temporal phrases. Expected event action is
`completed the purchase`; the time phrase is not the event value.

The prompt and event guard are unchanged from v3. The pack's diagnostic-only
time labels and expected atoms are not included in model requests. Relative and
absolute time expressions are not normalized or scored in this run.

## Run Controls

- Cases: `EVTCONF-01`, `EVTCONF-02`, `EVTCONF-03` from the frozen confirmation
  pack; none of their source sentences appears in the prior R4C pack.
- One fresh local Qwen request per case; temperature `0`, seed `42`, thinking
  disabled, max completion `256`.
- Direct loopback to pinned llama.cpp only, zero retries, stop after an
  infrastructure failure.
- No hosted API, API key, embedding model, judge, MemoryStore mutation, or
  clinical content.
- Preserve every request, response, score, runtime checkpoint, and manifest
  hash.

## Evidence Boundary

These three examples are a small synthetic confirmation set written after the
R4C-04 failure was observed. A positive result indicates transfer to these
new phrasings only; it is not a blinded or public benchmark and does not
establish natural-dialogue generalization, temporal-memory quality, revision
materialization, or medical transfer.
