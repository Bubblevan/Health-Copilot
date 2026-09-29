# MEM-3B0 - Revision Identity Protocol

Status: method freeze before identity-model calls. This stage consumes frozen FlatProp only and stops for Human Reflection.

## Starting Point

- Canonical main at start: `656dca96dea76ee63e6fea59c0aa264a877b9c9e`.
- Fresh topic: `mem3b0-revision-identity-20260930`.
- At branch creation, `origin/main`, local `main`, HEAD, and merge-base were `656dca96dea76ee63e6fea59c0aa264a877b9c9e`, ahead/behind `0/0`. Before execution preflight, `origin/main` advanced to `f2632b69a9e15a7ce08b58dc73ca1c76d379c50b` with RAG E5-B2 results; this latest main was fast-forwarded into the Memory topic and is the current protocol base. No peer topic branch was merged directly. Tracked and staged state were clean at branch creation. Pre-existing `.cache/`, `.pytest-temp-mem3a2s-20260929/`, `.pytest-tmp-mem3a3-finalverify-20260929/`, and `.tmp-mem3a3-pytest/` were untracked and are preserved.
- Frozen input: `runs/memory/mem3/mem3a3r-recursive-flatprop-frozen-10-20260929/flatprop_inventory.jsonl`, 8,112 propositions, SHA-256 `250a24a10b532cb01bd841b965511df019fc0e597ec1e85ee12d7830c166c5fe`.
- Upstream gate: `MEM3A3R_RECURSIVE_FLATPROP_FROZEN_10_DIAGNOSTIC=YES`.
- Global gates: `MEDMEMORYBENCH_COMPATIBILITY_AUDITED=YES`; `LONGMEM_MEMORY_MECHANISM_FROZEN=NO`.

## Question and Boundary

The model proposes which propositions refer to the same mutable semantic slot. It does not decide which observation is current, infer a revision chain, or execute a MemoryStore operation. The frozen MemoryStore and all 8,112 input propositions remain unchanged and active at version 1 with no supersedes link.

Identity calls receive exactly `memory_id`, `proposition_text`, `source_authority`, and `scope_id`. They never receive `observed_at`, benchmark questions, question dates, gold answers, answer-session IDs, question type, retrieval rank, predictions, or correctness. The source inventory is read-only and is projected into this four-field payload before each request.

## Frozen Method

- Identity contract: `revision-identity-v1`.
- Model: frozen local Qwen3-8B Q4_K_M, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- Runtime: pinned llama.cpp 10068 / build `571d0d540`, loopback `127.0.0.1:8081`, 131,072 context, 99 GPU layers, Flash Attention, Q4_0 GPU KV; no hosted fallback.
- Generation: temperature 0, seed 42, thinking disabled, `max_tokens=8192`.
- Initial batches: 32, sorted by `scope_id`, then `memory_id`.
- Each response has one dynamic-schema row per batch ID. Missing, duplicate, unknown IDs, extra fields, invalid kinds, and invalid keys are fatal.
- Only `finish_reason=length` causes the incomplete parent result to be discarded and the batch deterministically bisected left then right. No request is retried; a one-item length stop is fatal.
- Key normalization is deterministic surface normalization only: NFKC, strip, lowercase, whitespace/hyphens to underscores, repeated underscores collapsed, then lower-snake syntax validation. There is no alias dictionary or fuzzy merge.
- Candidate groups include `SINGLETON_STATE` only and use exact `(scope_id, subject_key, attribute_key)`. The frozen candidate-group artifact contains only scope, identity keys, memory IDs, and proposed values. All observations stay active. Timestamps, sessions, and source authority are joined only after identity records, candidate groups, call ledger, and core identity manifest are frozen.
- Slot-fragmentation and collision diagnostics are deterministic lexical heuristics; they never merge keys or split groups.

## Forbidden Work

No embeddings, retrieval, reader-answer, judge, hosted API, LongMemEval 102 DEV, public TEST, MedMemoryBench, benchmark answer metrics, MemoryStore writes, supersession, UPDATE, DELETE, M10-Flat, RevMem, or MEM-3B1 work occurs in this stage.

Before critical-case inspection, freeze `revision_identity_records.jsonl`, `candidate_revision_groups.json`, `identity_call_ledger.jsonl`, and `identity_run_manifest.json` with SHA sidecars. Then inspect the Instagram and gym examples plus negative, event, preference, fragmentation, and collision controls. Semantic quality is reviewed by Human Reflection and is not a structural completion gate.
