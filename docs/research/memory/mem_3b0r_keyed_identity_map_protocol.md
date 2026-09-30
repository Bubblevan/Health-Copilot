# MEM-3B0R - Keyed Identity Map and Local Recovery

Status: protocol frozen before MEM-3B0R identity-proposer calls. This stage changes the identity wire/recovery protocol only; it does not change revision-identity semantics.

## Lineage and Immutable Inputs

- Memory topic: `mem3b0-revision-identity-20260930`.
- Historical MEM-3B0 gate remains `MEM3B0_REVISION_IDENTITY_FROZEN_DIAGNOSTIC=NO`; its run directory and v1 validator are immutable.
- The v1 failure is evidence that an array with an ID enum does not enforce exactly-once coverage. It is not evidence about identity semantic quality.
- Frozen FlatProp input: `runs/memory/mem3/mem3a3r-recursive-flatprop-frozen-10-20260929/flatprop_inventory.jsonl`, exactly 8,112 rows, SHA-256 `250a24a10b532cb01bd841b965511df019fc0e597ec1e85ee12d7830c166c5fe`.
- No FlatProp regeneration, MemoryStore mutation, embeddings, retrieval, reader-answer, judge, hosted API, LongMemEval scoring, TEST, or MedMemoryBench run is permitted.

## Semantic Contract

The semantic identity prompt and definitions remain the frozen `revision-identity-v1` contract. Inputs are only `memory_id`, `proposition_text`, `source_authority`, and `scope_id`. No observed timestamp, benchmark question/date/gold, answer-session ID, question type, reader prediction, retrieval rank, or correctness label may enter the proposer request.

Allowed kinds remain `SINGLETON_STATE`, `SET_STATE`, `EVENT`, `NON_REVISIONAL`, and `UNKNOWN`. No temporal resolution, revision decision, supersession, or storage side effect is performed.

## Wire v2

Contract: `revision-identity-wire-v2-keyed-map`. The response is an `identities` object whose dynamic required property set is exactly the IDs in that request batch. Each keyed payload contains only `revision_kind`, `subject_key`, `attribute_key`, and `value_text`; it does not repeat `memory_id`. `additionalProperties=false` is set at both levels. Raw JSON is parsed with duplicate-object-key detection before a mapping is constructed.

Validator identity binds the frozen semantic contract SHA, wire-v2 contract SHA, validator source SHA, local-recovery implementation SHA, local-recovery policy SHA, and model artifact SHA. The validator and recovery implementation currently share one source module, so the two implementation SHA fields are expected to match; they remain separately named in the manifest to make their roles explicit. The historical v1 wire/validator remains unchanged.

## Frozen Model and Batching

- Model: local Qwen3-8B Q4_K_M, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- Runtime: pinned llama.cpp 10068 / `571d0d540`, loopback `127.0.0.1:8081`, 131,072 context, 99 GPU layers, Flash Attention, Q4_0 GPU KV.
- Generation: temperature 0, seed 42, thinking disabled, max completion 8,192 tokens; no hosted or alternate-model fallback.
- Initial batches contain 32 records in canonical `scope_id`, then `memory_id` order. Batch size is not tuned from the failed v1 response.
- Before inference, every dynamic request is token-counted locally through the frozen llama.cpp template/tokenizer. Record prompt-token and schema-token distributions and request byte sizes. Conservatively verify rendered prompt tokens plus dynamic-schema tokens plus the 8,192-token completion reserve fit the 131,072 context.

## Failure and Recovery Policy

Global integrity/provider/model failures stop the stage: source SHA or inventory corruption, duplicate frozen input IDs, MemoryStore mutation, model/runtime identity mismatch, label leakage, cache/artifact integrity failure, code/protocol mismatch, or inability to obtain a provider response. No same-request retries are allowed.

Local model-output failures are limited to the current batch: completion length, malformed assistant JSON, duplicate raw JSON object key, invalid/missing/unexpected root or identity keys, invalid payload fields, illegal kind, invalid subject/attribute syntax, or empty value. Discard the parent semantic output. For batch size greater than one, split the canonical contiguous batch in half, left first then right. Successful children are reused from the frozen local response journal. Record batch ID, parent ID, depth, error, input IDs, and child IDs.

For a single-record local model-output failure, emit the deterministic research-overlay record `UNKNOWN / unknown / unknown`, preserve the exact frozen proposition as `value_text`, and mark `identity_origin=HARNESS_UNKNOWN_FALLBACK` plus its structured `fallback_reason`. This is not a model prediction and is excluded from candidate singleton groups. Infrastructure, source, provider/model, and artifact failures never use this fallback.

## Freeze Order and Diagnostics

Every source ID must receive exactly one terminal identity record, either `MODEL_VALIDATED` or `HARNESS_UNKNOWN_FALLBACK`. Before timestamps or diagnostic case history are inspected, freeze identity records, identity call ledger, recovery lineage, and `identity_run_manifest.json`, all with SHA sidecars. Only afterward join timestamps, sessions, and source authority; build exact `(scope_id, subject_key, attribute_key)` candidate groups from model-validated singleton states; calculate counts and non-causal fragmentation/collision heuristics; and inspect Instagram `1cea1afa` and gym `c4ea545c`.

Semantic quality is separate from structural completion. Instagram/gym success, low fallback rate, low fragmentation, and high grouping coverage are not structural gate thresholds. No automatic key merging/splitting or MemoryStore materialization is allowed.

## Completion Gate

Set `MEM3B0R_REVISION_IDENTITY_OVERLAY_COMPLETE=YES` only when the frozen input SHA is valid, all 8,112 IDs have exactly one terminal identity, all local failures are resolved by deterministic subdivision or conservative singleton fallback, retries and hosted calls are zero, recovery lineage is auditable, no labels leak, MemoryStore is unchanged, embedding/retrieval/reader/judge/MedMemoryBench/102-DEV/TEST activity is zero, and all required artifact sidecars verify. A semantic-quality closeout is reported separately. Stop before MEM-3B1.
