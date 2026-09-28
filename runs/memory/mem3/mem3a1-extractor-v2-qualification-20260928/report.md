# MEM-3A.1 - Provenance-by-Reference Extractor v2 Qualification

Gate: MEM3A1_EXTRACTOR_V2_QUALIFIED=NO

This is a writer-contract qualification, not a benchmark run. It performs no proposition materialization, embedding, retrieval, reader calls, label access, answer scoring, revision grouping, MEM-3B, or RevMem work.

## Frozen Protocol

- Source: frozen MEM-2C raw-span-v1 inventory; SHA256 93414251555c003e3acaa51968f5cf07a85ba3a189743a66301dd20ed7e09c26.
- Source-span catalog sessions and tokenizer preflight: 477/477; prompt + 4096 output reserve fits the 131072 context; truncations: 0.
- Frozen qualification identities: 12; selection SHA256 369bc2828acc9198fdf4af741d424303ed002c58691fe62c1a169599da4c8ddd.
- Prompt SHA256: 0aea4338d5de1e8dac753e3aae98caf36461cc15b1cf8d782a36f298f63cf627; contract SHA256: fee5ae49c13d7b55d75c29af38146c2a9e079ac7ca24e5aeff151dee7a31b666.
- Local model: health-memory-qwen3-8b; model SHA256 d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785; endpoint http://127.0.0.1:8081/v1.
- Hosted calls: 0; TEST access: false; benchmark-label access: false.
- Writer reproducibility mode: ARTIFACT_FROZEN_NOT_BITWISE_REPLAY.

## Qualification Results

- Initial writer calls attempted: 1; retries: 0.
- Successful frozen packets: 0/12.

## First Failure

- Failure detail: {"code": "UNEXPECTED_OUTPUT_FIELD", "detail": "unexpected root field", "evidence_ref": null, "field": "choices", "proposition_index": null}.
- The single HTTP 200 response body was captured before validation. Inspection found the standard llama.cpp completion envelope; the generated JSON is nested at `choices[0].message.content`, but this runner passed the entire envelope to the proposition-packet validator, which expected `{ "propositions": [...] }` at the HTTP root. This is a Harness response-unwrapping defect, so the model output was not qualified either way.
- Stop condition honored: no offline reparsing/replay, no retry, and no calls for the remaining 11 selected sessions. The terminal gate remains NO. The exact body is retained only at the local ignored cache path recorded in `writer_call_ledger.jsonl`.

## Interpretation Boundary

Counts above are descriptive only and have no pass thresholds. Provenance roles, exact source quotes, turn/span indices, offsets, and content hashes are harness-derived from the frozen span catalog. Human semantic-review fields remain null. No quality or benchmark-ranking claim is made.

Raw model responses remain only in the git-ignored local forensic cache; committed artifacts contain response hashes and normalized review packets, not raw response bodies.

STOP: do not run the remaining 465 sessions, full MEM-3A, revision materialization, embedding, retrieval, reader evaluation, TEST, or MEM-3B.
