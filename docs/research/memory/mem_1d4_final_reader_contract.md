# MEM-1D4 Minimal Date-Aware Final Reader Contract

- Gate: `MEM1D4_FINAL_READER_CONTRACT_FROZEN=YES`
- Selection rule: protocol-only; no answer-quality-based reader selection or tuning.
- Research position: controlled re-evaluation of public memory architectures under a unified fully-local model stack.
- Scope: frozen MEM-1D1 DEV ten questions × five system identities; reader-only; no memory architecture change.

## Contract

- Reader: `health-memory-qwen3-8b`, artifact SHA256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- Final contract SHA256: `57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3`.
- v3 raw template SHA256: `1c123f04a21fa3bb58a4ce3c2bbfad0347501e9f9ee9d561d0ee16de3468a510`.
- v3 canonical message-template SHA256: `8e44fc1461529a8e421da5a3e9744a8f84ab1fd280e1b2e1f3e369a1351948dd`.
- Official source: LongMemEval `9e0b455f4ef0e2ab8f2e582289761153549043fc` `src/generation/run_generation.py`, raw UTF-8 SHA256 `4f1eb3c69d7ad40f04065b9c0bc86f6582441018fc6ff751d162d66c95baf672`.
- Source alignment only: Health-Copilot keeps its own shared answer head; no exact official prompt reproduction claim.
- Structural finding: official `question_date` is required and copied exactly from the hash-verified D3 table.
- v3 equals the v1 answering policy plus only the `Current Date` field; there are no temporal examples, task/category hints, retrieval instructions, or reasoning directives.
- Permanent choice: date-complete, minimal, shared, category-agnostic, source-aligned; chosen by protocol, not DEV scores. Reader-v4 tuning on these results is prohibited absent an infrastructure or benchmark-contract defect.
- Explicit selection gate: `FINAL_READER_SELECTED_BY_PROTOCOL_NOT_DEV_SCORE=YES`.
- Contract enforcement: the file SHA is pinned in the shared validator, included in context-controlled run manifests, prediction rows, and cache identity; any mismatch fails closed. M10-Base, RevMem, 102 DEV, and held-out TEST runners must bind this same SHA.

## Execution

- Reader calls: `50/50`; systems: `fullcontext, openclaw, mem0, simplemem, propmem`.
- Memory-system calls: `0`; embedding: `0`; judge: `0`; hosted: `0`.
- ContextBundles: `50/50` verified from MEM-1D1, copied/rebuilt: `False`.
- D1/D2/D3 historical evidence byte/hash unchanged: `True`.
- TEST access: `False`.
- FullContext fit: `10/10`; every request had 256 output tokens reserved under the 131072 context limit.
- Token preflight uses the frozen llama.cpp `/apply-template` and `/tokenize`; server prompt usage is compared with preflight for each answer.

| Question ID | Preflight prompt tokens | Server prompt tokens | Reserve | Fits | Truncated |
|---|---:|---:|---:|---|---|
| 1cea1afa | 108161 | 108161 | 256 | True | False |
| 1c549ce4 | 106159 | 106159 | 256 | True | False |
| 778164c6 | 103496 | 103496 | 256 | True | False |
| fca70973 | 107542 | 107542 | 256 | True | False |
| a82c026e | 107713 | 107713 | 256 | True | False |
| gpt4_e061b84g | 105305 | 105305 | 256 | True | False |
| gpt4_f420262c | 106729 | 106729 | 256 | True | False |
| 8550ddae | 107849 | 107849 | 256 | True | False |
| 06878be2 | 106737 | 106737 | 256 | True | False |
| c4ea545c | 106195 | 106195 | 256 | True | False |

## Three-Way Diagnostic

The deterministic summary and 50-row review packet report v1, v2, and v3 token precision/recall/F1/normalized EM and exact-string change flags. These are descriptive prompt-sensitivity measurements, not a leaderboard or reader-selection criterion.
- Temporal slice: `10` system×question rows across `gpt4_e061b84g`, `gpt4_f420262c`.
- Non-temporal slice: `40` rows, reported separately.
- A changed answer string is not automatically a failure. Human semantic-review fields remain null; no automatic causal/failure labels were assigned.
- D2 provenance/evidence artifacts remain preserved; this stage does not assign temporal failure loci.

## Frozen Artifacts

- Final contract: `docs/research/memory/final_reader_contract.json` and SHA sidecar.
- v3 template: `docs/research/memory/shared_reader_v3_final.txt`.
- Review packet: `docs/research/memory/mem_1d4_reader_review.json`.
- Reader-only run: `runs/memory/mem1/mem1d4-reader-v3-20260928`.
- Run prediction SHA256: `641b55568cb763688f7fc3af37eeb5bf338664707f5178437864e48966432ec5`; call-ledger SHA256: `51c2634ef1882447593b72af8b6a3586253eef5de561f09e59295c0c369836c1`.

STOP: do not start MEM-1 102-case DEV, TEST, M10-Base, RevMem, or RL without a separately reviewed next-stage request.
