# MEM-1D3 Question-Date Contract Repair

- Gate: `MEM1D3_QUESTION_DATE_CONTRACT_REPAIRED=YES`
- Scope: exact frozen MEM-1D1 ten DEV IDs × five systems; 50 reader-only calls; no memory ingestion/retrieval, embedding, judge, hosted API, TEST QA payload, 102 DEV, M10-Base, RevMem, or RL.
- Root cause: official LongMemEval records include `question_date`, but the pinned compatibility normalizer previously discarded it; D1 temporal outputs therefore lack the query-time anchor.
- Interpretation: `MEM1D_TEMPORAL_INTERPRETATION_INVALIDATED_BY_MISSING_QUESTION_DATE=YES`; this invalidates only D1 temporal causal interpretation, not the frozen D1 artifacts.
- Question dates are copied exactly from the official source record; no dates are synthesized or inferred.

## Frozen Inputs And Runtime

- DEV records with official question dates and source-record hashes: `10/10`.
- Frozen ContextBundles re-used and hash-verified: `50/50` (no bundle copy or reconstruction).
- Reader: `health-memory-qwen3-8b`; model SHA256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- llama.cpp: `version: 10068 (571d0d540)
built with Clang 20.1.8 for Windows x86_64`; binary SHA256 `3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb`.
- Runtime: 131072 context, 99 GPU layers, Flash Attention, Q4_0 GPU KV; generation temperature 0, seed 42, thinking false, max new tokens 256.
- Reader-v2 raw template SHA256: `2419137d93616a3eecf621337abc81c69fd0572a8a1fa825aaf067187fa7458b`; canonical message-template SHA256: `a06cf171967ade9e8b3b2b90525ea635e538ed4383c31d2f3d64a28d135d2041`.
- Frozen v1 historical template SHA256: `0ff70b000bd4b43db85ae587731fea35b691febc815cfbb57c2acb1c8b295b2b` (not overwritten).
- Hosted API calls: `0`; judge calls: `0`; embedding/model initialization: `0`; memory-system calls: `0`.

## FullContext Prompt Fit

Prompt lengths use the running frozen llama.cpp `/apply-template` plus `/tokenize` endpoints with the reader's chat template and `enable_thinking=false`.

| Question ID | Prompt tokens | Reserve | Context limit | Fits | Truncated |
|---|---:|---:|---:|---:|---:|
| 1cea1afa | 108199 | 256 | 131072 | True | False |
| 1c549ce4 | 106197 | 256 | 131072 | True | False |
| 778164c6 | 103534 | 256 | 131072 | True | False |
| fca70973 | 107580 | 256 | 131072 | True | False |
| a82c026e | 107751 | 256 | 131072 | True | False |
| gpt4_e061b84g | 105343 | 256 | 131072 | True | False |
| gpt4_f420262c | 106767 | 256 | 131072 | True | False |
| 8550ddae | 107887 | 256 | 131072 | True | False |
| 06878be2 | 106775 | 256 | 131072 | True | False |
| c4ea545c | 106233 | 256 | 131072 | True | False |

## Paired Counterfactual

- Temporal cases (`gpt4_e061b84g`, `gpt4_f420262c`): `10` system×question rows; changed answer strings `8/10`; mean raw ΔF1 `-0.0242`.
- Non-temporal questions (other eight): `40` system×question rows; changed answer strings `23/40`; mean raw ΔF1 `-0.0272`.
- These are lexical diagnostics only, not improvement claims. All 50 rows remain `PENDING_HUMAN_REVIEW`; no outcome or failure-locus labels were assigned automatically.
- Knowledge-update cases `1cea1afa` and `c4ea545c`, multi-session case `1c549ce4`, and preference cases remain separate axes. D1 temporal answers are historical diagnostics; subsequent temporal interpretation must use reader v2.

## Evidence Integrity

- MEM-1D1 artifact/sidecar and D2 provenance-overlay hashes unchanged: `True`.
- Reader-only calls completed: `50/50`; call-ledger rows: `50`.
- FullContext prompt fit and non-truncation: `True`.
- Question-date source audit: `docs/research/memory/mem_1d3_question_dates.json`.
- Paired artifact: `docs/research/memory/mem_1d3_reader_counterfactual.json`.
- New run directory: `runs/memory/mem1/mem1d3-reader-v2-20260928`; ContextBundles are referenced by hash, not copied.
- Research positioning updated in `docs/research/memory/baseline_sources.md`; RevMem is only a harness-native revision-aware memory experiment, not a novelty/SOTA claim.

STOP: do not proceed to 102 DEV, M10-Base, RevMem, or RL without a separate human review/authorization.
