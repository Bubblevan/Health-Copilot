# MEM-1D1 Frozen 10-Case Diagnostic

- Run: `mem1d1-frozen-10-20260927`
- Gate: `MEM1_FROZEN_10_CASE_DIAGNOSTIC=YES`
- Dataset: `longmemeval_s` revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`
- Dataset SHA256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- Selection SHA256: `5a38ff79d79be6a9db531227d63d6dc22dc619d11d2c701b2b4cc0295f04c911`
- Prediction SHA256: `b90d6d6e7b08a720d358b14c114c553b432f4bf488fb2773b8060056cc34125f`
- Matrix: `5 systems x 10 questions = 50 predictions`
- TEST access: `false`; hosted calls: `0`; judge calls: `0`; native answer head: `not run`
- Case split: `1cea1afa` is KNOWN_GATE_CASE; the other nine are FRESH_DIAGNOSTIC_CASES.
- The manifest contains no abstention cases; no abstention-performance claim is made.
- Token F1 is deterministic lexical overlap, not semantic correctness. This small diagnostic is not a product ranking or benchmark claim.
- Post-run finalization repair (`run_mem1.py` SHA256 `473025878049deceef6e8913dae8999a6dc1a1877e17edc5aba22f0e1a87b8de`) separated ContextBundle content validation from SHA-sidecar freezing. It did not modify the 50 prediction, bundle, call-ledger, or warning-ledger rows and made no model calls; it only validated/froze those artifacts and regenerated deterministic summaries.

## Frozen Evidence

| Artifact | SHA256 |
|---|---|
| `context_bundles.jsonl` | `18f38700f7cc0b3b44ab45c8c29007a21c44b79e600c07254665cebb9bb9e26c` |
| `predictions.jsonl` | `b90d6d6e7b08a720d358b14c114c553b432f4bf488fb2773b8060056cc34125f` |
| `call_ledger.jsonl` | `480ab337bdd02ce8f573d0737d61931224698d338a8bcd39e2b0f3ac319b946b` |
| `baseline_warnings.jsonl` | `838de6f71474fcab98ce88cb06797aa75cfb1a0e9b000529e311fd3f402dfb7e` |

## All 10 And Fresh 9

| System | Cases | Token F1 | Precision | Recall | Norm. EM |
|---|---|---:|---:|---:|---:|
| fullcontext | all_10 (n=10) | 0.066 | 0.058 | 0.437 | 0.000 |
| fullcontext | fresh_9 (n=9) | 0.073 | 0.064 | 0.486 | 0.000 |
| openclaw | all_10 (n=10) | 0.136 | 0.091 | 0.275 | 0.000 |
| openclaw | fresh_9 (n=9) | 0.151 | 0.101 | 0.306 | 0.000 |
| mem0 | all_10 (n=10) | 0.114 | 0.082 | 0.195 | 0.000 |
| mem0 | fresh_9 (n=9) | 0.127 | 0.091 | 0.216 | 0.000 |
| simplemem | all_10 (n=10) | 0.058 | 0.041 | 0.114 | 0.000 |
| simplemem | fresh_9 (n=9) | 0.064 | 0.045 | 0.127 | 0.000 |
| propmem | all_10 (n=10) | 0.203 | 0.130 | 0.474 | 0.000 |
| propmem | fresh_9 (n=9) | 0.189 | 0.123 | 0.415 | 0.000 |

## Per-Category

| System | Category | N | Token F1 | Precision | Recall | Norm. EM |
|---|---|---:|---:|---:|---:|---:|
| fullcontext | single-session-user | 2 | 0.055 | 0.029 | 0.500 | 0.000 |
| fullcontext | single-session-assistant | 1 | 0.073 | 0.040 | 0.400 | 0.000 |
| fullcontext | single-session-preference | 2 | 0.135 | 0.186 | 0.272 | 0.000 |
| fullcontext | multi-session | 1 | 0.125 | 0.067 | 1.000 | 0.000 |
| fullcontext | temporal-reasoning | 2 | 0.032 | 0.017 | 0.214 | 0.000 |
| fullcontext | knowledge-update | 2 | 0.008 | 0.004 | 0.500 | 0.000 |
| openclaw | single-session-user | 2 | 0.431 | 0.286 | 0.875 | 0.000 |
| openclaw | single-session-assistant | 1 | 0.500 | 0.333 | 1.000 | 0.000 |
| openclaw | single-session-preference | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| openclaw | multi-session | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| openclaw | temporal-reasoning | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| openclaw | knowledge-update | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| mem0 | single-session-user | 2 | 0.333 | 0.250 | 0.500 | 0.000 |
| mem0 | single-session-assistant | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| mem0 | single-session-preference | 2 | 0.080 | 0.052 | 0.173 | 0.000 |
| mem0 | multi-session | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| mem0 | temporal-reasoning | 2 | 0.158 | 0.107 | 0.300 | 0.000 |
| mem0 | knowledge-update | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| simplemem | single-session-user | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| simplemem | single-session-assistant | 1 | 0.455 | 0.294 | 1.000 | 0.000 |
| simplemem | single-session-preference | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| simplemem | multi-session | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| simplemem | temporal-reasoning | 2 | 0.062 | 0.056 | 0.071 | 0.000 |
| simplemem | knowledge-update | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| propmem | single-session-user | 2 | 0.453 | 0.293 | 1.000 | 0.000 |
| propmem | single-session-assistant | 1 | 0.500 | 0.333 | 1.000 | 0.000 |
| propmem | single-session-preference | 2 | 0.094 | 0.057 | 0.269 | 0.000 |
| propmem | multi-session | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| propmem | temporal-reasoning | 2 | 0.053 | 0.036 | 0.100 | 0.000 |
| propmem | knowledge-update | 2 | 0.167 | 0.100 | 0.500 | 0.000 |

## Memory And Context Diagnostics

| System | Recall@5 | Recall@10 | MRR | Provenance coverage | Reader context tokens | Embedding tokens | Reader prompt tokens | Retrieval ms | Ingestion ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fullcontext | 1.000 | 1.000 | 1.000 | 10/10 | 106485.100 | 106486.100 | 106564.600 | 0.000 | 0.000 |
| openclaw | 0.000 | 0.000 | 0.000 | 10/10 | 7173.900 | 7174.900 | 7253.300 | 265.119 | 9717.299 |
| mem0 | - | - | - | 0/10 | 442.600 | 443.600 | 523.000 | 149.743 | 542570.956 |
| simplemem | - | - | - | 0/10 | 2255.900 | 2256.900 | 2335.300 | 17715.961 | 1020887.784 |
| propmem | 0.693 | 0.747 | 0.665 | 10/10 | 3530.700 | 3531.700 | 3610.100 | 281.886 | 1005175.959 |

Category counts (fixed manifest, no rebalancing): single-session-user=2, single-session-assistant=1, single-session-preference=2, multi-session=1, temporal-reasoning=2, knowledge-update=2

FullContext session recall/MRR represent trivial full-history context coverage, not retrieval performance. Null retrieval metrics reflect unavailable source-session provenance, not zero.

## SimpleMem Fidelity

Official source: `v0.1.0` / `7da777f56a15db81bb261d296c89cad5915e8d67`.
Planning enabled for all cases: `True`; reflection settings: `['True']`.
Per-question traces are retained in `predictions.jsonl` and `mem_1d1_case_review.json`; no individual lexical/structured result count is required to be nonzero.

| Semantic | Keyword | Structured | Merge/deduplicate | Reflection | Native answer head calls all suppressed |
|---:|---:|---:|---:|---:|---|
| 77 | 10 | 10 | 28 | 10 | True |

## Local Usage And Warnings

| System | Reader calls | Qwen local ms | Embedding calls | Embedding tokens | Baseline-internal warnings | Infra failures |
|---|---:|---:|---:|---:|---:|---:|
| fullcontext | 10 | 1232714.669 | 0 | 0 | 0 | 0 |
| openclaw | 10 | 53888.183 | 628 | 1679989 | 0 | 0 |
| mem0 | 10 | 5156175.305 | 1851 | 66107 | 8 | 0 |
| simplemem | 10 | 10337699.983 | 87 | 93290 | 16 | 0 |
| propmem | 10 | 9796184.827 | 2322 | 2049048 | 0 | 0 |

Embedding accounting is local `Qwen/Qwen3-Embedding-0.6B` (local_transformers); hosted API spend is `$0`.
Baseline-internal recoveries are reported separately from infrastructure failures. No causal failure labels were added; `failure_attribution_hint` remains heuristic.

Case-level review artifact: `docs/research/memory/mem_1d1_case_review.json`.
Run artifact directory: `D:\MyLab\Jianli\Health-Copilot\runs\memory\mem1\mem1d1-frozen-10-20260927`.

No local Qwen judge, 102-DEV run, TEST, native-answer track, M10-Flat, RevMem, Memora/FAMA, LongMemEval-V2, SFT or RL was run.
