# MEM-Q1 PropMem-TVR Paired LoCoMo Closeout

## Experiment

- Benchmark: official LoCoMo locomo10.json, 10 conversations and 1,986 QA; dataset SHA256 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4.
- MemEval commit: 807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4; LoCoMo commit: 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376.
- Reader/writer: local Qwen3-8B Q4_K_M. Embedding: Qwen3-Embedding-0.6B local CUDA FP16. Judge: NONE. Hosted APIs: NONE.
- PropMem is the pinned upstream implementation with only the audited local provider/embedding compatibility patch; temporal boost and knowledge updates are disabled.
- The answer adapter uses the pinned PropMem prompt templates and response schema verbatim while keeping transport exceptions observable for exact infra retries.
- TVR is deterministic and query-time only, operates on already-retrieved propositions, requires the same entity and cosine >= 0.85, does not mutate the cache, and makes no extra LLM calls.
- Chunk session provenance is a metadata-only sidecar derived from official LoCoMo turn IDs; chunk text/order and embedding vectors were parity-checked.
- Shared ingestion cache SHA256: 8c355eba569dc416f0bb4a7cc50279350e22b6b9d878ecd0191329c0b2b4ab07.
- Blind question manifest SHA256: 38546c06108cfea3fb65cb6bab1edce090b41f9f30d6616d500bd9d8a8923a2c. Predictions were frozen before scoring.

## Results

| Metric | FullContext | PropMem-local | PropMem + TVR | TVR delta |
|---|---:|---:|---:|---:|
| Overall Token-F1 | 0.4413 | 0.3814 | 0.3789 | -0.0025 |
| Factual Token-F1 | 0.4088 | 0.1671 | 0.1647 | -0.0025 |
| Temporal Token-F1 | 0.3217 | 0.2079 | 0.2079 | -0.0000 |
| Inferential Token-F1 | 0.1158 | 0.0970 | 0.0830 | -0.0140 |
| Multi-hop Token-F1 | 0.5229 | 0.2692 | 0.2621 | -0.0071 |
| Adversarial Token-F1 | 0.4641 | 0.9148 | 0.9215 | +0.0067 |
| Normalized EM (all; empty-gold uses refusal semantics) | n/a | 0.2573 | 0.2603 | +0.0030 |
| Answerable-only normalized EM | 0.1855 | 0.0713 | 0.0713 | +0.0000 |
| Empty-gold adversarial refusal accuracy | 0.4662 | 0.9032 | 0.9167 | n/a |
| Mean reader prompt tokens | 19642 | 2994 | 3001 | n/a |
| Context reduction vs FullContext | n/a | 84.8% | 84.7% | n/a |

FullContext normalized EM in its source report is answerable-only (286/1,542); the all-case refusal-aware EM above is computed for the two paired arms.

## Paired Uncertainty

- Overall paired delta: -0.002525; 95% stratified paired bootstrap CI [-0.009865, +0.004854] (10,000 replicates, seed 42).
- Temporal paired delta: -0.000033; 95% stratified paired bootstrap CI [-0.019631, +0.018955].

## Efficiency

- Shared ingestion: 3.86 hours; writer calls 272; embedding calls 1106; 7884 propositions; 889 chunks; store size 42246506 bytes.
- Paired QA wall: 8.90 hours; local LLM calls 4244; hosted calls 0; permanent infra failures 0; retries 0.
- Mean reader latency: PropMem 8.62 s; TVR 7.14 s.
- Mean retrieval latency 350.89 ms; mean TVR resolver overhead 0.475 ms.

## Integrity

- PropMem predictions SHA256: a759491abdd844d29a3478e054f6f4f96529a5492d600843223e459834e12e65.
- PropMem+TVR predictions SHA256: f71e6f86c372c4eda911779373c21a67ae97312349afdc9aea971d85182af1ca.
- Reader model SHA256: d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785; llama.cpp binary SHA256: 3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb.
- Reader runtime KV placement: cpu; 8081 command line is pinned in run_config.json. Reader layers remain on GPU; both arms shared this endpoint.
- Retrieval signatures identical: True; question ID sets identical: True.
- The pre-existing 8092 service was left untouched; the owned 8081 endpoint served this run.

## Limitations

- Local Qwen runtime differs from MemEval's published GPT-backed LoCoMo protocol; published PropMem 0.605 is an external historical coordinate only.
- TVR is a query-time adaptation over upstream PropMem, not a new proposition-memory algorithm. Its similarity primitive follows the pinned PropMem update threshold.
- LoCoMo alone is not evidence of medical-domain transfer.

## Resume Claim Eligibility

- NO: RESUME_METHOD_IMPROVEMENT_CLAIM=NO
