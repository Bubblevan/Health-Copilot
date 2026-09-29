# U1 Deterministic Integration Prototype

Run: `u1-synthetic-20260929-01`; fixture status: `SYNTHETIC_CONTRACT_FIXTURE; NOT_MEDICAL_BENCHMARK`.

This is a synthetic contract prototype, not a medical benchmark or a model-quality evaluation.
No provider was called and no training was run.

## Case outcomes

| Episode | Valid arms | Successful arms | A* | Single/Team union |
| --- | --- | --- | --- | --- |
| U1-NONE | NONE, TEAM | NONE, TEAM | NONE | PASS |
| U1-MEM | NONE, MEMORY, TEAM, MEMORY+TEAM | MEMORY, MEMORY+TEAM | MEMORY | PASS |
| U1-RAG | NONE, RAG, TEAM, RAG+TEAM | RAG, RAG+TEAM | RAG | PASS |
| U1-MEM-RAG | NONE, MEMORY, RAG, TEAM, MEMORY+RAG, MEMORY+TEAM, RAG+TEAM, ALL | MEMORY+RAG, ALL | MEMORY+RAG | PASS |
| U1-TEAM | NONE, TEAM | TEAM | TEAM | PASS |
| U1-MEM-TEAM | NONE, MEMORY, TEAM, MEMORY+TEAM | MEMORY+TEAM | MEMORY+TEAM | PASS |
| U1-ALL | NONE, MEMORY, RAG, TEAM, MEMORY+RAG, MEMORY+TEAM, RAG+TEAM, ALL | ALL | ALL | PASS |
| U1-OOD-INSUFFICIENT | NONE, TEAM | NONE, TEAM | NONE | PASS |
| U1-TEMPORAL-LEAKAGE | NONE, MEMORY, TEAM, MEMORY+TEAM | ∅ | ∅ | PASS |

## Gates

- Runtime/evaluation/privileged payloads are separate typed planes.
- All provider calls and token counters are zero; deterministic latency is zero.
- Abstract cost uses `u1-abstract-cost-v1`; no USD cost is synthesized.
- `episodes.jsonl` contains runtime fields only; OPD student and teacher records use separate JSONL files.
- ESL mapping is schema-only. No evaluation query or gold was accessed.
