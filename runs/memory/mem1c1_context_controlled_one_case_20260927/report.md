# MEM-1 Local Main Track Run Report

- Run: `mem1c1_context_controlled_one_case_20260927`
- Track: `main_local_only_context_controlled`
- Split: `DEV` (1 frozen DEV questions)
- Dataset SHA256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- Prediction SHA256: `42d231228db11175e8676ef961dd8fe384cdc03a2712bd2afe128bb5a380cc03`
- TEST access: `false`
- Hosted API: `NONE`; judge: `NONE`; required API key: `NONE`

| System | Quality N | Infra failures | Token F1 | Precision | Recall | Norm. EM | Abstention N | Abstention accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fullcontext | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | - |
| openclaw | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | - |
| mem0 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | - |
| simplemem | 1 | 0 | 0.1818 | 0.1000 | 1.0000 | 0.0000 | 0 | - |
| propmem | 1 | 0 | 0.3333 | 0.2000 | 1.0000 | 0.0000 | 0 | - |

## Category Metrics

| System | Category | N | Token F1 | Precision | Recall | Norm. EM |
|---|---|---:|---:|---:|---:|---:|
| fullcontext | single-session-user | 0 | - | - | - | - |
| fullcontext | single-session-assistant | 0 | - | - | - | - |
| fullcontext | single-session-preference | 0 | - | - | - | - |
| fullcontext | multi-session | 0 | - | - | - | - |
| fullcontext | temporal-reasoning | 0 | - | - | - | - |
| fullcontext | knowledge-update | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| openclaw | single-session-user | 0 | - | - | - | - |
| openclaw | single-session-assistant | 0 | - | - | - | - |
| openclaw | single-session-preference | 0 | - | - | - | - |
| openclaw | multi-session | 0 | - | - | - | - |
| openclaw | temporal-reasoning | 0 | - | - | - | - |
| openclaw | knowledge-update | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mem0 | single-session-user | 0 | - | - | - | - |
| mem0 | single-session-assistant | 0 | - | - | - | - |
| mem0 | single-session-preference | 0 | - | - | - | - |
| mem0 | multi-session | 0 | - | - | - | - |
| mem0 | temporal-reasoning | 0 | - | - | - | - |
| mem0 | knowledge-update | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| simplemem | single-session-user | 0 | - | - | - | - |
| simplemem | single-session-assistant | 0 | - | - | - | - |
| simplemem | single-session-preference | 0 | - | - | - | - |
| simplemem | multi-session | 0 | - | - | - | - |
| simplemem | temporal-reasoning | 0 | - | - | - | - |
| simplemem | knowledge-update | 1 | 0.1818 | 0.1000 | 1.0000 | 0.0000 |
| propmem | single-session-user | 0 | - | - | - | - |
| propmem | single-session-assistant | 0 | - | - | - | - |
| propmem | single-session-preference | 0 | - | - | - | - |
| propmem | multi-session | 0 | - | - | - | - |
| propmem | temporal-reasoning | 0 | - | - | - | - |
| propmem | knowledge-update | 1 | 0.3333 | 0.2000 | 1.0000 | 0.0000 |

## Memory Diagnostics

| System | Context reader tokens | Context embedding tokens | Recall@5 | Recall@10 | MRR | Reader prompt tokens | Retrieval ms | Ingestion ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fullcontext | 108064.0000 | 108065.0000 | 1.0000 | 1.0000 | 1.0000 | 108137.0000 | 0.0000 | 0.0000 |
| openclaw | 7267.0000 | 7268.0000 | 0.0000 | 0.0000 | 0.0000 | 7340.0000 | 84.8961 | 7570.0974 |
| mem0 | 452.0000 | 453.0000 | - | - | - | 526.0000 | 59.3810 | 548119.2964 |
| simplemem | 1462.0000 | 1463.0000 | - | - | - | 1535.0000 | 11681.7518 | 515957.9518 |
| propmem | 3314.0000 | 3315.0000 | 1.0000 | 1.0000 | 1.0000 | 3387.0000 | 130.4297 | 726760.6007 |

## Retrieval Diagnostics by Category

| System | Category | Context reader tokens | Recall@5 | Recall@10 | MRR | Retrieval ms | Ingestion ms |
|---|---|---:|---:|---:|---:|---:|---:|
| fullcontext | single-session-user | - | - | - | - | - | - |
| fullcontext | single-session-assistant | - | - | - | - | - | - |
| fullcontext | single-session-preference | - | - | - | - | - | - |
| fullcontext | multi-session | - | - | - | - | - | - |
| fullcontext | temporal-reasoning | - | - | - | - | - | - |
| fullcontext | knowledge-update | 108064.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| openclaw | single-session-user | - | - | - | - | - | - |
| openclaw | single-session-assistant | - | - | - | - | - | - |
| openclaw | single-session-preference | - | - | - | - | - | - |
| openclaw | multi-session | - | - | - | - | - | - |
| openclaw | temporal-reasoning | - | - | - | - | - | - |
| openclaw | knowledge-update | 7267.0000 | 0.0000 | 0.0000 | 0.0000 | 84.8961 | 7570.0974 |
| mem0 | single-session-user | - | - | - | - | - | - |
| mem0 | single-session-assistant | - | - | - | - | - | - |
| mem0 | single-session-preference | - | - | - | - | - | - |
| mem0 | multi-session | - | - | - | - | - | - |
| mem0 | temporal-reasoning | - | - | - | - | - | - |
| mem0 | knowledge-update | 452.0000 | - | - | - | 59.3810 | 548119.2964 |
| simplemem | single-session-user | - | - | - | - | - | - |
| simplemem | single-session-assistant | - | - | - | - | - | - |
| simplemem | single-session-preference | - | - | - | - | - | - |
| simplemem | multi-session | - | - | - | - | - | - |
| simplemem | temporal-reasoning | - | - | - | - | - | - |
| simplemem | knowledge-update | 1462.0000 | - | - | - | 11681.7518 | 515957.9518 |
| propmem | single-session-user | - | - | - | - | - | - |
| propmem | single-session-assistant | - | - | - | - | - | - |
| propmem | single-session-preference | - | - | - | - | - | - |
| propmem | multi-session | - | - | - | - | - | - |
| propmem | temporal-reasoning | - | - | - | - | - | - |
| propmem | knowledge-update | 3314.0000 | 1.0000 | 1.0000 | 1.0000 | 130.4297 | 726760.6007 |

Session-retrieval metrics are null when a baseline does not expose auditable source-session provenance; this is not scored as a retrieval miss.
Context reader tokens use the loaded frozen Qwen3-8B llama.cpp tokenizer; context embedding tokens use the frozen Qwen3-Embedding-0.6B tokenizer. Context-size comparisons use context reader tokens.

Reader / answer model, memory-internal LLM, memory system, embedding model and judge are separate manifest roles.
Published upstream results are historical coordinates, not controlled-stack comparisons.
