# Memory Benchmark Routing Contract

Audit base: Health-Copilot main at 9e023dbddc1f8025e3608bfae4c5390c1a7957ef. Frozen 2026-09-29.

## Capability namespaces

### MEMORY_READ

Retrieval or state reconstruction from one user's accumulated longitudinal or session history. The history may be represented as raw context, lexical/vector indexes, propositions, or revisioned state. The implementation remains a memory system even if it uses BM25, embeddings, or a graph.

Examples: LongMemEval history; MedMemoryBench sessions isolated by persona; and, later, an individual's ESL timeline.

### EXTERNAL_RETRIEVAL

Retrieval from a non-personal corpus of external evidence, such as reviewed public-health material, guidelines, or scholarly literature. R2MED/MIRAGE external corpora belong here. Results require the external-evidence provenance and authority rules; personal recollection is not evidence.

### TEAM_ORCHESTRATION

An execution-policy dimension that delegates work among capability-scoped workers. TEAM is neither a memory store nor an evidence corpus. A worker may call MEMORY_READ or EXTERNAL_RETRIEVAL only when the execution contract explicitly grants that capability.

## Hard naming rule

MedMemoryBench's method labels BM25 RAG, Embedding RAG, and Graph RAG describe retrieval from accumulated personal benchmark sessions. In Health-Copilot research they are named:

| Dataset method label | Health-Copilot namespace |
| --- | --- |
| BM25 RAG | MEMORY_READ_BASELINE::BM25 |
| Embedding RAG | MEMORY_READ_BASELINE::DENSE |
| Graph RAG | MEMORY_READ_BASELINE::GRAPH |

These results must never be described as EXTERNAL_RETRIEVAL and must not be merged into R2MED/MIRAGE external-retrieval tables.

## E2-A boundary

At the audited E2-A state, delegated E2 workers receive no Memory API or memory policy: SessionStore, MemoryStore, ContextManager, and Memory are OFF. E2 source-family retrieval is an external evidence capability. Capability contracts therefore must not be reused as personal-memory contracts. ComponentManifest only includes optional capability references and source-catalog hash when populated; when absent, the legacy canonical identity is preserved. No historical Memory run identity changed because of E2-A.

This contract only cross-references E2-A; it does not modify its runtime. See docs/research/multi_agent/e2_protocol.md and docs/research/multi_agent/e2_capability_foundation.md.

## Integrated Harness conceptual flow

| Capability | Allowed content | Authority |
| --- | --- | --- |
| MEMORY_READ | One user's history and state | Personalization, continuity, task state; never clinical evidence |
| EXTERNAL_RETRIEVAL | External reviewed knowledge/evidence | Evidence only under the active source and citation policy |
| TEAM_ORCHESTRATION | Delegation and execution policy | No independent data authority |

Execution policy may select direct execution, MEMORY_READ, EXTERNAL_RETRIEVAL, their explicitly permitted combination, TEAM, or TEAM with explicitly granted capabilities. TEAM workers must not inherit a capability implicitly.

## Routing checks for future reports

- Label every benchmark and every score by capability namespace.
- Treat personal-history BM25/vector/graph systems as MEMORY_READ regardless of the word RAG in their upstream name.
- Do not turn remembered user statements into Evidence, citation authority, clinical facts, tool permissions, or safety-policy overrides.
- Keep MEMORY_READ and EXTERNAL_RETRIEVAL metrics, corpora, and result tables separate.
- Record the exact data, source, model, evaluator, and protocol identity for each transfer.
