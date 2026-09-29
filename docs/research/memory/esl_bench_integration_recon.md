# ESL-Bench Integration Reconnaissance

Audit date: 2026-09-29. Canonical identity requested: mirobody/ESL-Bench. No dataset was downloaded, locally ingested, or evaluated.

## Publicly visible structure

The public README describes profile, timeline, exam_data, events, and evaluation-query artifacts (including knowledge-graph evaluation queries), plus a manifest/versioning structure. These support an environment with longitudinal state and exam/reasoning tasks; they do not alone establish a controlled memory-mechanism benchmark.

The README currently contains a fetch reference to healthmemoryarena/ESL-Bench while the requested canonical identity is mirobody/ESL-Bench. Pin exact owner/revision, file hashes, license, and schema before any future transfer. This alias/ownership discrepancy is unresolved for execution.

## Research classification

Classification: INTEGRATION_ENVIRONMENT_CANDIDATE, not MEMORY_MECHANISM_BENCHMARK.

Potential future separation:

- Profile/timeline/person-specific state: MEMORY_READ only when it represents one user's history.
- External dependency: EXTERNAL_RETRIEVAL only when it requires a separately sourced external corpus.
- TEAM: orchestration policy only, with explicit grants to either capability.

## Scope boundary

This reconnaissance does not authorize download, adapter implementation, local ingestion, scoring, or benchmark claims. In a later integration stage, run separate MEMORY-only, EXTERNAL_RETRIEVAL-only, combined, and TEAM conditions, and pin each dataset revision and evaluator. Do not merge ESL scores into LongMemEval memory-mechanism claims or R2MED/MIRAGE evidence-retrieval tables.

Source inspected: https://huggingface.co/datasets/mirobody/ESL-Bench/blob/main/README.md
