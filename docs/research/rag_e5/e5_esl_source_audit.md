# ESL-Bench Source Audit for RAG-E5

## Scope

ESL-Bench is used as a longitudinal-state substrate for an
`ESL-Bench-derived Integration Overlay`, not as an official ESL-Bench leaderboard
evaluation. The audit script reads the source `manifest.json` and per-user
`profile.json`, `timeline.json`, and `exam_data.json` only. It does not open
native question JSONL, answer-bearing JSONL, or `kg_evaluation_queries.json`.
For 202608, it performs schema/file-identity checks only and does not inspect
task outcomes.

The source snapshot is present at
`D:/MyLab/Jianli/external/datasets/ESL-Bench`. It has no `.git` metadata, so
there is no local Git commit to cite. Identity is pinned by the published batch
checksums, the local manifest SHA-256, exact cohort IDs, and the aggregate
SHA-256 over sorted per-user raw-state file paths and bytes. Reproduce the audit
with:

```powershell
.venv\Scripts\python.exe tools\research\rag_e5\audit_esl_source.py `
  --source-root D:\MyLab\Jianli\external\datasets\ESL-Bench
```

The generated machine-readable identity is
[`runs/rag_e5/esl_source_manifest.json`](../../../runs/rag_e5/esl_source_manifest.json).

## Frozen cohort boundary

| Role | Batch | Users | Published dataset | Published batch checksum |
| --- | --- | ---: | --- | --- |
| Development only | 202607 | 20 (`user5200_AT_demo`–`user5219_AT_demo`) | `sample280-20260730` | `sha256:6f4ce4faf6dee27a723feceeb9f9cb4a91beb7995ce6304c322b338e95582bd5` |
| Future integration holdout | 202608 | 20 (`user5300_AT_demo`–`user5319_AT_demo`) | `sample320-20260830` | `sha256:7745391e881d5ee1d4e006f2220aecb8cea1512edad5782d535c23bb2017a548` |

The cohorts are disjoint. Development/validation must be split by `user_id`,
never by query. Although the 202607 batch has a released native answer file, it
is not an E5 teacher source. E5 builds a separately named integration overlay
from raw state and reviewed public evidence. The 202608 native questions and
hidden evaluation semantics are not training labels and remain untouched.

## Structural compatibility findings

Every batch has 20 user directories and all 20 users have each of the three
allowed raw-state files. Profile top-level fields are stable:
`metadata`, `demographics`, `personality`, and `health_profile`. Demographics,
personality type, and health-profile field names are structurally compatible.

Optional schema differences must be handled explicitly by a future loader:

- `profile.metadata` has six fields in 202607 but only `language` and
  `profile_id` in 202608.
- Both timeline roots share `user_id`, `generated_at`, `entry_count`, and
  `entries`. The event-entry schema includes `end_time` in 202607 but not in
  202608; measurement and exam-entry records retain their shared core fields.
- Both `exam_data.json` files are arrays of six records per user. The common
  fields include date, type, location, indicators, recommendations, and weekday.
  `indicators` is an object/map in both pinned batches.
  202607 also has `abnormal_findings`, `overall_assessment`, and
  `generation_summary`; 202608 omits those optional keys.

The fingerprint script enforces only common required structure and reports
schema inventories without emitting source values. The exact per-batch file
index fingerprints are stored in the machine-readable manifest.

## Identity

- ESL-Bench manifest version: `16`.
- Manifest updated: `2026-09-07T08:34:58Z`.
- Local `manifest.json` SHA-256:
  `d181b25117955cb3bae240a3e7a440d019b7e76fc3f67252a540549c4097cce9`.
- 202607 raw-state file index SHA-256:
  `faa1f93cd9216fa218d33e41c6e8b2bddbe26ec1303027d3cc7d453ce8669137`.
- 202608 raw-state file index SHA-256:
  `728ac1a16365644aee7a7a1f07e8acb7179ac3fbe470dbf2ad36a162ceda7ccf`.

## Use boundary

The raw user state may later provide runtime history/memory context. Native ESL
questions and answer artifacts are separate from the overlay and cannot be
silently folded into its task generator, policy features, or teacher labels. The
future holdout identity can be validated now; outcomes cannot be inspected
until a final lock and a one-shot holdout run are authorized by the E5 gates.
