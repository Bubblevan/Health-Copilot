# U2-F Distribution and Integrity Hardening

Date: 2026-09-30

Dataset: `health-copilot-owned-longitudinal-v1`

## Hard scale and split checks

| Split | Target | Generated | Subjects | Result |
|---|---:|---:|---:|---|
| TRAIN | 4,096 episodes / 320 subjects | 4,096 | 320 | PASS |
| DEV_IID | 512 episodes / 64 subjects | 512 | 64 | PASS |
| DEV_STRUCTURAL | 512 episodes / 64 subjects | 512 | 64 | PASS |
| Total | under 8,000 episodes | 5,120 | 448 | PASS |

All subjects have one shared 80-row timeline spanning 729 days. Later snapshots are monotonic and add records for all 448 subjects. TRAIN history-regime counts are SHORT 896 (21.9%), MEDIUM 1,280 (31.25%), LONG 1,280 (31.25%), and SATURATED 640 (15.6%); every minimum is met. The 1–4 visible-record share is below the 70% ceiling. Split audit reports zero subject, persona, seed, or sibling leakage.

## Capability and task coverage

TRAIN's dependency-derived capability distribution is:

| Requirement | Episodes | Share | Reference range | Result |
|---|---:|---:|---:|---|
| NONE | 916 | 22.4% | 10–25% | PASS |
| MEMORY | 1,045 | 25.5% | 20–40% | PASS |
| RAG | 807 | 19.7% | 15–30% | PASS |
| MEMORY+RAG | 826 | 20.2% | 15–30% | PASS |
| INSUFFICIENT | 502 | 12.3% | 5–15% | PASS |

The five required patient-record types are available in every split and appear as required Memory evidence. Across TRAIN, required evidence rows by type are CONVERSATION 675, EVENT 631, EXAM 620, MEASUREMENT 620, and PROFILE 676; the largest type is about 21% of this evidence pool, far below the 80% concentration ceiling.

| TRAIN scenario family | Count | TRAIN scenario family | Count |
|---|---:|---|---:|
| COMPOSITIONAL_MULTI_FACT | 256 | MEMORY_EXTERNAL_JOIN | 352 |
| CURRENT_ONLY | 684 | MEMORY_LOOKUP | 487 |
| DISTRACTOR_HEAVY | 232 | MEMORY_MULTI_RECORD | 124 |
| EXTERNAL_LOOKUP | 375 | MEMORY_REVISION | 206 |
| EXTERNAL_MULTI_SOURCE | 216 | MEMORY_TEMPORAL_COMPARE | 130 |
| EXTERNAL_VERSIONED | 216 | TEMPORAL_BOUNDARY | 136 |
| INSUFFICIENT_EVIDENCE | 464 | MEMORY_EXTERNAL_CONFLICT | 218 |

All six answer types occur in TRAIN: ABSTAIN 464, BOOLEAN 216, EXACT_SET 1,462, EXACT_TOKEN 1,772, NUMERIC 48, and ORDERED_SEQUENCE 134. All five insufficient-evidence subtypes are covered. Capability requirement is derived from required graph facts and their availability, not from resource existence. In particular, wrong-family and unresolved-conflict values remain distractors while the requested adjudicated value is marked unavailable.

## Longitudinal, temporal, and evidence complexity

TRAIN revision-depth counts are depth 1: 66, depth 2: 66, and depth 3: 74. All six required memory-distance buckets occur: <1 day 248, 1–7 days 112, 8–30 days 384, 31–180 days 1,101, 181–365 days 380, and >365 days 386.

Dependency depths are 1: 2,689, 2: 967, 3: 370, 4: 68, and 5: 2. Depths 1–3 are all represented. TRAIN evidence worlds span all three regimes: SMALL 1,380, MEDIUM 1,356, and LARGE 1,360. DEV_IID has 170/182/160; DEV_STRUCTURAL has 178/184/150 in SMALL/MEDIUM/LARGE order.

| History regime | Distractor distribution |
|---|---|
| SHORT | LOW 742; MEDIUM 410 |
| MEDIUM | MEDIUM 1,536 |
| LONG | HIGH 896; EXTREME 640 |
| SATURATED | EXTREME 896 |

The temporal/revision audit checked 342 future personal records and hid all 342; it made 71,456 publication/effectivity checks, 258 revision-chain checks, 166 decision-boundary checks, and 272 external-version selection checks. All checks passed in the frozen run `55955b2eff38`.

## Matched-pair coverage and duplicate controls

The generated universe contains 388 same-surface/different-requirement matched pairs (15.2% of episodes) and 1,616 different-surface/same-graph pairs (63.1% of episodes). Both exceed their 10% and 20% minima. Exact duplicates are permitted only for designated matched surface groups. The frozen run has 388 duplicate rows wholly inside those groups, zero unapproved exact query duplicate groups, zero exact runtime-serialization duplicates, and zero normalized query overlap across splits. Latent graph duplicates remain confined to sibling groups.

The final frozen run's manifest (`dataset_root_hash=e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134`) and `distribution_report.json`, `matched_pair_diagnostics.json`, `temporal_revision_audit.json`, and `duplicate_audit.json` are the authoritative per-run records. All scale, temporal, duplicate, lineage, source-independence, and counterfactual gates pass. This document is a compact distribution summary; the machine-readable reports preserve the full strata and checks.
