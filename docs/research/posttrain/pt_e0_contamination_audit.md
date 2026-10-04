# PT-E0 Training Source Contamination Audit

This report contains counts and stable IDs only. It does not print HealthBench Professional or LiveMedBench prompts or rubric text.

## Frozen raw source counts and exact dedup

| Source | Raw rows | Unique question families | Within-source duplicate rows | Within-source duplicate rate | Clean unique families |
|---|---:|---:|---:|---:|---:|
| o1_en | 19704 | 19677 | 27 | 0.1370% | 19664 |
| o1_zh | 20171 | 17172 | 2999 | 14.8679% | 17163 |
| medreason | 32682 | 32624 | 58 | 0.1775% | 31938 |
| huatuo | 177703 | 177703 | 0 | 0.0000% | 177703 |
| rl | 40644 | 40310 | 334 | 0.8218% | 40297 |

## Exact cross-source overlap

| Source A | Source B | Raw matching row pairs | Unique prompt families |
|---|---|---:|---:|
| o1_en | medreason | 5796 | 5771 |
| o1_en | rl | 13704 | 13540 |
| medreason | rl | 4501 | 4446 |

Cross-SFT exact duplicate prompt families: **5771 / 241405 union-unique SFT families (2.3906%)**; raw matching cross-SFT row pairs: **5796**.
The rate denominator is the union of exact-unique raw prompt families across the four SFT sources; a prompt duplicated across more than two sources is counted once in the numerator.

## External evaluation contamination removed

| Training source | Benchmark | Exact rows | Near-duplicate rows | 64-character overlap rows |
|---|---|---:|---:|---:|
| o1_en | diagnosisarena | 0 | 0 | 13 |
| o1_en | cmb | 0 | 0 | 0 |
| o1_en | hbpro | 0 | 0 | 0 |
| o1_en | livemedbench | 0 | 0 | 0 |
| o1_zh | diagnosisarena | 0 | 0 | 0 |
| o1_zh | cmb | 0 | 4 | 6 |
| o1_zh | hbpro | 0 | 0 | 0 |
| o1_zh | livemedbench | 0 | 0 | 0 |
| medreason | diagnosisarena | 0 | 0 | 17 |
| medreason | cmb | 0 | 1 | 0 |
| medreason | hbpro | 0 | 0 | 2 |
| medreason | livemedbench | 0 | 0 | 4 |
| huatuo | diagnosisarena | 0 | 0 | 0 |
| huatuo | cmb | 0 | 0 | 0 |
| huatuo | hbpro | 0 | 0 | 0 |
| huatuo | livemedbench | 0 | 0 | 0 |
| rl | diagnosisarena | 0 | 0 | 12 |
| rl | cmb | 0 | 0 | 0 |
| rl | hbpro | 0 | 0 | 0 |
| rl | livemedbench | 0 | 0 | 1 |

## MedReason provenance

| dataset_name | Rows |
|---|---:|
| LastHumanity | 57 |
| MMLU | 827 |
| MedXpertQA | 666 |
| huatuo | 6475 |
| medmcqa | 6197 |
| medqa | 8016 |
| pubmedqa | 603 |
| pubmedqa_artificial | 8094 |
| pubmedqa_unlabeled | 1747 |
Rows excluded unconditionally because dataset_name == MedXpertQA: **666**.
Recorded provenance counts for MedQA / MedMCQA / PubMedQA / MMLU: **8016 / 6197 / 10444 / 827**.

## SFT/RL prompt-family separation

- Raw shared families across all SFT sources and RL: **13620**.
- Clean shared families assigned once across stages: **13612**.
- Raw shared medical-o1 families: **13540**; clean: **13532**.
- Raw unmatched medical-o1 SFT-only families: **23309**; all-SFT unmatched families: **227785**.
- Clean medical-o1 shared families assigned to SFT: **6777**; RL train: **6364**; RL dev: **391**.
- Frozen buckets: SFT 0–49, RL train 50–96, RL dev 97–99 from first 16 SHA256 hex digits modulo 100. This is applied to every exact SFT/RL shared prompt family, including MedReason and Huatuo rows.
- Candidate SFT/RL prompt-family overlap after splitting: **0**.

## Final clean candidates

- SFT candidate families before internal DEV: **233909**.
- SFT internal DEV: **2000**; SFT train candidates: **231909**.
- RL train candidates: **32296**; RL internal DEV: **1179**.

Exact source row IDs and split lists are stored in the ignored PT-E0 run directory. No SFT or RL weights were updated in PT-E0.
