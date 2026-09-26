# R2MED CRB compact-schema follow-up

## Decision

The valid/fallback DEV diagnostic showed a positive CRB-Q signal, so one compact-schema variant was run across all 393 DEV queries. It improved on the previous CRB-Q result, but did not beat the strongest cost-matched GAR baseline. The R2MED improvement gate therefore remains **negative**. No TEST run, method lock, second prompt revision, or follow-on retrieval change was made.

## Diagnostic that authorized the single repair

The original CRB-Q generation had 156 valid structured outputs out of 393 (39.7%). On those same valid queries, full CRB-Q exceeded a four-channel original-query fallback reconstructed with its DEV-selected `k20-W0` fusion by a paired macro nDCG@10 delta of **+0.0602**, positive on all three DEV subsets. The per-subset valid counts were 60/150, 16/114, and 80/129.

The fallback replay audit matched the cached CRB-Q top-100 ranking on 235/237 fallback queries. The two differences were confined to the fallback stratum and are listed in the diagnostic JSON; they are not part of the valid-query paired effect. This was a screening diagnostic, not a final benchmark result.

## One compact-schema run

- Dataset/split: R2MED DEV only — PMC-Treatment (150), PMC-Clinical (114), IIYi-Clinical (129).
- Generator: the same local Qwen3-8B-Q4_K_M GGUF, SHA-256 `d98cdcbd…5745785`.
- Generation: temperature 0, reasoning disabled, 256-token maximum, exactly one call per query, no retries; 393 local calls and no paid API calls.
- Compact JSON: `q` (short search query), `t` (up to four terms), `e` (one short evidence sentence). These fields were normalized to the existing CRB retrieval inputs.
- Retrieval: same Lucene BM25 and BGE-large corpora, four channels, top-100, and the original CRB-Q DEV-selected `k20-W0` RRF configuration. No fusion retuning.
- Output validity: 382/393 (97.2%); 11 fell back on JSON/constraint validation; zero outputs were truncated. The predeclared 99% validity target was not met. Valid counts: PMC-Treatment 147/150, PMC-Clinical 106/114, IIYi-Clinical 129/129.

## DEV result

| Subset | Compact CRB-Q | Original CRB-Q | LameR-MV | Compact CRB-Q − LameR |
| --- | ---: | ---: | ---: | ---: |
| PMC-Treatment | 0.3870 | 0.3254 | 0.4571 | −0.0701 |
| PMC-Clinical | 0.2650 | 0.2030 | 0.2627 | +0.0023 |
| IIYi-Clinical | 0.2243 | 0.2161 | 0.1796 | +0.0447 |
| Equal-subset macro | **0.2921** | 0.2482 | **0.2998** | **−0.0077** |

Compact CRB-Q gained **+0.0439 nDCG@10** over the old CRB-Q overall and exceeded LameR-MV on two subsets, but its large PMC-Treatment deficit outweighed those gains. The predeclared DEV gate requires at least +0.005 macro nDCG@10 over the strongest cost-matched GAR plus positive deltas on at least two subsets; it was not met. Secondary macro MRR@10 was 0.3493. Recall@100 was 0.7105 versus LameR-MV's 0.7076, but this does not override the primary nDCG gate.

## Stop status

`CRB_COMPACT_GENERATION_VALIDITY = IMPROVED`

`CRB_DEV_DELTA_VS_OLD = +0.0439`

`CRB_DEV_DELTA_VS_STRONGEST_GAR = -0.0077`

`PUBLIC_BENCHMARK_IMPROVEMENT = NO`

`RESUME_HEADLINE_READY = NO`

`TEST_ACCESSED = NO`

The one authorized compact-schema variant is complete. Under the sprint stop rule, do not tune another prompt/schema, add a reranker, or run TEST without a new direction. Aggregate DEV results and generation/retrieval artifact hashes are in `runs/rag_r2med_crb/compact_repair/dev_report.json` and `runs/rag_r2med_crb/compact_repair/generation/generation_manifest.json`.
