# Factorized Slot Proposal Diagnostic

Failure-driven DEV iteration over the same six source records. This is a mechanism diagnostic, not a benchmark result or independent holdout.

- Pairwise accuracy: 0.667 (2/3)
- Revision-positive recall: 0.500 (1/2)
- Coexistence specificity: 1.000 (1/1)
- False revision merges: 0/1
- Response root: array

## Proposals

| Record | Dimension | Object | Value | Unit | Cardinality |
|---|---|---|---|---|---|
| r01 | schedule | gym visits | 3 | sessions_per_week | SINGLE_VALUE_AT_A_TIME |
| r02 | search intent | jewelry store | 1 | searches | SINGLE_VALUE_AT_A_TIME |
| r03 | follower count | Instagram account | 600 | followers | SINGLE_VALUE_AT_A_TIME |
| r04 | frequency | gym routine | 4 | times_per_week | SINGLE_VALUE_AT_A_TIME |
| r05 | follower count | Instagram account | 500 | followers | SINGLE_VALUE_AT_A_TIME |
| r06 | search intent | running shoes | 1 | searches | SINGLE_VALUE_AT_A_TIME |

## Pair Audit

| Control | Pair | Prediction | Pass |
|---|---|---|---|
| revision_positive | r03, r05 | SAME_SINGLE_VALUE_SLOT | True |
| revision_positive | r01, r04 | COEXIST_NOT_REVISION | False |
| coexistence_negative | r02, r06 | COEXIST_NOT_REVISION | True |

A passing result on these reused records would qualify the representation for a broader frozen-DEV diagnostic only. It would not establish general revision admission quality.
