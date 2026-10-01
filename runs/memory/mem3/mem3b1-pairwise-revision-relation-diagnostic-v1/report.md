# Pairwise Revision Relation Diagnostic

One local Qwen pass over every pair in the second non-overlapping 15-group sample. The model proposes semantic labels only; deterministic harness code computes the admission rule. Group IDs, timestamps, scopes, and review labels were omitted from the model request.

- Pairs: 17; groups: 15/15
- Group accuracy against reviewed strata: 0.667
- Auto-update precision: None
- Auto-update recall: 0.0
- False auto-updates: 0; missed true-singleton groups: 5
- Proposed auto-update pair count: 0

## Group Results

| Review stratum | Proposed update | Correct vs prior slot review | Pair relation outputs |
|---|---:|---:|---|
| true_singleton_slot | False | False | YES/SINGLE_VALUE_AT_A_TIME/IDENTICAL |
| true_singleton_slot | False | False | YES/SINGLE_VALUE_AT_A_TIME/COMPLEMENTARY, YES/SINGLE_VALUE_AT_A_TIME/COMPLEMENTARY, YES/SINGLE_VALUE_AT_A_TIME/COMPLEMENTARY |
| true_singleton_slot | False | False | YES/SINGLE_VALUE_AT_A_TIME/IDENTICAL |
| true_singleton_slot | False | False | YES/SINGLE_VALUE_AT_A_TIME/IDENTICAL |
| true_singleton_slot | False | False | NO/COMPOSITE_TASK_STATE/UNKNOWN |
| false_revision_merge | False | True | YES/SINGLE_VALUE_AT_A_TIME/COMPLEMENTARY |
| false_revision_merge | False | True | YES/COMPOSITE_TASK_STATE/COMPLEMENTARY |
| false_revision_merge | False | True | YES/COMPOSITE_TASK_STATE/UNKNOWN |
| false_revision_merge | False | True | YES/SET_OR_MULTI_VALUE/COEXISTING |
| false_revision_merge | False | True | YES/SET_OR_MULTI_VALUE/COEXISTING |
| uncertain | False | True | NO/SINGLE_VALUE_AT_A_TIME/MUTUALLY_EXCLUSIVE |
| uncertain | False | True | YES/SINGLE_VALUE_AT_A_TIME/COMPLEMENTARY |
| uncertain | False | True | NO/COMPOSITE_TASK_STATE/COMPLEMENTARY |
| uncertain | False | True | NO/COMPOSITE_TASK_STATE/COMPLEMENTARY |
| uncertain | False | True | NO/SINGLE_VALUE_AT_A_TIME/MUTUALLY_EXCLUSIVE |

## Interpretation

This diagnostic separates topic/slot similarity from overwrite safety. A model-proposed relation never mutates storage; only the frozen three-part rule may admit an update, after which deterministic timestamps establish ordering.
