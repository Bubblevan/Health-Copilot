# B0Q Reviewed Slot Sample: Typed Proposal Diagnostic

Stable stratified sample of prior human-reviewed B0Q groups. Labels and group membership were kept outside the model request. This remains a development diagnostic because the broader B0Q review informed method iteration.

- Groups: 15/15; records: 30
- Group accuracy: 0.933
- Admission precision: 1.0
- Admission recall: 0.8
- False admissions: 0; missed true slots: 1

## Stratified Results

| Review stratum | Correct | N | Accuracy | Predicted admissions |
|---|---:|---:|---:|---:|
| true_singleton_slot | 4 | 5 | 0.800 | 4 |
| false_revision_merge | 5 | 5 | 1.000 | 0 |
| uncertain | 5 | 5 | 1.000 | 0 |

## Group Decisions

| Label | Predicted Admit | Correct |
|---|---:|---:|
| true_singleton_slot | True | True |
| true_singleton_slot | True | True |
| true_singleton_slot | True | True |
| true_singleton_slot | True | True |
| true_singleton_slot | False | False |
| false_revision_merge | False | True |
| false_revision_merge | False | True |
| false_revision_merge | False | True |
| false_revision_merge | False | True |
| false_revision_merge | False | True |
| uncertain | False | True |
| uncertain | False | True |
| uncertain | False | True |
| uncertain | False | True |
| uncertain | False | True |

## Interpretation

The model saw only shuffled proposition text with neutral record IDs. The deterministic harness then applied scope and typed-slot normalization outside the model. This is not a final benchmark estimate and does not change the frozen B0Q review.
