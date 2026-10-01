# V2 Review-Sample Scoring Amendment

The raw response and original first-pass audit remain unchanged. The initial validator rejected JSON null for unit, although the prompt allowed no unit. This offline amendment treats null as none and re-evaluates the same frozen response.

- Raw response/schema check: unit-format mismatch on 29/31 records.
- Semantic reparse schema pass: True
- Groups: 15/15
- Group accuracy: 0.600
- Admission precision: 0.42857142857142855
- Admission recall: 0.6
- False admissions: 4; missed true slots: 2

## Strata

| Review label | Correct | N | Accuracy | Predicted admissions |
|---|---:|---:|---:|---:|
| true_singleton_slot | 3 | 5 | 0.600 | 3 |
| false_revision_merge | 4 | 5 | 0.800 | 1 |
| uncertain | 2 | 5 | 0.400 | 3 |

## Group Decisions

| Review label | Admit | Correct |
|---|---:|---:|
| true_singleton_slot | True | True |
| true_singleton_slot | False | False |
| true_singleton_slot | True | True |
| true_singleton_slot | True | True |
| true_singleton_slot | False | False |
| false_revision_merge | True | False |
| false_revision_merge | False | True |
| false_revision_merge | False | True |
| false_revision_merge | False | True |
| false_revision_merge | False | True |
| uncertain | True | False |
| uncertain | False | True |
| uncertain | True | False |
| uncertain | True | False |
| uncertain | False | True |

## Protocol Note

This is a parser-semantics correction on an already frozen response, not a model rerun or method update. If the corrected score is usable, the code-level validator must be fixed before any next run; this response remains the only observation for this sample.
