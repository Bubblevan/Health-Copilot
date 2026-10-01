# MEM-3B1 Slot Proposal Diagnostic

Exploratory local-Qwen diagnostic over six source-grounded facts. The frozen response is scored after generation against two revision-positive pairs and one coexistence-negative pair. This is not a benchmark result.

- Proposal coverage/schema: 6/6 records; schema pass: True
- Pairwise accuracy: 0.667 (2/3)
- Same-slot positive recall: 0.500 (1/2)
- Coexistence control specificity: 1.000 (1/1)
- False revision merges: 0/1 control
- Valid source quote references: 6/6
- Output envelope: model returned JSON array; frozen request used json_object.
- One local model completion, no hosted calls.

## Pair Audit

| Control | Pair | Prediction | Result | Slot equality details |
|---|---|---|---|---|
| revision_positive | r03, r05 | SAME_SINGLE_VALUE_SLOT | PASS | entity=True, attribute=True, single=True |
| revision_positive | r01, r04 | COEXIST_NOT_REVISION | FAIL | entity=True, attribute=False, single=True |
| coexistence_negative | r02, r06 | COEXIST_NOT_REVISION | PASS | entity=True, attribute=False, single=True |

## Proposals

| Record | Entity | Attribute | Value | Cardinality |
|---|---|---|---|---|
| r01 | self | gym_days | tuesday_thursday_saturday | SINGLE_VALUE_AT_A_TIME |
| r02 | self | jewelry_store_search | mall | SINGLE_VALUE_AT_A_TIME |
| r03 | self | instagram_followers | 600 | SINGLE_VALUE_AT_A_TIME |
| r04 | self | gym_frequency | four_times_a_week | SINGLE_VALUE_AT_A_TIME |
| r05 | self | instagram_followers | 500 | SINGLE_VALUE_AT_A_TIME |
| r06 | self | running_shoes_search | racing | SINGLE_VALUE_AT_A_TIME |

## Interpretation

The model correctly canonicalized the two Instagram follower facts and separated the running-shoe and jewelry-store interests. It missed the gym transition by naming the older schedule gym_days and the later frequency gym_frequency. Thus this diagnostic has zero false merges on its one coexistence control, but only one of two revision positives is admitted.

The semantic failure motivates a factorized typed slot representation: state dimension plus object/qualifier plus value type/unit, rather than a single unconstrained attribute string. This is a next hypothesis to test, not a result established by this six-record diagnostic.

The response-envelope mismatch was transport-level only: the model returned a valid JSON array despite the requested object wrapper. The frozen answer content is accepted by this offline scorer without altering it.
