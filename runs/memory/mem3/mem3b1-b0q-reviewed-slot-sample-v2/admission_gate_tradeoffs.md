# Slot Admission Gate Trade-Offs

Offline analysis of the frozen second 15-group diagnostic. No model call was made and no raw proposal was changed. A reviewed true-singleton group is the positive class; false-merge and uncertain groups are conservatively negative.

| Gate | Correct | N | Accuracy | Precision | Recall | False admissions | Missed true |
|---|---:|---:|---:|---:|---:|---:|---:|
| typed_slot_key_only | 9 | 15 | 0.600 | 0.429 | 0.600 | 4 | 2 |
| typed_slot_plus_qualifier_equality | 12 | 15 | 0.800 | 1.000 | 0.400 | 0 | 3 |
| typed_slot_plus_distinct_values | 10 | 15 | 0.667 | 0.500 | 0.200 | 1 | 4 |
| typed_slot_plus_both | 11 | 15 | 0.733 | 1.000 | 0.200 | 0 | 4 |

## Reading

The broad slot key alone has high recall but merges some complementary or uncertain task dimensions. Requiring exact equality of value qualifiers tests one conservative alternative; requiring all values to differ tests another. These are post-hoc development diagnostics on the same frozen 15 groups and are not independent performance estimates.

The next method decision must separate slot identity from revision compatibility: a shared topical slot may accumulate evidence without superseding an earlier fact. Materialization should require a separately established, mutually exclusive value transition.
