# Typed Slot Canonicalization Diagnostic

This post-hoc audit reuses the exact same six records and frozen local Qwen proposals. It is a failure-repair diagnostic, not independent evaluation or benchmark evidence.

- Raw factorized string-key result: 2/3 pairs (1/2 revision positives, 1/1 coexistence control).
- After deterministic typed normalization: 3/3 pairs.
- Revision-positive recall: 1.000 (2/2).
- Coexistence specificity: 1.000 (1/1); false revision merges: 0.
- Historical AS_OF state: 2/2; CURRENT latest state: 2/2.

## Materialized Controls

| Transition | Older value | Current value | Current status | Historical retained |
|---|---|---|---|---|
| r05 -> r03 | 500 | 600 | ACTIVE | True |
| r01 -> r04 | 3 | 4 | ACTIVE | True |

## Interpretation

The explicit alias/unit normalizer resolves the gym naming drift in these records while keeping distinct shopping targets apart and including scope in identity. Since the rules were added after observing these same six records, the 3/3 result is a development diagnostic only. The next meaningful gate is to freeze the normalizer and test it on untouched DEV state pairs before any benchmark run.
