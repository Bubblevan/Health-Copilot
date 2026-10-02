# RAG-E6A report

## Executive result

The CFEC evidence-utilization hypothesis did not survive BUILD evaluation.
CFEC-v1.4 is not resume evidence and was not promoted to FROZEN_DEV. On the
predeclared 156-query RAG slice, its strong arm underperformed the matched
Vanilla-STRONG reader by 75.0 percentage points in grounded task success. This
report preserves that result as a method-selection negative, not as a claim that
retrieval itself is ineffective.

## Frozen comparison

BUILD used the frozen U2-F subject split and executed every one of the 818 BUILD
episodes across all five arms (4,090 arm executions) before evaluator truth was
opened. The RAG slice contains 156 episodes from 49 subjects. Vanilla and CFEC
used the same pinned LameR-MV retrieval, top-10 evidence, Qwen3-8B-Q4_K_M
generator, temperature 0, reasoning-disabled configuration, and 512-token
completion ceiling. The only method difference was the reader execution graph
and prompts. Scoring opened BUILD truth only after output and call-journal hashes
were validated. FROZEN_DEV/FUTURE_TRAIN truth and reserved TEST/OOD were not
opened or materialized.

| RAG-slice metric | VANILLA_STRONG | CFEC_STRONG | Difference |
|---|---:|---:|---:|
| Grounded task success | 120/156 (76.92%) | 3/156 (1.92%) | -75.00 pp |
| Answer-value correct | 120/156 (76.92%) | 39/156 (25.00%) | -51.92 pp |
| Grounding pass | 129/156 (82.69%) | 64/156 (41.03%) | -41.67 pp |
| Output-contract pass | 156/156 (100%) | 3/156 (1.92%) | -98.08 pp |
| Mean external-evidence coverage | 92.09% | 42.95% | -49.14 pp |

The primary paired subject-clustered bootstrap used 10,000 resamples (seed
`20260930`): grounded-task-success delta `-0.7500`, 95% CI `[-0.8295, -0.6711]`.
The full-evidence utilization delta was `-0.7548`, 95% CI
`[-0.8356, -0.6727]`. All registered BUILD gates failed. These are BUILD
development diagnostics, not confirmatory holdout estimates.

| BUILD gate | Result |
|---|---|
| Primary point delta at least +10 pp | FAIL |
| Primary CI lower bound above zero | FAIL |
| Grounding degradation at most 1 pp | FAIL |
| Utilization point delta at least +10 pp | FAIL |

## Failure attribution

Retrieval was not the explanation: Vanilla-STRONG had full required-evidence
coverage on 155/156 RAG queries and no retrieval-contract failures. The failure
was downstream in CFEC's model-mediated decomposition and claim interface.

On the RAG slice, the decomposer produced four requirements for 144/156 queries,
three for 11, and two for one. Inspection showed paraphrase-like duplicates for
single information needs. Across 611 CFEC-STRONG claim calls, 512 emitted at
least one whitespace-padded alias such as `[ E1 ]`; the parser deliberately
accepts only the exact issued form `[E1]`. A subset also bracketed synthetic
question tokens such as `[SYNKEY-…]`, which are ordinary content rather than
evidence aliases. These categories overlap. Strict parsing correctly rejected
those outputs; the Harness did not accept model-generated IDs, evidence IDs, or
provenance. This was a generation/contract failure, not a boundary leak.

The run made 12,124 generator calls: 12,124 nonempty, 13 truncated, and zero
rejected by the prompt-context guard. The low output-contract pass rate is not
explained by truncation.

## Interpretation and decision

CFEC-v1.4's additional decomposition and claim calls introduced failure modes
that the one-pass Vanilla reader avoided. The BUILD result is strong evidence
against promoting this CFEC implementation. The experiment does not establish
that all evidence-utilization methods fail, nor does it erase separate public
R2MED retrieval results; those answer different questions and must be reported
separately.

Decision: close the CFEC method family for this protocol. Do not create a
CFEC-v1.5 by adjusting prompts against the same failures, and do not open
FROZEN_DEV or reserved TEST/OOD for CFEC. Any alternative reader method needs a
new, pre-registered BUILD protocol and a small gold-blind output-contract
preflight before full execution. A positive BUILD result would still need to pass
the frozen development/holdout process before it could support a headline claim.

## Artifact identity

- Code commit: `9389734ca0ad52d4d9f86b2ce52a93dc4d693bea`
- BUILD reader-output SHA-256: `45ce2fcf5ca0bacd7b100243f8cf081285bc976b5df937ccfca19ac15a280cfd`
- Generation-call journal SHA-256: `6264e6f876f518b4bd3c19e321bb8630b832d4992df0638c9cf7a09915cbb536`
- Score report: `runs/rag_e6/build_v5/build_score_report.json`
- Scored episodes: `runs/rag_e6/build_v5/build_scored_episodes.jsonl`
