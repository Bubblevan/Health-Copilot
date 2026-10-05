# Parser v7 gold-blind audit

## Scope

This offline audit re-applies the frozen deterministic parser v7 to the four frozen B0/B2 checkpoints. It reads candidate-view case IDs and answer-schema metadata plus each checkpoint's `answer_text`, `parsed_answer`, and `safety_flags`. It does not read scorer gold, `score.correct`, or any evaluator labels. It makes no provider calls and does not change the checkpoints.

The machine-readable case inventory and input hashes are in [`audit.json`](../../runs/common_eval/harness-v1-base-20261005/parser-audit-v7-gold-blind/audit.json).

## Results

| Dataset | Arm | Parsed | Unparsed classification |
|---|---|---:|---|
| CMB-COMMON-1024 | B0 | 969/1,024 | 55 safety-gate routes |
| CMB-COMMON-1024 | B2 | 960/1,024 | 55 safety-gate routes; 9 Adaptive runtime failures |
| DiagnosisArena-915 | B0 | 915/915 | none |
| DiagnosisArena-915 | B2 | 685/915 | 230 Adaptive runtime failures |

The audit found no stored-vs-recomputed parser mismatch and no unparsed answer with a simple, explicit final-choice candidate that the parser had missed. The unparsed answers in these checkpoints are explained by safety routes or runtime failure records; none were silently dropped from the accuracy denominator.

## Decision

Freeze parser v7 for the upcoming system factorial and reuse it for Base, SFT, and GSPO tracks. Runtime failure, safety route, model abstention, and parser failure remain separate reported categories; all frozen cases stay in the primary accuracy denominator. If a later model emits a new explicit answer format, review unparsed outputs without gold first. A confirmed parser correction must be applied uniformly and all existing arms must be rescored offline before comparing models.

This audit covers existing no-thinking B0/B2 checkpoints. It does not establish parser behavior for the new thinking-enabled system protocol; its checkpoints must undergo the same gold-blind audit before scores are frozen.
