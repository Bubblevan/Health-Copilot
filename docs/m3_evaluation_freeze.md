# M3 evaluation QA and freeze checkpoint

This document records runtime freeze status and outstanding evaluation QA. It
does not modify M0--M3 frozen benchmark inputs, prompts, policies, or runtime
implementation.

| component | status |
| --- | --- |
| M0 | frozen |
| M1 | frozen |
| M2 | frozen; empirical conclusion `PARTIALLY_SUPPORTED` |
| M3 capability | frozen |
| M3 claim-first | frozen |
| M3.1 evidence bind | frozen |
| M3 result validator | frozen |
| M3.1 focused run | completed |
| M3 expansion eval | completed |
| Expansion annotation | pending human review for `m3cx-005` and `m3cx-019` |
| Fine-grained verifier | `UNSUPPORTED`/`CONTRADICTED` confusions remain |
| M4 | not started |

## Checkpoints

- Runtime implementation checkpoint:
  `4ecf06b38dc22bfe58624352d07275f27aa2cf15`
  (`fix: isolate M3 claim support evidence`).
- Evaluation-QA starting checkpoint / M3 final-freeze candidate:
  `1a1d97f191b6dc972d682319e892e2b4d1f37320`
  (`eval: add approved M3 expansion live results`).

The runtime checkpoint is frozen. Final confirmation of the candidate checkpoint
remains conditional on human disposition of the two documented capability
annotation ambiguities. No M3.2 or M4 work is authorized by this record.

## Focused-run reporting boundary

The latest three-trial M3 focused diagnostic (`runs/m3/20260920T114456+0800/`)
observed OOD tool execution `0/12`, expected answers `18/18`, unexpected
abstains `0/18`, mean model turns `1.30`, and mean tool executions `0.30`.
An earlier same-configuration small sample observed `15/18` expected answers.
These are small stochastic regression diagnostics, not stable answer-rate,
clinical-quality, or generalization estimates.

See `evals/expansion/m3_expansion_annotation_audit.md` for preservation of the
30-case capability artifact and the disposition-versus-fine-grained claim
support metric distinction.
