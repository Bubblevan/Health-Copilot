# M4 public record/replay closeout

## Scope

This artifact records a fixed six-case public regression diagnostic derived verbatim from reviewed M1 fixtures:
two paraphrase recoveries, two direct-hit controls, and two OOD controls. It does not alter M1, M2, or M3 gold and
does not estimate clinical quality or generalization.

## Live record

- Route match: 6/6.
- OOD tool executions: 0/2.
- Provider calls: 16; tool executions: 2.
- The two paraphrase cases each used Agent → policy → recovery tool → Agent → claim-support verifier.
- The OOD cases each reached an `INSUFFICIENT` policy decision and did not execute the tool.

## Semantic replay

- Route match: 6/6.
- All provider exchanges were consumed: 16/16.
- All recorded tool exchanges were consumed: 2/2.
- Replay provider/tool executors contain no live provider client or registry dispatch path. Request or tool mismatch fails closed.
- Replay is a control-flow replay using captured exchanges, not retrieval of a saved final response.

## Privacy and limitations

The artifact uses `public_eval_content` because its fixtures and exchanges are intentionally public evaluation content.
It persists no API key. Production traces default to `metadata_only`, which rejects content-bearing fields. Same-model
generation/policy/verification remains same-model verification, not independent validation. M5 was not started.
