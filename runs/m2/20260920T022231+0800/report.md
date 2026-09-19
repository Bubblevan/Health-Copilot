# M2 standalone grounding evaluation

## Metrics

```json
{
  "case_count": 5,
  "supported_accept_rate": 1.0,
  "unsupported_reject_rate": 1.0,
  "contradicted_reject_rate": 1.0,
  "coverage_failure_reject_rate": 0.0,
  "fabricated_citation_reject_rate": 1.0,
  "deterministic_citation_rejections": 1,
  "semantic_verifier_rejections": 2,
  "verifier_errors": 0
}
```

## Fixture verdicts

- `m2g-001` (supported): expected `['supported']`, observed `[<ClaimVerdict.SUPPORTED: 'supported'>]`, accepted=`True`, rejection_stage=`None`.
- `m2g-002` (unsupported): expected `['unsupported']`, observed `[<ClaimVerdict.UNSUPPORTED: 'unsupported'>]`, accepted=`False`, rejection_stage=`semantic_verifier`.
- `m2g-003` (contradicted): expected `['contradicted']`, observed `[<ClaimVerdict.UNSUPPORTED: 'unsupported'>]`, accepted=`False`, rejection_stage=`semantic_verifier`.
- `m2g-004` (coverage_missing): expected `['supported']`, observed `[<ClaimVerdict.SUPPORTED: 'supported'>]`, accepted=`True`, rejection_stage=`None`.
- `m2g-005` (fabricated_citation): expected `[]`, observed `[]`, accepted=`False`, rejection_stage=`deterministic_citation`.
