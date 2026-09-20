# M3 standalone claim-support evaluation

M3 verifies claims only. It does not call a semantic answer-coverage judge.

```json
{
  "case_count": 7,
  "supported_accept_rate": 1.0,
  "unsupported_reject_rate": 1.0,
  "contradicted_reject_rate": 1.0,
  "multi_claim_all_supported_accept_rate": 0.0,
  "multi_claim_one_unsupported_reject_rate": 1.0,
  "fabricated_citation_reject_rate": 1.0,
  "wrong_citation_binding_reject_rate": 1.0,
  "deterministic_citation_rejections": 1,
  "semantic_verifier_rejections": 5,
  "verifier_errors": 0
}
```

## Fixture results

- `m3s-001` (supported): expected=['supported'], observed=[<ClaimVerdict.SUPPORTED: 'supported'>], accepted=True, stage=None.
- `m3s-002` (unsupported): expected=['unsupported'], observed=[<ClaimVerdict.UNSUPPORTED: 'unsupported'>], accepted=False, stage=semantic_verifier.
- `m3s-003` (contradicted): expected=['contradicted'], observed=[<ClaimVerdict.CONTRADICTED: 'contradicted'>], accepted=False, stage=semantic_verifier.
- `m3s-004` (multi_claim_all_supported): expected=['supported', 'supported'], observed=[<ClaimVerdict.UNSUPPORTED: 'unsupported'>, <ClaimVerdict.SUPPORTED: 'supported'>], accepted=False, stage=semantic_verifier.
- `m3s-005` (multi_claim_one_unsupported): expected=['supported', 'unsupported'], observed=[<ClaimVerdict.SUPPORTED: 'supported'>, <ClaimVerdict.UNSUPPORTED: 'unsupported'>], accepted=False, stage=semantic_verifier.
- `m3s-006` (fabricated_citation): expected=[], observed=[], accepted=False, stage=deterministic_citation.
- `m3s-007` (wrong_citation_binding): expected=['unsupported'], observed=[<ClaimVerdict.UNSUPPORTED: 'unsupported'>], accepted=False, stage=semantic_verifier.
