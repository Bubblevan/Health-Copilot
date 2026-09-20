# M3 standalone claim-support evaluation

M3 verifies claims only. It does not call a semantic answer-coverage judge.

```json
{
  "case_count": 21,
  "supported_accept_rate": 1.0,
  "unsupported_reject_rate": 1.0,
  "contradicted_reject_rate": 1.0,
  "multi_claim_all_supported_accept_rate": 1.0,
  "multi_claim_one_unsupported_reject_rate": 1.0,
  "fabricated_citation_reject_rate": 1.0,
  "wrong_citation_binding_reject_rate": 1.0,
  "deterministic_citation_rejections": 1,
  "semantic_verifier_rejections": 11,
  "verifier_errors": 0
}
```

## Fixture results

- `m3sx-001` (supported): expected=['supported'], observed=[<ClaimVerdict.SUPPORTED: 'supported'>], accepted=True, stage=None.
- `m3sx-002` (contradicted): expected=['contradicted'], observed=[<ClaimVerdict.CONTRADICTED: 'contradicted'>], accepted=False, stage=semantic_verifier.
- `m3sx-003` (supported): expected=['supported'], observed=[<ClaimVerdict.SUPPORTED: 'supported'>], accepted=True, stage=None.
- `m3sx-004` (contradicted): expected=['contradicted'], observed=[<ClaimVerdict.CONTRADICTED: 'contradicted'>], accepted=False, stage=semantic_verifier.
- `m3sx-005` (supported): expected=['supported'], observed=[<ClaimVerdict.SUPPORTED: 'supported'>], accepted=True, stage=None.
- `m3sx-006` (contradicted): expected=['contradicted'], observed=[<ClaimVerdict.CONTRADICTED: 'contradicted'>], accepted=False, stage=semantic_verifier.
- `m3sx-007` (supported): expected=['supported'], observed=[<ClaimVerdict.SUPPORTED: 'supported'>], accepted=True, stage=None.
- `m3sx-008` (contradicted): expected=['contradicted'], observed=[<ClaimVerdict.CONTRADICTED: 'contradicted'>], accepted=False, stage=semantic_verifier.
- `m3sx-009` (supported): expected=['supported'], observed=[<ClaimVerdict.SUPPORTED: 'supported'>], accepted=True, stage=None.
- `m3sx-010` (wrong_citation_binding): expected=['unsupported'], observed=[<ClaimVerdict.UNSUPPORTED: 'unsupported'>], accepted=False, stage=semantic_verifier.
- `m3sx-011` (multi_claim_all_supported): expected=['supported', 'supported'], observed=[<ClaimVerdict.SUPPORTED: 'supported'>, <ClaimVerdict.SUPPORTED: 'supported'>], accepted=True, stage=None.
- `m3sx-012` (multi_claim_one_unsupported): expected=['supported', 'unsupported'], observed=[<ClaimVerdict.SUPPORTED: 'supported'>, <ClaimVerdict.CONTRADICTED: 'contradicted'>], accepted=False, stage=semantic_verifier.
- `m3sx-013` (fabricated_citation): expected=[], observed=[], accepted=False, stage=deterministic_citation.
- `m3sx-014` (wrong_citation_binding): expected=['unsupported'], observed=[<ClaimVerdict.UNSUPPORTED: 'unsupported'>], accepted=False, stage=semantic_verifier.
- `m3sx-015` (supported): expected=['supported'], observed=[<ClaimVerdict.SUPPORTED: 'supported'>], accepted=True, stage=None.
- `m3sx-016` (contradicted): expected=['contradicted'], observed=[<ClaimVerdict.CONTRADICTED: 'contradicted'>], accepted=False, stage=semantic_verifier.
- `m3sx-017` (supported): expected=['supported'], observed=[<ClaimVerdict.SUPPORTED: 'supported'>], accepted=True, stage=None.
- `m3sx-018` (contradicted): expected=['contradicted'], observed=[<ClaimVerdict.CONTRADICTED: 'contradicted'>], accepted=False, stage=semantic_verifier.
- `m3sx-019` (multi_claim_all_supported): expected=['supported', 'supported'], observed=[<ClaimVerdict.SUPPORTED: 'supported'>, <ClaimVerdict.SUPPORTED: 'supported'>], accepted=True, stage=None.
- `m3sx-020` (wrong_citation_binding): expected=['unsupported'], observed=[<ClaimVerdict.UNSUPPORTED: 'unsupported'>], accepted=False, stage=semantic_verifier.
- `m3sx-021` (unsupported): expected=['unsupported'], observed=[<ClaimVerdict.CONTRADICTED: 'contradicted'>], accepted=False, stage=semantic_verifier.
