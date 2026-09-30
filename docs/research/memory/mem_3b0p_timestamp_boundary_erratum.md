# MEM-3B0P Timestamp-Boundary Erratum

**Applies to:** `mem3b0p-pairwise-admission-20260930`
**Disposition:** structural completion gate failed; no MEM-3B1 readiness

## Finding

The pre-inference projection code used Python's `json.loads(..., object_pairs_hook=...)`. The decoder had already decoded every object value and passed the complete key/value pair list into the hook before the hook discarded fields outside the allowlist. Therefore timestamp values may have been transiently present in that temporary pair list before semantic freeze. The count cannot be recovered from the run artifacts, so it is recorded as `UNQUANTIFIED`.

No timestamp value was retained in the projected source rows, candidate groups, pairwise requests, verdicts, human review, or downstream calculations. Pair construction and semantic judgments use only the frozen allowlisted fields and proposition text. This limits the impact but does not satisfy the literal protocol requirement that timestamp-bearing metadata must not be loaded before the semantic-freeze marker.

## Corrective Audit

After all 256 frozen pairwise requests completed, the reader was hardened to scan JSONL objects and skip non-allowlisted values before JSON decoding. The hardened reader deterministically rebuilt the candidate groups, grounding decisions, pair manifest, and contract identities. The rebuild must match the already-frozen artifacts byte-for-byte; otherwise the run is rejected. Pairwise requests were not repeated because the protocol prohibits same-request retries.

The original `semantic_freeze.json` and its sidecar are preserved as the historical freeze record. `timestamp_boundary_erratum.json` supersedes its zero timestamp-load statement. The safety closeout therefore records:

```text
MEM3B0P_PAIRWISE_REVISION_ADMISSION_COMPLETE=NO
MEM3B0P_MEM3B1_READY=NO
```

The pairwise outputs remain diagnostic evidence only. Do not integrate this run as a structurally complete MEM-3B0P stage or start MEM-3B1 from it.
