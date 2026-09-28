# MEM-3A.1R - Repaired Extractor-v2 Qualification

Gate: `MEM3A1R_EXTRACTOR_V2_QUALIFIED=NO`

The historical MEM-3A.1 `NO` record remains unchanged and is classified as a response-unwrapping Harness defect. The v2 prompt/schema are unchanged. Qualification validates exact assistant content bytes extracted from the frozen OpenAI-compatible HTTP envelope, including Harness-derived evidence references.

- Qualification packets: 1/12.
- New provider calls in this repaired qualification: 1 (historical envelope imported with zero provider calls).
- Retries: 0; hosted calls: 0; provenance reconstruction failures: 0; envelope failures: 0; extractor failures: 1.
- Prompt SHA: `0aea4338d5de1e8dac753e3aae98caf36461cc15b1cf8d782a36f298f63cf627`; contract SHA: `fee5ae49c13d7b55d75c29af38146c2a9e079ac7ca24e5aeff151dee7a31b666`.

## First-Response Import

- Imported from `mem3a1-extractor-v2-qualification-20260928` for `HARNESS_RESPONSE_UNWRAPPING_DEFECT`.
- Historical HTTP envelope SHA256: `9c9357dac3b12665b5410973ca77b2e8888d188eae0abb45109257219d7b5d0b`; assistant content SHA256 after unwrap: `76b8ce570cd635d6cc30b603eb1d5f45ed7f88a1d227cf06035f3b8770ebd639`.
- The old failed journal/ledger and raw cache remain unmodified.

No benchmark labels or question answers were used. Structural qualification auto-advances to the full extraction only when the 12/12 gate passes.

## First Failure

- Code: `MEMORY_KIND_ROLE_MISMATCH`; session: `111015b58514849de9f6b2ce0f0c1355f14ce4d9dcf536a2a0adf444735a1b93`; proposition index: `0`; field: `memory_kind`.
- Detail: user_fact requires user-only evidence; got ['assistant']
- The assistant content was successfully unwrapped and the Harness evidence reference resolved. This is a genuine extractor-v2 packet validation failure, not an HTTP-envelope or provenance-reconstruction failure.
- Generated packet detail: {"evidence": [{"evidence_ref": "S0447", "source_role": "assistant", "source_turn_index": 1}], "memory_kind": "user_fact", "proposition_index": 0, "proposition_text": "The first dataset shows that calls containing the keyword \"Error\" were very few until June and August of 2020, with one error each month."}.
- No retry was issued; the full 477-session extraction and all MEM-3A.2 downstream stages were not started.
- Compact proposition/evidence-role diagnostics are in `qualification_failure_diagnostics.jsonl`; raw HTTP envelopes remain local and ignored.
