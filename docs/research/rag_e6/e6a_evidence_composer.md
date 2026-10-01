# Claim-First Evidence Composer (CFEC-v1.4 BUILD candidate)

CFEC is a reader-side execution graph over a frozen retrieval result, not a new
retriever. It is designed to isolate whether converting passages into small,
provenance-bearing claims helps answer multi-evidence questions.

1. **Decompose.** Given only the question, the model writes up to four short,
   distinct answer requirements, one per line. It should merge duplicate asks and
   exclude metadata and style/format/provenance instructions. The harness drops
   empty lines, keeps the first four, and assigns `req_1`…`req_n`; model-generated
   IDs are never used.
2. **Extract claims.** For each requirement, a separate call sees that
   requirement and the same rank-ordered top-10 evidence used by Vanilla. It
   returns only directly relevant short claims, ignores unrelated table rows or
   distractor keys, preserves exact values and value-to-entity relationships, and
   cites issued `[E#]` aliases with literal square
   brackets (e.g. `[E1]`), or `UNSUPPORTED`. Parenthetical or bare aliases are
   invalid and do not resolve to evidence.
   Unknown aliases invalidate the claim output, even when a valid alias is also
   present; claims without a valid issued alias are rejected.
3. **Compose.** The final call sees the original question, the Harness-assigned
   requirement ID/text mapping, and validated claims grouped by those IDs. It does
   not see raw evidence. The mapping is rendered by the Harness; generated text
   cannot create or change IDs or claim assignments. The composer is instructed
   to cover every distinct requested supported value, preserve exact tokens and
   pairings, and omit unrelated claims. Its final answer cannot set the
   used-evidence field: the harness computes that as the union of evidence IDs
   attached to validated claims. For the returned answer surface, the prompt
   requires exactly one `FINAL:` line and citations drawn only from those
   validated claims. A malformed, unknown-alias, or truncated output fails the
   task-success contract. Provenance remains harness-derived from validated
   claims, not trusted from generated text.

All generator calls in every arm share a 512-token response cap. This bounds
runaway completion; exceeding the cap is recorded as truncation and cannot count
as task success.

Requirement identity, evidence aliases, retrieval actions/results, provenance,
and evaluation scope are deterministic harness state. LLM output is untrusted
text and cannot alter any of those fields. There are no retries. The two CFEC
retrieval conditions share one question-only decomposition within each episode,
then make independent claim-extraction and composition calls over their own
fixed evidence.
