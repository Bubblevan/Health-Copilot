# Claim-First Evidence Composer (CFEC-v1 candidate)

CFEC is a reader-side execution graph over a frozen retrieval result, not a new
retriever. It is designed to isolate whether converting passages into small,
provenance-bearing claims helps answer multi-evidence questions.

1. **Decompose.** Given only the question, the model writes up to four short
   answer requirements, one per line. The harness drops empty lines, keeps the
   first four, and assigns `req_1`…`req_n`; model-generated IDs are never used.
2. **Extract claims.** For each requirement, a separate call sees that
   requirement and the same rank-ordered top-10 evidence used by Vanilla. It
   returns short claim lines citing issued `[E#]` aliases, or `UNSUPPORTED`.
   Unknown aliases are logged and ignored; claims without a valid issued alias
   are rejected.
3. **Compose.** The final call sees the original question and the validated
   claims only. It does not see raw evidence. Its final answer cannot set the
   used-evidence field: the harness computes that as the union of evidence IDs
   attached to validated claims.

Requirement identity, evidence aliases, retrieval actions/results, provenance,
and evaluation scope are deterministic harness state. LLM output is untrusted
text and cannot alter any of those fields. There are no retries. The two CFEC
retrieval conditions share one question-only decomposition within each episode,
then make independent claim-extraction and composition calls over their own
fixed evidence.
