# RAG-E5-B3 — Reader Contract Measurement Recovery

This is a new reader-measurement protocol over exact frozen B2 task, state, retrieval, and bridge artifacts. B2 is not rescored or overwritten.

- B3 recovery lock: `530e511d1d16b23c6e9d861aa8fd18ad7a863a248e326fcd70daee912408e6fe`
- Execution manifest: `86b883dfb0dfce46bd9dde00377fd5e86432e6bdaeee55c5bce0da3951d56a97`
- B2 artifact-set SHA-256: `e528263a5141e6ac476252a9bfb04a1a95fc53475c5617bec23fb106980168f1`
- Cases / arms / new reader calls: 60 / 180 / 180
- New retrieval / bridge calls: 0 / 0
- Measurement recovery gate: **FAIL**

## Reader measurement comparison (no B2 quality rescore)

| Measurement | B2 | B3 |
|---|---:|---:|
| JSON-valid rate | 0.556 | 0.339 |
| finish_reason=length rate | 0.444 | 0.000 |
| state facts exactly match runtime projection / stateful arms | 0.000 | 0.008 |

The measurement gate failed; no teacher artifact was opened and no conditional-value or quality result is reported.

## Boundary

B2 outcome artifacts remain immutable. This report does not establish that B3 proves B2 wrong; it reports a separately frozen reader-contract measurement.
