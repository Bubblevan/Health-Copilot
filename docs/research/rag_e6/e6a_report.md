# RAG-E6A report

Status: protocol implementation and synthetic reader smoke are complete; no
BUILD/FROZEN_DEV model calls and no TRAIN evaluator-truth access have occurred.

Current verified source coordinates: U2-F TRAIN is 4,096 runtime episodes across
320 subjects; U3-R STANDARD and STRONG retrieval implementations remain the
frozen retrieval source. The E6A split is deterministically hash-assigned before
truth access. Quantitative results will be added only after the corresponding
execution freeze and authorized scoring step.

The local Qwen3-8B reader smoke used five calls over two synthetic passages only
(no benchmark or dataset rows): LameR bridge, Vanilla final, requirement
decomposition, one claim extraction, and claim-only composition. All contract
checks passed, including strict `[E#]` citation parsing. The initial smoke caught
an invalid parenthetical `(E1)` citation; the parser remained strict and the
claim prompt was clarified. A conservative pre-generation context guard now
rejects over-budget prompts before the completion endpoint is called. See
`runs/rag_e6/smoke-v3/smoke_report.json` for the passing smoke report.
