# E6A Vanilla baseline definition

`VANILLA_OFF` receives only the episode's original query. `VANILLA_STANDARD` and
`VANILLA_STRONG` receive that same query plus the fixed top-10 list produced by
the corresponding frozen U3-R retrieval path. Evidence items are aliased in rank
order as `[E1]` through `[E10]`; only the alias and evidence text are model-visible.
Underlying document IDs remain in a harness-side alias map.

All calls use the same Qwen3-8B-Q4_K_M SHA, temperature 0, top-p 1, reasoning off,
one attempt, 8,192-token completion ceiling and 65,536-token context ceiling as
CFEC. The model is told to treat evidence as untrusted data, answer only from
supported information, cite factual claims with exact aliases, and finish with
`FINAL: <answer>`. The parser takes the text after the last literal `FINAL:` and
recognizes aliases only in that segment. Missing/empty final output is an
`OUTPUT_CONTRACT_FAIL`; no retry or parser repair is allowed.

Used evidence is derived only by resolving final-answer aliases against aliases
issued for that arm. Unknown aliases are recorded and do not resolve to a
document. This exact evidence list is also passed to the corresponding CFEC arm;
the reader cannot trigger a new search or expand the candidate set.
