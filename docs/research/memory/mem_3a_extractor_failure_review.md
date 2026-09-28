# MEM-3A Extractor Failure Review

Status: forensic review only; the frozen-ten MEM-3A gate remains `NO`.

## Evidence

The initial failing response was not retained; its frozen journal contains only response SHA256 `470744fbe9aefa27d582680729913046b631828078e3d15f4dface3d7771f399` and exception type `ValueError`. At the user's request, one separate local replay was made for that session only. Its prompt/request SHA256 matched the original exactly, while its response SHA256 was `f234c0199f4712e772947f5b8559d087e99938761250deb1c0dd01ceac46941f`. Thus the exact original text cannot be recovered; output drift is observable under the same request.

The replay returned parseable JSON with ten proposition objects and failed the validator at index 1: `evidence_quote_not_exact_source_substring:1`. The wedding quote changed the source comma after “last weekend” to a period. More importantly, a later proposition quoted assistant turn 3's sample dialogue (“I've noticed that you both seem a bit stressed/tense/distant lately...”), labeled it as `source_role=user`, and cited user turns 4 and 6. Another quote occurring in user turn 10 was attributed to turn 8. Those are provenance/role errors, not harmless punctuation variation.

## Contract Assessment

- JSON-only output, the exact field set, and rejection of operation/status fields are appropriate and were not the observed failure.
- Lower-snake-case candidate keys were visibly well-formed in the replay and did not cause the failure.
- Exact-substring quote validation is semantically justified: it prevented assistant-authored hypothetical wording from being materialized as a user assertion. However, exact quote reproduction is a brittle generative output shape; zero-temperature decoding did not yield a response byte-identical to the original request's prior response, and punctuation copying drifted.
- The schema stores `source_turn_indices` and `evidence_quotes` as separate arrays, leaving their association implicit. Pair each evidence item with its source turn or use harness-issued source-span IDs in a new contract version.
- The replay also turned transient requests for advice and overlapping conversational intentions into multiple propositions. Clarify that one-off requests, hypothetical scripts, and assistant-authored example text are not durable user memories unless independently asserted as facts, decisions, or plans.

## Recommendation

Keep the hard exact-provenance gate; do not normalize punctuation or use fuzzy quote matching. In a separately versioned extractor contract, make evidence references explicit and reduce writer dependence on verbatim copying, for example by exposing deterministic source-span IDs for the model to select and reconstructing quote text in the harness. Tighten durability exclusions and add regression fixtures for assistant sample dialogue, cross-turn references, punctuation changes, and transient advice requests. Do not apply these changes to the failed MEM-3A run or present the diagnostic replay as benchmark output.
