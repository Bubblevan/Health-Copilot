# E5-B4 CPU pre-score manifest erratum

Before teacher labels were opened, execution-manifest review found an aggregation-only bug: the runner appended token counters for all 180 arms, including 60 no-model-call arms with zero usage, but then tested whether the resulting list contained exactly 120 entries. This incorrectly set `token_usage_complete=false` and left the aggregate totals null even though all 120 actual generation arms carried usage.

`tools/research/rag_e5/repair_e5b4_cpu_usage_manifest.py` corrects only the two aggregate token totals and the completeness flag. It cross-checks all 180 arm-file hashes, the 120 generation-arm raw usage fields, and the append-only STARTED/COMPLETED ledger pairs. It refuses missing/inconsistent usage, failed calls, changed arms, a changed ledger, or a manifest that is not the exact pre-score run.

The original runner-produced manifest is preserved byte-for-byte at `runs/rag_e5/e5b4_cpu_execution_manifest_pre_erratum.json`. The canonical manifest contains a pointer to the erratum, while `runs/rag_e5/e5b4_cpu_manifest_erratum.json` records before/after hashes and derived totals. No prompt, response, arm, action, state claim, citation, retrieval, score, or method setting is changed. The correction reads no teacher/qrels data and runs before scoring.

The cause is a deterministic bookkeeping defect, not missing usage from the model server. Never infer or synthesize token counts; the correction is accepted only if every completed model-call arm and matching ledger row contains the same nonnegative integer values.
