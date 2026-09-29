# E5-B2 Execution Erratum OPS-001

Status: operational recovery after the first committed arm; no scoring or
teacher access had occurred.

The first OFF reader inference completed exactly once and its immutable arm was
written under the frozen lock. The runner then raised while printing progress:
non-STRONG arms store `bridge: null`, but the progress logger treated that field
as a mapping. The first arm and its call-ledger entries are retained unchanged.

Recovery resumes at the next incomplete `run_id`. The compatibility adapter
normalizes nullable bridge metadata only in memory, after artifact hash
verification; it does not modify the stored arm. It records the adapter hash,
recovery commit, pre-recovery call-ledger hash, first-arm hash, and unchanged
protocol lock in `external/rag_e5/e5b2/operational_erratum.json` and the eventual
execution manifest.

No inference is retried. Prompts, retrieval profiles, model settings, scorer,
case IDs, run IDs, and the frozen lock remain unchanged. All 180 arms must still
be frozen before the evaluator may open the teacher artifact.
