# RAG-E5 Integration Transfer — E5-A Qualification Protocol v2

`qualification_semantics = rag-e5-index-qualification-v2`. This protocol
version separates corpus/index artifact integrity from retrieval effectiveness.
The correction and before/after protocol hashes are recorded in the
[`E5-A3.1 qualification erratum`](e5a31_qualification_erratum.md).

## Stage boundary

This is an integration-transfer study, not a new RAG subsystem sprint. The
R2MED, MIRAGE, and question-only-router closeouts remain frozen. E5 asks whether
decision-time longitudinal state, internal memory, eligible source families,
and resource constraints make the value of a retrieval action more predictable
inside a medical-agent harness.

The policy action space is exactly `OFF`, `STANDARD`, and `STRONG`. The answer
model, memory packet, prompt, verifier, and all non-retrieval execution settings
must be shared across the three counterfactual arms. E5 does not reopen R2MED or
MIRAGE results and does not train a policy in E5-A.

## Pinned inputs and untouched boundaries

- Health-Copilot base commit: `9e023dbddc1f8025e3608bfae4c5390c1a7957ef`.
- External baseline project: `D:/MyLab/Jianli/external/rag/R2MED`, commit
  `11244a4925a39082967a6c9d38ef01f279c316a5`.
- Frozen R2MED evaluation lock SHA-256:
  `0b80fad6668c3a57833c55beb1db359c65440cf015effd10da770c630fb3e41d`.
- R2MED source manifest SHA-256:
  `b70c4f01b37f58c77597f1e28cc35a52585785142f3f625f928173c81be3874e`.
- ESL-Bench raw source identity and allowed file list are pinned in
  [`runs/rag_e5/esl_source_manifest.json`](../../../runs/rag_e5/esl_source_manifest.json).

E5 reads only the per-user ESL-Bench `profile.json`, `timeline.json`, and
`exam_data.json` as its longitudinal substrate. The source audit does not open
native benchmark question JSONL, released answer files, or
`kg_evaluation_queries.json`; no 202608 native outcome is inspected. The 202607
and 202608 user cohorts are disjoint and remain a user-level split.

## Integration-task contract

An `E5IntegrationCase` contains only case/user/batch identity, the task question,
decision timestamp, and a longitudinal-state reference. Runtime capability
metadata is constructed separately from environment source families, the
active corpus identity, permissions, and budget state. Evaluator metadata is a separate type and owns
the task-family label, dependency assertions, and evaluation payload reference.
It is never passed to the execution policy.

- T0 — longitudinal-only: requires internal state; no external evidence group.
- T1 — external-only: requires an external evidence group; no patient-specific
  state dependency.
- T2 — longitudinal × external: requires both internal-state fields and an
  external evidence group. Removing either input must make the task incomplete
  or unsolvable under the frozen evaluator.

The task-family label and dependency annotations are teacher/evaluator-only.
They may be used for stratified reporting after execution, never as a policy
feature. Task examples and answers are not constructed in E5-A.

## Policy observation contract

The sole policy input is
`health_ai_copilot.execution_policy.ExecutionPolicyObservation`. Its frozen
field-level contract is
[`runs/rag_e5/feature_contract.json`](../../../runs/rag_e5/feature_contract.json).
Fields are decision-time query, history/memory summaries, environment-sourced
capability metadata, budgets/deadline, sanitized prior non-retrieval failures,
and an optional deterministic bounded state summary. Capability metadata has
`capability_context_source="environment"` provenance and is independent of T0,
T1, and T2 teacher annotations. It excludes benchmark labels,
gold answers/source groups, retrieval relevance, post-retrieval signals, and all
three counterfactual outcomes. `from_dict` rejects unknown fields, and the
separate leakage guard rejects evaluator-only fields even when nested.

The three frozen views are V0 (query only), V1 (query plus all structured
metadata), and V2 (V1 plus the bounded state summary). Their exact field sets
are recorded beside the field-level contract so a future observability study
cannot quietly change view composition.

No Jev router, multi-agent/team assignment, or learned intent label is included.
Any future task intent must be derived from the current query/state at runtime
and frozen as a new feature contract before use.

## Frozen action profiles

Exact profile parameters and provenance are recorded in
[`runs/rag_e5/retrieval_action_profiles.json`](../../../runs/rag_e5/retrieval_action_profiles.json)
and typed by `RetrievalCapabilitySpec`.

| Action | Bound profile | Frozen behavior | Generator |
| --- | --- | --- | --- |
| OFF | none | No external retrieval or external evidence | No |
| STANDARD | `r2med-bm25-bge-rrf-v1` | Lucene BM25 (`k1=.9`, `b=.4`) + pinned BGE-large; top-100 each; RRF `k=60`, weights `[1,1]` | No |
| STRONG | `r2med-lamer-mv-v1` | Frozen LameR-MV: BM25 top-10 feedback; BM25(original/bridge) and BGE-large(original/bridge), top-100; RRF `k=20`, weights `[1,2,1,2]` | Pinned Qwen3-8B, one call, temperature 0, reasoning off, 256-token cap |

One E5 action is one harness-level external retrieval invocation; its fixed
profile may execute two or four internal retrieval views. `source_families` in
the specs are allowlists, not evidence that every family is currently eligible.
The action resolver intersects them with the active reviewed source catalog and
fails closed if no family is eligible.
The retrieval algorithms and model identities are bound, but the external
corpus identity remains null until the reviewed guideline corpus is ingested and
frozen.

STRONG reuses the upstream R2MED LameR method definition from the external
baseline at the pinned commit. The E5 question adapter is frozen to the pinned
generic medical-exam LameR template without prompt edits. The upstream
`generate_hypothetical_doc.py` data loader is **not** reusable: it constructs
fields from qrels/relevant documents. A future E5 runner must build the prompt
from the current question and the same query's BM25 top-10 only.

## Current E5-A capability gate

The E5-A2 owner-review snapshot is historical and has been superseded by the
E5-A3 owner decision and the E5-A3.1 qualification semantics correction. The
owner approved three WHO guideline sources; physical-activity and total-fat
sources are task-authoring eligible, while hypertension pharmacology is
retrieval-context/distractor only. Deterministic extraction retained all 22
declared recommendation sections in 26 guideline chunks. The existing 30
reviewed public-health cards were mapped one-to-one without changing their text
or source family.

E5-A3.1 independently validated corpus, BM25, and dense-index integrity. All
three views contain their expected documents and unique ordered IDs; the
candidate/raw/text/model/index hashes match; BM25 and dense structural search
probes are finite and deterministic; and all 112 dense vectors are finite,
normalized within `1e-5`, and nonzero. The same candidate identity
`9b19ad467f47641032277707cb1cfdb1d05c2fb39c558039b180efc7394692bd` is now the
active frozen external corpus. STANDARD and STRONG are bound to it without
changing either action's frozen method-config hash.

Source-title smoke is retained as a retrieval diagnostic, not an index gate.
BM25 is 10/10 for all three views. Frozen BGE is 8/10 for
`PUBLIC_HEALTH_ONLY`; two Chinese CDC probes rank their expected sources 18th
and 15th. `GUIDELINE_ONLY` and the existing combined 10-query sample are 10/10,
but the combined sample does not include those two public-health queries. This
is a known transfer limitation, plausibly contributed to by applying the
frozen English embedding model to Chinese public-health text; that causal
explanation is not proven. No smoke query, model, cutoff, corpus, chunk, or
index was modified to obtain activation. See the
[`E5-A3.1 structural integrity report`](../../../runs/rag_e5/e5a31_dense_integrity_report.json),
[`retained retrieval diagnostic`](../../../runs/rag_e5/e5a31_retrieval_diagnostic.json),
[`language inventory`](../../../runs/rag_e5/external_language_inventory.json), and
[`final activation report`](../../../runs/rag_e5/e5a_final_activation_report.json).

`E5A_READY = YES`. This is corpus/index capability readiness only; it is not
evidence that retrieval improves downstream outcomes. No E5-B task construction,
counterfactual run, oracle calculation, policy training, or 202608 outcome
inspection occurred in E5-A.

The A2 details remain in the
[`historical E5-A2 qualification record`](e5a2_guideline_corpus.md); they
describe the state before owner approval and must not be read as the current
activation status.

## Stage gates and stop rules

E5-B may start only after a reviewed guideline corpus is admitted to the active
retriever, its source-family catalog and document hashes are frozen, and the
E5-A gate is explicitly rerun. E5-B then runs OFF/STANDARD/STRONG on retired
development users only. Policy training is permitted only if the predeclared
action-diversity, oracle-headroom, and T2-dependency gates pass. E5-C uses
user-disjoint folds. The 202608 user batch remains a future holdout until a
separate method lock is committed; it is never used to tune task construction or
features.

R2MED tuning, MIRAGE labels, Jev integration, Multi-Agent routing, RL, model
changes, and a new retriever are outside this protocol.
