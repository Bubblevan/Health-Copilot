# RAG-E5 Integration Transfer — E5-A Foundation Protocol

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
decision timestamp, a longitudinal-state reference, and the source families
allowed by the runtime catalog. Evaluator metadata is a separate type and owns
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
Fields are decision-time query, history/memory summaries, eligible capability
metadata, budgets/deadline, sanitized prior non-retrieval failures, and an
optional deterministic bounded state summary. It excludes benchmark labels,
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

The existing reviewed catalog has 30 sources, all classified as
`public_health` (catalog hash
`7cef21fceb5c04577fed2541dbefd2798900299825b711ab06c8c14e41b29c19`). The
existing capability contracts mark:

- `PUBLIC_HEALTH`: eligible.
- `GUIDELINE`: not yet eligible; no reviewed guideline/recommendation corpus is
  connected to the active retriever.
- `LITERATURE`: not yet eligible and reserved for this E5 version.

The schema, state substrate, action contracts, and leakage tests are in place,
but the requested guideline capability is not. Therefore `E5A_READY = NO` and
E5 must stop before any T2 case construction, counterfactual outcome run, oracle
calculation, policy training, or 202608 outcome inspection. See
[`e5_external_corpus.md`](e5_external_corpus.md) for the bounded next prerequisite.

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
