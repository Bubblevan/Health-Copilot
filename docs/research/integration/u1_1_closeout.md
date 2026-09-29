# U1.1 Closeout

## Result and scope

U1.1 repairs the U1 research prototype's execution fairness and training-view
provenance on deterministic synthetic fixtures. The branch is based exactly on
U1 commit `c6df93da827a505c757ecdf064fffc7301854680` and does not include
Memory or RAG branch changes.

U1 established declared Single/Team capability equality and the surrounding
contract plumbing. It did not establish executable capability parity: the old
executor only invoked arbitrary `query_part_*` tools in its Team path, and the
synthetic `requires_team` field could affect task success. The old
`U1-TEAM -> A* = TEAM` result was fixture semantics, not evidence of a Team
benefit. The historical U1 run remains unchanged.

U1.1 gives both architectures one deterministic tool registry. Single invokes
all enabled tools sequentially; Team assigns those same tools to workers. The
executor preflights tool implementations, resource authority, and aggregate
cost before reads or calls. Executable parity compares authority, tool IDs,
implementation hashes, model/evaluator identity, and the shared global budget.
Comparative bundle attribution is emitted only when that parity is verified.

The evaluator judges answer facts, evidence, grounding, and safety without an
architecture input. Synthetic machinery expectations are stored separately
from evaluation labels and do not enter action-set derivation or training views.

## Synthetic outcome and accounting

In the corrected `U1-TEAM` fixture, both Single and Team invoke
`query_part_a` and `query_part_b` with identical implementation, input,
resource-version, and output hashes. Both succeed with answer `7`. Single is
the minimum-cost action (`A* = NONE`); Team remains available as a
counterfactual and is attributed `UNNECESSARY_TEAM` in this fixture. This is a
contract repair, not a claim about real Team systems.

The versioned abstract cost is activation plus observed execution. For this
fixture, Single costs 2 units for two tool calls; Team costs 5 units for
orchestration, two started workers, and the same two tool calls. Both use the
same 24-unit global cap. Team's worker shares partition that cap into 12 and
12 units; measured worker usage remains within those shares. These are research
units, not USD.

## Training-view provenance

All generated student actions are `SCRIPTED_PROBE`. The student packet uses
the selected arm's execution-visible outcome and contains no evaluator or
teacher fields. OPD is `SCHEMA_PROBE`; no policy rollout or optimization was
run. GRPO-compatible groups identify their source as
`COUNTERFACTUAL_ENUMERATION`, not policy sampling. SFT candidates identify
`DETERMINISTIC_COUNTERFACTUAL_ORACLE` and record backend, evaluator, cost
model, epsilon, and all evaluated arms.

## Data and track boundaries

The run uses nine synthetic episodes, makes zero provider calls, and accesses
no external benchmark or evaluation/test rows. Runtime, SFT, and student
payloads contain no evaluation gold; student payloads contain no privileged
counterfactual values. The new run is
`runs/integration/u1-1-fairness-20260929-01/`; the historical
`runs/integration/u1-synthetic-20260929-01/` was not modified.

No production Memory or RAG implementation was bound. E2-B, L4, and
post-training were not started. U0 commit `df8cac1fafbbb8851aa2481c85c311c045c01230`
is `LOCAL_ONLY`: no remote branch contains it, and it remains on the existing
local `codex/rag-e5-foundation-20260929` history. It was not cherry-picked or
included here; no RAG branch was pushed as a proxy for a dedicated U0 ref.

## Verification

All Python tooling used the repository's `.venv` through `uv run`.

- `pytest -q --ignore=tests/test_e0_benchmark.py`: 619 passed, 2 skipped.
- `ruff check .`: passed.
- `python -m compileall -q src`: passed.
- `git diff --check`: passed.
- Run-bundle audit: all required U1.1 gates pass; all student-action sources
  are `SCRIPTED_PROBE`; OPD is `SCHEMA_PROBE`; GRPO is
  `COUNTERFACTUAL_ENUMERATION`; no benchmark rows or locked gold were read.

The pytest run reported one dependency deprecation warning from `jieba`'s
`pkg_resources` import. It did not report missing project dependencies.

## Gates

| Gate | Result |
| --- | --- |
| `U1_1_DECLARED_CAPABILITY_EQUIVALENCE` | YES |
| `U1_1_EXECUTABLE_CAPABILITY_EQUIVALENCE` | YES |
| `U1_1_SINGLE_SHARED_TOOLS_EXECUTABLE` | YES |
| `U1_1_TEAM_HAS_NO_EXTRA_DATA_AUTHORITY` | YES |
| `U1_1_ARCHITECTURE_AGNOSTIC_EVALUATOR` | YES |
| `U1_1_REQUIRES_TEAM_GOLD_REMOVED_OR_QUARANTINED` | YES |
| `U1_1_SINGLE_TOOL_COST_ACCOUNTED` | YES |
| `U1_1_GLOBAL_BUDGET_PARITY` | YES |
| `U1_1_TEAM_VALUE_COUNTERFACTUAL_ONLY` | YES |
| `U1_1_STUDENT_ACTION_SOURCE_EXPLICIT` | YES |
| `U1_1_NO_POLICY_SAMPLE_TAG_GENERATED` | YES |
| `U1_1_OPD_STATUS` | SCHEMA_PROBE |
| `U1_1_GRPO_ROLLOUT_SOURCE` | COUNTERFACTUAL_ENUMERATION |
| `U1_1_RUNTIME_EVAL_PRIVILEGED_ISOLATION` | YES |
| `U1_1_EXTERNAL_TEST_CONTENT_ACCESSED` | NO |
| `MEMORY_TRACK_MODIFIED` / `RAG_TRACK_MODIFIED` | NO / NO |
| `E2_B_STARTED` / `L4_STARTED` / `POST_TRAINING_STARTED` | NO / NO / NO |

## Next stage

If the research owner accepts this contract repair, the recommended next stage
is U2-D dataset, split, and license qualification. U1.1 does not start U2-D.
