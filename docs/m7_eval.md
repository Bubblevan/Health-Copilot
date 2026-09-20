# M7 — Unified Agent Eval System

M7 implements Harness Macro H4 as an evaluation plane outside the product runtime. It
does not change M0–M6 execution semantics, retrieval configuration, policy contracts,
replay contracts, or frozen metrics. The old milestone helpers remain available for
parity; `health-eval` is the unified entry point for new artifacts.

## Contracts and execution

`EvalSuiteRegistry` is an explicit source-defined table. It does not scan entry points,
import paths from strings, install packages, or discover third-party plugins. Each
`EvalSuite` declares an ID/version, target kind, dataset path and SHA-256, supported
execution modes, default runtime profile, grader IDs, metric-definition version and
content policy. `EvalRunSpec` is the canonical identity of one run. `EvalCase` wraps
only a case ID and preserves the complete suite-specific JSON payload, including M3
evidence/query/scope state.

The runner produces `CaseRunRecord` (one case/trial), `trajectory_v1`, deterministic
`GraderResult`, inspectable `MetricResult`, and `FailureRecord`. Graders never call an
LLM. Quality denominators contain only scored PASS/FAIL results; UNGRADED and
infrastructure ERROR remain visible and are not silently treated as quality failures.
A zero denominator is represented by `value: null`. Fine-grained claim verdicts and
final disposition are separate fields.

Execution modes are explicit:

- `offline`: deterministic gates and retriever-only suites; a provider sentinel raises
  if accidentally called;
- `replay`: consumes recorded provider/tool exchanges and never creates a live
  provider; replay reports remaining exchanges and live-call status;
- `live`: disabled unless `--allow-live-provider` is present, and the check happens
  before runtime/provider construction.

Public case/trajectory content requires both suite permission and
`--public-eval-content`. Metadata-only traces contain hashes, IDs, counts and statuses;
the trace policy rejects medical question/answer/evidence fields.

## Registered historical suites

The registry contains: `m0-regression-v1`, `m1-focused-v1`, `m2-policy-v1`,
`m2-grounding-v1`, `m2-focused-v1`, `m3-capability-v1`, `m3-claim-support-v1`,
`m3-focused-v1`, `m4-replay-v1`, `m5-product-retrieval-v1`, and
`m5-focused-e2e-v1`. Dataset hashes are checked before construction. Historical M0–M5
helpers and artifact paths are not renamed.

The unified M0 run reproduces the frozen baseline: safety route accuracy 1.0,
retrieval Hit@1 0.9032258064516129, Hit@3 0.9516129032258065, and MRR
0.9274193548387096. The M5 product suite remains the component-derived 80-case suite,
not an independent external generalization test. The six-case M4/M5 focused arm is a
compatibility diagnostic; one stochastic run cannot causally attribute provider call
count or latency differences to retrieval.

## Artifact bundle

Every run is written below `runs/m7/<timestamp>/`:

```text
run_spec.json
run_manifest.json
component_manifest.json       # when a runtime was constructed
case_results.jsonl
grader_results.jsonl
failures.jsonl
trajectories.jsonl             # trajectory_v1
metrics.json                   # numerator, denominator, definition version
report.md
traces/<case>-t<trial>.jsonl
cases.jsonl                    # public suites only, with explicit opt-in
```

`run_manifest.json` records suite/version, dataset hash, run-spec hash, component
manifest hash, metric definition version, content policy and hashes of bundle files.
The manifest itself has a canonical hash. `component_manifest.json` is the M6
provenance manifest.

## CLI

```powershell
python -m health_ai_copilot.eval.cli list
python -m health_ai_copilot.eval.cli run --suite m0-regression-v1 --execution offline
python -m health_ai_copilot.eval.cli run --suite m4-replay-v1 --execution replay `
  --run-dir runs/m4/20260920T132709+0800 --public-eval-content
python -m health_ai_copilot.eval.cli run --suite m1-focused-v1 --execution live `
  --allow-live-provider --public-eval-content
python -m health_ai_copilot.eval.cli compare runs/m7/<left> runs/m7/<right>
```

`compare` rejects suite/version, dataset-hash, or metric-definition mismatches. It may
compare different profiles when the remaining evaluation identity is compatible.

## Failure taxonomy and boundary

Failures are emitted from observed evidence, not inferred by a judge. Stages include
input, safety, retrieval, action selection, policy, tool execution, grounding,
termination, budget, provider, replay, and evaluation infrastructure. Codes include
`unexpected_route`, `retrieval_miss`, `unnecessary_recovery`,
`unexpected_tool_execution`, `policy_false_veto`, `citation_integrity_failed`,
`claim_support_failed`, `max_model_turns`, `max_tool_calls`, and `replay_mismatch`.
One case may emit multiple stage-specific failures.

M7 is an evaluation system, not an agent capability expansion. M8 Agent Team, MCP,
Sandbox, Memory, multimodal runtime and post-training are not implemented here.
