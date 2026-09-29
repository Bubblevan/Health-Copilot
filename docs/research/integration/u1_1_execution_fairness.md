# U1.1 Execution Semantics Fairness

## Scope and status

This is a research-only deterministic contract repair using synthetic fixtures.
It calls no provider, runs no model training, binds no production Memory/RAG
implementation, and reads no external benchmark rows.

The invariant is:

```text
ExecutableCapability(SINGLE)
==
Union(ExecutableCapability(TEAM workers))
```

The existing manifest comparison remains available as
`capability_equivalence_report`. `executable_capability_equivalence_report`
adds registered implementation coverage and implementation hashes. Team
workers use partitions of the common tool set. The executor uses one shared
registry for either architecture, and the parity report detects mismatched
implementation hashes when comparing alternate registries. Resource authority
and missing implementations are checked before invoking a tool.

## Audit finding

At the U1 base, `capability_equivalence_report` compared declared tool IDs, but
`DeterministicIntegrationExecutor.execute` invoked `query_part_*` only inside
the Team branch. A tool could therefore be declared for Single without a
Single execution path. The U1 `U1-TEAM` fixture compounded this by setting
`EvaluationPlane.requires_team`, which made Team usage part of synthetic task
success.

Before the U1.1 repair, the correct finding is:

```text
U1_EXECUTABLE_CAPABILITY_PARITY = NOT_ESTABLISHED
```

The previous `U1-TEAM -> A* = TEAM` result is a contract-fixture result. It is
not empirical evidence that Team orchestration improves task success.

## Shared tool execution

`DeterministicToolRegistry` registers one implementation per tool ID. Each
implementation is fingerprinted from its tool ID, version, and source. A
`ToolInvocation` has architecture-neutral inputs: tool ID, query, decision
time, and the episode's resource versions and scopes. The executor does not
pass architecture or worker identity to the implementation. Single calls every
granted non-disabled tool sequentially; Team assigns those same calls to
workers. `memory_read` and `external_retrieval` are also registered tools and
respect their `CapabilityAction` switches.

Every observation records input hash, implementation hash, resource versions,
resource IDs, output, and output hash. Equal tool inputs against equal resource
versions therefore produce the same deterministic observation for both
architectures. A failed capability check or budget preflight occurs before the
first tool call or resource read.

## Evaluation and attribution

`EvaluationPlane` describes required facts, evidence, memory records, and
abstention conditions only. It contains no architecture preference. The
evaluator no longer receives `CapabilityAction`; it judges the observed answer,
evidence, safety, and grounding.

Arm failure categories remain evaluator outcomes. Comparative interpretations
are `BundleCounterfactualAttribution` values derived after corresponding Single
and Team arms run. The possible outcomes include
`ORCHESTRATION_GAIN_CANDIDATE`, `UNNECESSARY_TEAM`, and `TEAM_FAILURE`. A
candidate is not a causal claim; later controlled evaluation is still required.
Synthetic fixture expectations live only in `SyntheticContractExpectation`
and never enter evaluator success, minimal-action labels, or training views.

## Cost and budget

The versioned abstract model records activation and actual usage separately:

```text
TotalAbstractCost = ActivationCost + ObservedUsage
```

Activation records memory, external retrieval, and Team orchestration. Usage
records workers actually started, tools executed, memory reads, retrieval
executions, and per-worker units. Single tool calls incur the same tool-use unit
as Team tool calls. Team workers receive partitions of the episode's one global
budget; the worker shares sum to the global cap. The executor plans the complete
call set and rejects an over-budget action before execution. Units are
versioned abstract research units; no USD cost is inferred.

## Synthetic contract outcome

For `U1-TEAM`, both Single and Team invoke `query_part_a` and `query_part_b`
through the same implementations and return `7`. Team pays orchestration and
worker-start costs, so the synthetic minimum-cost action is `NONE` (Single).
This is a fairness repair, not a negative empirical result about Team systems.
