from dataclasses import fields, replace

import pytest

from health_ai_copilot.research.integration.actions import (
    ACTION_BY_KEY,
    ActionKey,
    capability_equivalence_report,
    executable_capability_equivalence_report,
)
from health_ai_copilot.research.integration.contracts import EvaluationPlane
from health_ai_copilot.research.integration.counterfactual import (
    BundleAttributionCategory,
    CounterfactualRunner,
    attribute_bundle_outcomes,
)
from health_ai_copilot.research.integration.evaluator import DeterministicIntegrationEvaluator
from health_ai_copilot.research.integration.executor import (
    DeterministicIntegrationExecutor,
    ExecutionResources,
)
from health_ai_copilot.research.integration.fixtures import build_synthetic_cases
from health_ai_copilot.research.integration.tools import (
    DeterministicToolRegistry,
    ToolInvocation,
    default_tool_registry,
)
from health_ai_copilot.research.integration.training_views import (
    OpdDataStatus,
    StudentActionSource,
    build_training_views,
)


def _case(case_id: str):
    return next(case for case in build_synthetic_cases() if case.case_id == case_id)


class RecordingRegistry(DeterministicToolRegistry):
    def __init__(self, source: DeterministicToolRegistry):
        super().__init__(source.registered_tools)
        self.calls: list[str] = []

    def invoke(self, request: ToolInvocation):
        self.calls.append(request.tool_id)
        return super().invoke(request)


class CountingStore:
    def __init__(self, store):
        self.store = store
        self.snapshot_calls = 0

    def snapshot(self, subject_id, as_of_time):
        self.snapshot_calls += 1
        return self.store.snapshot(subject_id, as_of_time)


class CountingWorld:
    def __init__(self, world):
        self.world_id = world.world_id
        self.version = world.version
        self.world = world
        self.retrieve_calls = 0

    def retrieve(self, *args, **kwargs):
        self.retrieve_calls += 1
        return self.world.retrieve(*args, **kwargs)


def _single_and_team_results(case):
    executor = DeterministicIntegrationExecutor()
    single = executor.execute(case.episode, ACTION_BY_KEY[ActionKey.NONE], case.resources)
    team = executor.execute(case.episode, ACTION_BY_KEY[ActionKey.TEAM], case.resources)
    return single, team


def test_manifest_and_executable_capability_parity_include_implementations() -> None:
    case = _case("U1-TEAM")
    declared = capability_equivalence_report(case.episode)
    executable = executable_capability_equivalence_report(case.episode)
    assert declared.equivalent
    assert executable.equivalent
    assert executable.single.tool_ids == ("query_part_a", "query_part_b")
    assert executable.single.tool_implementation_hashes == executable.team_union.tool_implementation_hashes
    assert not executable.single.unimplemented_tool_ids
    assert not executable.team_union.unimplemented_tool_ids


def test_single_and_partitioned_team_run_the_same_shared_tool_implementations() -> None:
    case = _case("U1-TEAM")
    single, team = _single_and_team_results(case)
    assert single.outcome.answer == team.outcome.answer == "7"
    assert single.outcome.tool_calls == team.outcome.tool_calls == 2
    assert [item.tool_id for item in single.tool_observations] == ["query_part_a", "query_part_b"]
    assert [item.tool_id for item in team.tool_observations] == ["query_part_a", "query_part_b"]
    assert [item.implementation_hash for item in single.tool_observations] == [
        item.implementation_hash for item in team.tool_observations
    ]
    assert [item.resource_versions for item in single.tool_observations] == [
        item.resource_versions for item in team.tool_observations
    ]
    assert [item.input_hash for item in single.tool_observations] == [
        item.input_hash for item in team.tool_observations
    ]
    assert [item.output_hash for item in single.tool_observations] == [
        item.output_hash for item in team.tool_observations
    ]
    assert all(item.resource_ids == () for item in single.tool_observations + team.tool_observations)
    assert sum(event.kind.value == "TOOL_CALL" for event in single.trace) == 2
    assert sum(event.kind.value == "TOOL_CALL" for event in team.trace) == 2
    assert sum(event.kind.value == "TEAM_DELEGATION" for event in single.trace) == 0
    assert sum(event.kind.value == "TEAM_DELEGATION" for event in team.trace) == 1


def test_evaluator_success_does_not_depend_on_architecture() -> None:
    case = _case("U1-TEAM")
    single, team = _single_and_team_results(case)
    evaluator = DeterministicIntegrationEvaluator()
    single_result = evaluator.evaluate(
        single.outcome, case.evaluation, observed_evidence_ids=single.observed_evidence_ids
    )
    team_result = evaluator.evaluate(
        team.outcome, case.evaluation, observed_evidence_ids=team.observed_evidence_ids
    )
    assert single.outcome.answer == team.outcome.answer
    assert single_result.outcome.task_success is team_result.outcome.task_success is True
    assert single_result.outcome.safety_pass is team_result.outcome.safety_pass is True
    assert single_result.outcome.grounding_pass is team_result.outcome.grounding_pass is True
    assert "requires_team" not in {item.name for item in fields(EvaluationPlane)}
    assert "requires_team" not in case.evaluation.to_evaluation_dict()
    assert case.contract_expectation.expected_team_tools == ("query_part_a", "query_part_b")


def test_team_tool_implementation_mismatch_fails_executable_equivalence() -> None:
    case = _case("U1-TEAM")
    original = default_tool_registry()

    def changed_query_part(request: ToolInvocation):
        return "different", ()

    altered_rows = [
        (tool.tool_id, tool.version,
         changed_query_part if tool.tool_id == "query_part_a" else tool.implementation)
        for tool in original.registered_tools
    ]
    altered = DeterministicToolRegistry.from_implementations(altered_rows)
    report = executable_capability_equivalence_report(
        case.episode, single_registry=original, team_registry=altered
    )
    assert not report.equivalent
    assert "tool_implementation_hashes" in report.mismatches


def test_single_declared_tool_without_implementation_fails_before_tool_call() -> None:
    case = _case("U1-TEAM")
    base = default_tool_registry()
    registry = RecordingRegistry(DeterministicToolRegistry(
        tool for tool in base.registered_tools if tool.tool_id != "query_part_a"
    ))
    executor = DeterministicIntegrationExecutor(registry)
    result = executor.execute(case.episode, ACTION_BY_KEY[ActionKey.NONE], case.resources)
    assert result.trace[0].kind.value == "ACTION_REJECTED"
    assert "NO_DETERMINISTIC_IMPLEMENTATION:query_part_a" in result.trace[0].detail
    assert registry.calls == []


def test_worker_undeclared_extra_tool_fails_before_execution() -> None:
    case = _case("U1-TEAM")
    surface = case.episode.tool_surface_ref
    bad_worker = replace(surface.workers[0], tool_ids=(*surface.workers[0].tool_ids, "undeclared_tool"))
    bad_surface = replace(surface)
    object.__setattr__(bad_surface, "workers", (bad_worker, surface.workers[1]))
    bad_episode = replace(case.episode, tool_surface_ref=bad_surface)
    registry = RecordingRegistry(default_tool_registry())
    result = DeterministicIntegrationExecutor(registry).execute(
        bad_episode, ACTION_BY_KEY[ActionKey.TEAM], case.resources
    )
    assert result.trace[0].kind.value == "ACTION_REJECTED"
    assert "tool_ids" in result.trace[0].detail
    assert registry.calls == []


@pytest.mark.parametrize("scope_kind", ["patient", "external"])
def test_worker_extra_resource_scope_fails_before_resource_read(scope_kind: str) -> None:
    case = _case("U1-ALL")
    surface = case.episode.tool_surface_ref
    worker = surface.workers[0]
    if scope_kind == "patient":
        worker = replace(worker, personal_state_scopes=(*worker.personal_state_scopes, "EXTRA_SCOPE"))
    else:
        worker = replace(worker, external_source_families=(*worker.external_source_families, "EXTRA_FAMILY"))
    bad_surface = replace(surface, workers=(worker, surface.workers[1]))
    bad_episode = replace(case.episode, tool_surface_ref=bad_surface)
    store = CountingStore(case.resources.patient_state_store)
    world = CountingWorld(case.resources.external_evidence_world)
    resources = ExecutionResources(store, world)
    registry = RecordingRegistry(default_tool_registry())
    result = DeterministicIntegrationExecutor(registry).execute(
        bad_episode, ACTION_BY_KEY[ActionKey.ALL], resources
    )
    assert result.trace[0].kind.value == "ACTION_REJECTED"
    assert result.trace[0].namespace == "ACTION_MASK"
    assert registry.calls == []
    assert store.snapshot_calls == 0
    assert world.retrieve_calls == 0


def test_global_budget_is_shared_and_excess_is_rejected_before_reads() -> None:
    case = _case("U1-MEM")
    counted = CountingStore(case.resources.patient_state_store)
    too_small = replace(case.episode.budget, global_units=1)
    state = replace(case.episode.observable_state, budget_class=too_small.budget_class)
    episode = replace(case.episode, budget=too_small, observable_state=state)
    result = DeterministicIntegrationExecutor().execute(
        episode, ACTION_BY_KEY[ActionKey.MEMORY], ExecutionResources(counted, None)
    )
    assert result.trace[0].kind.value == "ACTION_REJECTED"
    assert "GLOBAL_BUDGET_UNAVAILABLE" in result.trace[0].detail
    assert counted.snapshot_calls == 0

    case = _case("U1-TEAM")
    bundle = CounterfactualRunner().run(case.episode, case.evaluation, case.resources)
    single = next(item for item in bundle.arms if item.action_key == ActionKey.NONE)
    team = next(item for item in bundle.arms if item.action_key == ActionKey.TEAM)
    assert single.cost.tool_units == team.cost.tool_units == 2
    assert team.cost.total_units > single.cost.total_units
    assert team.execution.budget_audit.global_budget_units == single.execution.budget_audit.global_budget_units
    assert team.execution.budget_audit.worker_budgets_sum_to_global
    assert team.execution.budget_audit.within_global_budget
    assert all(units <= dict(team.execution.budget_audit.worker_budgets)[worker]
               for worker, units in team.execution.budget_audit.worker_used_units)


def test_team_value_is_bundle_attribution_and_minimal_action_prefers_single() -> None:
    case = _case("U1-TEAM")
    bundle = CounterfactualRunner().run(case.episode, case.evaluation, case.resources)
    assert tuple(item.value for item in bundle.oracle_action_set.actions) == ("NONE",)
    assert any(item.category == BundleAttributionCategory.UNNECESSARY_TEAM
               for item in bundle.attributions)
    assert all(item.category.value != "MISSING_TEAM" for item in bundle.attributions)


def test_bundle_can_emit_only_candidate_orchestration_gain_after_parity() -> None:
    case = _case("U1-TEAM")
    bundle = CounterfactualRunner().run(case.episode, case.evaluation, case.resources)
    single, team = bundle.arms
    failed_single = replace(single, evaluation=replace(
        single.evaluation,
        outcome=replace(single.evaluation.outcome, task_success=False),
    ))
    attrs = attribute_bundle_outcomes((failed_single, team), capability_parity_verified=True)
    assert [item.category for item in attrs] == [BundleAttributionCategory.ORCHESTRATION_GAIN_CANDIDATE]
    assert attribute_bundle_outcomes((failed_single, team), capability_parity_verified=False) == ()


def test_training_views_require_explicit_scripted_action_and_label_provenance() -> None:
    case = _case("U1-MEM")
    bundle = CounterfactualRunner().run(case.episode, case.evaluation, case.resources)
    with pytest.raises(TypeError):
        build_training_views(case.episode, bundle)
    views = build_training_views(
        case.episode,
        bundle,
        student_action="NONE",
        student_action_source=StudentActionSource.SCRIPTED_PROBE,
    )
    assert views.student_packet.student_action == "NONE"
    assert views.student_packet.student_action_source == StudentActionSource.SCRIPTED_PROBE
    assert views.teacher_packet.opd_data_status == OpdDataStatus.SCHEMA_PROBE
    assert dict(views.sft_candidate.provenance)["label_source"] == "DETERMINISTIC_COUNTERFACTUAL_ORACLE"
    assert views.grpo_group.to_dict()["rollout_source"] == "COUNTERFACTUAL_ENUMERATION"
    assert "POLICY_SAMPLE" not in str(views.student_packet.to_dict())
    with pytest.raises(ValueError, match="cannot claim POLICY_SAMPLE"):
        build_training_views(
            case.episode,
            bundle,
            student_action="NONE",
            student_action_source=StudentActionSource.POLICY_SAMPLE,
            policy_identity="fake-policy",
            policy_sample_id="fake-sample",
        )
