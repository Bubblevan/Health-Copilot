"""Counterfactual execution with arm outcomes and bundle-level attribution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .actions import (
    ACTION_BY_KEY,
    AbstractCost,
    ActionAvailability,
    ActionKey,
    ActivationCost,
    CapabilityAction,
    CapabilityEquivalenceReport,
    ExecutableCapabilityEquivalenceReport,
    ObservedUsage,
    capability_equivalence_report,
    executable_capability_equivalence_report,
)
from .contracts import EvaluationPlane, IntegrationEpisode
from .evaluator import DeterministicIntegrationEvaluator, EvaluationResult
from .executor import DeterministicIntegrationExecutor, ExecutionResources, ExecutionResult
from .replay import ReplayIdentity, replay_identity


class BundleAttributionCategory(StrEnum):
    ORCHESTRATION_GAIN_CANDIDATE = "ORCHESTRATION_GAIN_CANDIDATE"
    UNNECESSARY_TEAM = "UNNECESSARY_TEAM"
    TEAM_FAILURE = "TEAM_FAILURE"


@dataclass(frozen=True)
class BundleCounterfactualAttribution:
    category: BundleAttributionCategory
    single_action: str
    team_action: str
    single_success: bool
    team_success: bool
    single_cost_units: int
    team_cost_units: int
    capability_parity_verified: bool

    def to_dict(self) -> dict[str, object]:
        return {"category": self.category.value,
                "single_action": self.single_action,
                "team_action": self.team_action,
                "single_success": self.single_success,
                "team_success": self.team_success,
                "single_cost_units": self.single_cost_units,
                "team_cost_units": self.team_cost_units,
                "capability_parity_verified": self.capability_parity_verified}


@dataclass(frozen=True)
class CounterfactualArmResult:
    action_key: ActionKey
    action: CapabilityAction
    identity: ReplayIdentity
    execution: ExecutionResult
    evaluation: EvaluationResult
    cost: AbstractCost

    def to_dict(self) -> dict[str, object]:
        return {"action_key": self.action_key.value, "action": self.action.to_dict(),
                "replay_identity": self.identity.to_dict(),
                "execution": self.execution.to_dict(),
                "evaluation": self.evaluation.to_dict(), "cost": self.cost.to_dict()}


@dataclass(frozen=True)
class MinimalSuccessfulActionSet:
    actions: tuple[ActionKey, ...]
    minimum_cost: int | None
    epsilon: int
    cost_model_version: str

    def to_dict(self) -> dict[str, object]:
        return {"actions": [item.value for item in self.actions],
                "minimum_cost": self.minimum_cost, "epsilon": self.epsilon,
                "cost_model_version": self.cost_model_version}


@dataclass(frozen=True)
class CounterfactualBundle:
    episode_id: str
    availability: ActionAvailability
    equivalence: CapabilityEquivalenceReport
    executable_equivalence: ExecutableCapabilityEquivalenceReport
    arms: tuple[CounterfactualArmResult, ...]
    oracle_action_set: MinimalSuccessfulActionSet
    attributions: tuple[BundleCounterfactualAttribution, ...]

    def to_dict(self) -> dict[str, object]:
        return {"episode_id": self.episode_id, "availability": self.availability.to_dict(),
                "single_team_equivalence": self.equivalence.to_dict(),
                "executable_capability_equivalence": self.executable_equivalence.to_dict(),
                "arms": [item.to_dict() for item in self.arms],
                "minimal_successful_action_set": self.oracle_action_set.to_dict(),
                "bundle_attributions": [item.to_dict() for item in self.attributions]}


class CounterfactualRunner:
    def __init__(self, *, epsilon: int = 0) -> None:
        if epsilon < 0:
            raise ValueError("cost epsilon must be non-negative")
        self.epsilon = epsilon
        self.executor = DeterministicIntegrationExecutor()
        self.evaluator = DeterministicIntegrationEvaluator()

    def run(
        self,
        episode: IntegrationEpisode,
        evaluation: EvaluationPlane,
        resources: ExecutionResources,
    ) -> CounterfactualBundle:
        if evaluation.episode_id != episode.episode_id:
            raise ValueError("evaluation plane episode_id mismatch")
        availability = ActionAvailability.for_episode(episode, self.executor.tool_registry)
        equivalence = capability_equivalence_report(episode)
        executable_equivalence = executable_capability_equivalence_report(
            episode, registry=self.executor.tool_registry
        )
        results: list[CounterfactualArmResult] = []
        for key in ActionKey:
            if key not in availability.valid:
                continue
            action = ACTION_BY_KEY[key]
            execution = self.executor.execute(episode, action, resources)
            evaluated = self.evaluator.evaluate(
                execution.outcome, evaluation,
                observed_evidence_ids=execution.observed_evidence_ids,
            )
            identity = replay_identity(episode, action, resources, self.executor, self.evaluator.version)
            arm_cost = execution.cost or AbstractCost(
                ActivationCost(), ObservedUsage(), episode.budget.cost_model_version
            )
            results.append(CounterfactualArmResult(
                key, action, identity, execution, evaluated, arm_cost
            ))
        action_set = minimal_successful_action_set(results, self.epsilon)
        attributions = attribute_bundle_outcomes(results, executable_equivalence.equivalent)
        return CounterfactualBundle(
            episode.episode_id,
            availability,
            equivalence,
            executable_equivalence,
            tuple(results),
            action_set,
            attributions,
        )


def _succeeded(arm: CounterfactualArmResult) -> bool:
    outcome = arm.evaluation.outcome
    return outcome.task_success and outcome.safety_pass and outcome.grounding_pass


def attribute_bundle_outcomes(
    arms: list[CounterfactualArmResult] | tuple[CounterfactualArmResult, ...],
    capability_parity_verified: bool,
) -> tuple[BundleCounterfactualAttribution, ...]:
    by_action = {arm.action_key: arm for arm in arms}
    pairings = (
        (ActionKey.NONE, ActionKey.TEAM),
        (ActionKey.MEMORY, ActionKey.MEMORY_TEAM),
        (ActionKey.RAG, ActionKey.RAG_TEAM),
        (ActionKey.MEMORY_RAG, ActionKey.ALL),
    )
    results = []
    for single_key, team_key in pairings:
        single = by_action.get(single_key)
        team = by_action.get(team_key)
        if single is None or team is None:
            continue
        single_success = _succeeded(single)
        team_success = _succeeded(team)
        category = None
        if not capability_parity_verified:
            continue
        if not single_success and team_success:
            category = BundleAttributionCategory.ORCHESTRATION_GAIN_CANDIDATE
        elif single_success and team_success and team.cost.total_units > single.cost.total_units:
            category = BundleAttributionCategory.UNNECESSARY_TEAM
        elif single_success and not team_success:
            category = BundleAttributionCategory.TEAM_FAILURE
        if category is not None:
            results.append(BundleCounterfactualAttribution(
                category, single_key.value, team_key.value,
                single_success, team_success,
                single.cost.total_units, team.cost.total_units,
                capability_parity_verified,
            ))
    return tuple(results)


def minimal_successful_action_set(
    arms: list[CounterfactualArmResult] | tuple[CounterfactualArmResult, ...], epsilon: int = 0
) -> MinimalSuccessfulActionSet:
    successful = [arm for arm in arms if _succeeded(arm)]
    model = successful[0].cost.model_version if successful else "u1.1-activation-observed-v1"
    if not successful:
        return MinimalSuccessfulActionSet((), None, epsilon, model)
    minimum = min(arm.cost.total_units for arm in successful)
    chosen = tuple(arm.action_key for arm in successful if arm.cost.total_units <= minimum + epsilon)
    return MinimalSuccessfulActionSet(chosen, minimum, epsilon, model)
