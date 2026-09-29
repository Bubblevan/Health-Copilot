"""Counterfactual execution over valid arms of one frozen episode."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import (
    ACTION_BY_KEY,
    AbstractCost,
    ActionAvailability,
    ActionKey,
    CapabilityAction,
    CapabilityEquivalenceReport,
    capability_equivalence_report,
    cost_for,
)
from .contracts import EvaluationPlane, IntegrationEpisode
from .evaluator import DeterministicIntegrationEvaluator, EvaluationResult
from .executor import DeterministicIntegrationExecutor, ExecutionResources, ExecutionResult
from .replay import ReplayIdentity, replay_identity


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
    arms: tuple[CounterfactualArmResult, ...]
    oracle_action_set: MinimalSuccessfulActionSet

    def to_dict(self) -> dict[str, object]:
        return {"episode_id": self.episode_id, "availability": self.availability.to_dict(),
                "single_team_equivalence": self.equivalence.to_dict(),
                "arms": [item.to_dict() for item in self.arms],
                "minimal_successful_action_set": self.oracle_action_set.to_dict()}


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
        availability = ActionAvailability.for_episode(episode)
        equivalence = capability_equivalence_report(episode)
        results: list[CounterfactualArmResult] = []
        workers = episode.tool_surface_ref.workers if episode.tool_surface_ref else ()
        for key in ActionKey:
            if key not in availability.valid:
                continue
            action = ACTION_BY_KEY[key]
            execution = self.executor.execute(episode, action, resources)
            evaluated = self.evaluator.evaluate(
                execution.outcome, evaluation, action,
                observed_evidence_ids=execution.observed_evidence_ids,
            )
            identity = replay_identity(episode, action, resources, self.executor, self.evaluator.version)
            results.append(CounterfactualArmResult(
                key, action, identity, execution, evaluated, cost_for(action, workers)
            ))
        return CounterfactualBundle(episode.episode_id, availability, equivalence,
                                    tuple(results), minimal_successful_action_set(results, self.epsilon))


def minimal_successful_action_set(
    arms: list[CounterfactualArmResult] | tuple[CounterfactualArmResult, ...], epsilon: int = 0
) -> MinimalSuccessfulActionSet:
    successful = [arm for arm in arms if arm.evaluation.outcome.task_success
                  and arm.evaluation.outcome.safety_pass and arm.evaluation.outcome.grounding_pass]
    model = successful[0].cost.model_version if successful else "u1-abstract-cost-v1"
    if not successful:
        return MinimalSuccessfulActionSet((), None, epsilon, model)
    minimum = min(arm.cost.total_units for arm in successful)
    chosen = tuple(arm.action_key for arm in successful if arm.cost.total_units <= minimum + epsilon)
    return MinimalSuccessfulActionSet(chosen, minimum, epsilon, model)
