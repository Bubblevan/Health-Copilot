"""Schema-only SFT, GRPO, and OPD compatibility artifacts; no model training."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ExecutionOutcome, IntegrationEpisode, ObservableState
from .counterfactual import CounterfactualBundle


@dataclass(frozen=True)
class SFTCandidate:
    observable_state: ObservableState
    acceptable_actions: tuple[str, ...]
    provenance: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {"observable_state": self.observable_state.to_dict(),
                "acceptable_actions": list(self.acceptable_actions),
                "provenance": dict(self.provenance)}


@dataclass(frozen=True)
class PolicyRolloutSample:
    action: str
    outcome: ExecutionOutcome
    cost_units: int
    reward: float

    def to_dict(self) -> dict[str, object]:
        return {"action": self.action, "outcome": self.outcome.to_dict(),
                "cost_units": self.cost_units, "reward": self.reward}


@dataclass(frozen=True)
class PolicyRolloutGroup:
    episode_id: str
    samples: tuple[PolicyRolloutSample, ...]
    group_rewards: tuple[float, ...]
    reward_version: str = "u1-success-safety-cost-v1"

    def to_dict(self) -> dict[str, object]:
        return {"episode_id": self.episode_id,
                "sampled_actions": [item.action for item in self.samples],
                "outcomes": [item.outcome.to_dict() for item in self.samples],
                "group_rewards": list(self.group_rewards),
                "reward_version": self.reward_version}


@dataclass(frozen=True)
class StudentVisibleOutcome:
    answer: str
    activated_capabilities: tuple[str, ...]
    provider_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int

    @classmethod
    def from_execution(cls, outcome: ExecutionOutcome) -> StudentVisibleOutcome:
        return cls(outcome.answer, outcome.activated_capabilities, outcome.provider_calls,
                   outcome.tool_calls, outcome.input_tokens, outcome.output_tokens)

    def to_dict(self) -> dict[str, object]:
        return {"answer": self.answer, "activated_capabilities": list(self.activated_capabilities),
                "provider_calls": self.provider_calls, "tool_calls": self.tool_calls,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


@dataclass(frozen=True)
class StudentPacket:
    episode_id: str
    student_visited_state: ObservableState
    student_action: str
    student_outcome: StudentVisibleOutcome
    provenance: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {"episode_id": self.episode_id,
                "student_visited_state": self.student_visited_state.to_dict(),
                "student_action": self.student_action,
                "student_outcome": self.student_outcome.to_dict(),
                "provenance": dict(self.provenance)}


@dataclass(frozen=True)
class TeacherCounterfactualOutcome:
    action: str
    task_success: bool
    safety_pass: bool
    grounding_pass: bool
    cost_units: int

    def to_dict(self) -> dict[str, object]:
        return {"action": self.action, "task_success": self.task_success,
                "safety_pass": self.safety_pass, "grounding_pass": self.grounding_pass,
                "cost_units": self.cost_units}


@dataclass(frozen=True)
class PrivilegedTeacherPacket:
    episode_id: str
    student_packet: StudentPacket
    counterfactual_action_outcomes: tuple[TeacherCounterfactualOutcome, ...]
    minimal_successful_action_set: tuple[str, ...]
    failure_attribution: tuple[str, ...]
    privileged_plane: str = "PRIVILEGED_TRAINING"

    def to_dict(self) -> dict[str, object]:
        return {"episode_id": self.episode_id, "privileged_plane": self.privileged_plane,
                "student_packet": self.student_packet.to_dict(),
                "counterfactual_action_outcomes": [item.to_dict() for item in self.counterfactual_action_outcomes],
                "minimal_successful_action_set": list(self.minimal_successful_action_set),
                "failure_attribution": list(self.failure_attribution)}


@dataclass(frozen=True)
class TrainingViews:
    sft_candidate: SFTCandidate
    grpo_group: PolicyRolloutGroup
    student_packet: StudentPacket
    teacher_packet: PrivilegedTeacherPacket


def build_training_views(episode: IntegrationEpisode, bundle: CounterfactualBundle) -> TrainingViews:
    accepted = tuple(item.value for item in bundle.oracle_action_set.actions)
    provenance = (("episode_hash", episode.episode_hash), ("schema", "u1-training-views-v1"))
    sft = SFTCandidate(episode.observable_state, accepted, provenance)
    samples: list[PolicyRolloutSample] = []
    for arm in bundle.arms:
        outcome = arm.evaluation.outcome
        reward = float(int(outcome.task_success and outcome.safety_pass and outcome.grounding_pass))
        reward -= arm.cost.total_units * 0.01
        samples.append(PolicyRolloutSample(arm.action_key.value, outcome, arm.cost.total_units, reward))
    group = PolicyRolloutGroup(episode.episode_id, tuple(samples), tuple(item.reward for item in samples))
    student_arm = next((item for item in bundle.arms if item.action_key.value == "NONE"), bundle.arms[0])
    student = StudentPacket(
        episode.episode_id, episode.observable_state, student_arm.action_key.value,
        StudentVisibleOutcome.from_execution(student_arm.evaluation.outcome), provenance,
    )
    teacher_outcomes = tuple(TeacherCounterfactualOutcome(
        item.action_key.value, item.evaluation.outcome.task_success,
        item.evaluation.outcome.safety_pass, item.evaluation.outcome.grounding_pass,
        item.cost.total_units,
    ) for item in bundle.arms)
    failures = tuple(sorted({item.evaluation.outcome.failure_category.value
                             for item in bundle.arms if item.evaluation.outcome.failure_category}))
    teacher = PrivilegedTeacherPacket(
        episode.episode_id, student, teacher_outcomes,
        accepted, failures,
    )
    return TrainingViews(sft, group, student, teacher)
