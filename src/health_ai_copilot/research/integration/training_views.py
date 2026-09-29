"""Schema-only SFT, GRPO, and OPD views with explicit sampling provenance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .contracts import ExecutionOutcome, IntegrationEpisode, ObservableState
from .counterfactual import CounterfactualBundle


class StudentActionSource(StrEnum):
    SCRIPTED_PROBE = "SCRIPTED_PROBE"
    POLICY_SAMPLE = "POLICY_SAMPLE"


class OpdDataStatus(StrEnum):
    SCHEMA_PROBE = "SCHEMA_PROBE"
    POLICY_VISITED = "POLICY_VISITED"


class GrpoRolloutSource(StrEnum):
    COUNTERFACTUAL_ENUMERATION = "COUNTERFACTUAL_ENUMERATION"
    POLICY_SAMPLE = "POLICY_SAMPLE"


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
    reward_version: str = "u1.1-success-safety-cost-v2"
    rollout_source: GrpoRolloutSource = GrpoRolloutSource.COUNTERFACTUAL_ENUMERATION

    def to_dict(self) -> dict[str, object]:
        return {"episode_id": self.episode_id,
                "actions": [item.action for item in self.samples],
                "outcomes": [item.outcome.to_dict() for item in self.samples],
                "group_rewards": list(self.group_rewards),
                "reward_version": self.reward_version,
                "rollout_source": self.rollout_source.value}


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
    student_action_source: StudentActionSource
    student_outcome: StudentVisibleOutcome
    provenance: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {"episode_id": self.episode_id,
                "student_visited_state": self.student_visited_state.to_dict(),
                "student_action": self.student_action,
                "student_action_source": self.student_action_source.value,
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
    arm_failure_categories: tuple[str, ...]
    bundle_counterfactual_attributions: tuple[str, ...]
    opd_data_status: OpdDataStatus
    privileged_plane: str = "PRIVILEGED_TRAINING"

    def to_dict(self) -> dict[str, object]:
        return {"episode_id": self.episode_id, "privileged_plane": self.privileged_plane,
                "opd_data_status": self.opd_data_status.value,
                "student_packet": self.student_packet.to_dict(),
                "counterfactual_action_outcomes": [item.to_dict() for item in self.counterfactual_action_outcomes],
                "minimal_successful_action_set": list(self.minimal_successful_action_set),
                "arm_failure_categories": list(self.arm_failure_categories),
                "bundle_counterfactual_attributions": list(self.bundle_counterfactual_attributions)}


@dataclass(frozen=True)
class TrainingViews:
    sft_candidate: SFTCandidate
    grpo_group: PolicyRolloutGroup
    student_packet: StudentPacket
    teacher_packet: PrivilegedTeacherPacket


def build_training_views(
    episode: IntegrationEpisode,
    bundle: CounterfactualBundle,
    *,
    student_action: str,
    student_action_source: StudentActionSource | str,
    policy_identity: str | None = None,
    policy_sample_id: str | None = None,
) -> TrainingViews:
    source = StudentActionSource(student_action_source)
    if (source == StudentActionSource.POLICY_SAMPLE
            and episode.source_provenance.startswith("SYNTHETIC_CONTRACT_FIXTURE")):
        raise ValueError("synthetic contract fixtures cannot claim POLICY_SAMPLE provenance")
    if source == StudentActionSource.POLICY_SAMPLE and not (policy_identity and policy_sample_id):
        raise ValueError("POLICY_SAMPLE requires explicit policy_identity and policy_sample_id")
    if source == StudentActionSource.SCRIPTED_PROBE and (policy_identity or policy_sample_id):
        raise ValueError("SCRIPTED_PROBE cannot carry policy sample identity")
    if episode.episode_id != bundle.episode_id:
        raise ValueError("counterfactual bundle episode_id mismatch")
    selected = next((item for item in bundle.arms if item.action_key.value == student_action), None)
    if selected is None or student_action not in {item.value for item in bundle.availability.valid}:
        raise ValueError("student_action must be a valid evaluated counterfactual arm")

    first = bundle.arms[0] if bundle.arms else None
    arm_actions = ",".join(item.action_key.value for item in bundle.arms)
    provenance = (
        ("episode_hash", episode.episode_hash),
        ("schema", "u1.1-training-views-v2"),
        ("label_source", "DETERMINISTIC_COUNTERFACTUAL_ORACLE"),
        ("execution_backend", first.identity.executor_version if first else "NONE"),
        ("evaluator", first.identity.evaluator_version if first else "NONE"),
        ("cost_model", bundle.oracle_action_set.cost_model_version),
        ("epsilon", str(bundle.oracle_action_set.epsilon)),
        ("evaluated_arms", arm_actions),
    )
    sft = SFTCandidate(
        episode.observable_state,
        tuple(item.value for item in bundle.oracle_action_set.actions),
        provenance,
    )
    samples: list[PolicyRolloutSample] = []
    for arm in bundle.arms:
        outcome = arm.evaluation.outcome
        reward = float(int(outcome.task_success and outcome.safety_pass and outcome.grounding_pass))
        reward -= arm.cost.total_units * 0.01
        samples.append(PolicyRolloutSample(arm.action_key.value, outcome, arm.cost.total_units, reward))
    group = PolicyRolloutGroup(
        episode.episode_id, tuple(samples), tuple(item.reward for item in samples),
        rollout_source=GrpoRolloutSource.COUNTERFACTUAL_ENUMERATION,
    )
    student_provenance = [("episode_hash", episode.episode_hash), ("schema", "u1.1-student-packet-v2")]
    if source == StudentActionSource.POLICY_SAMPLE:
        student_provenance.extend((
            ("policy_identity", policy_identity or ""),
            ("policy_sample_id", policy_sample_id or ""),
        ))
    student = StudentPacket(
        episode_id=episode.episode_id,
        student_visited_state=episode.observable_state,
        student_action=selected.action_key.value,
        student_action_source=source,
        student_outcome=StudentVisibleOutcome.from_execution(selected.execution.outcome),
        provenance=tuple(student_provenance),
    )
    teacher_outcomes = tuple(TeacherCounterfactualOutcome(
        item.action_key.value, item.evaluation.outcome.task_success,
        item.evaluation.outcome.safety_pass, item.evaluation.outcome.grounding_pass,
        item.cost.total_units,
    ) for item in bundle.arms)
    failures = tuple(sorted({item.evaluation.outcome.failure_category.value
                             for item in bundle.arms if item.evaluation.outcome.failure_category}))
    bundle_attributions = tuple(
        f"{item.category.value}:{item.single_action}->{item.team_action}"
        for item in bundle.attributions
    )
    status = (OpdDataStatus.POLICY_VISITED if source == StudentActionSource.POLICY_SAMPLE
              else OpdDataStatus.SCHEMA_PROBE)
    teacher = PrivilegedTeacherPacket(
        episode.episode_id, student, teacher_outcomes,
        tuple(item.value for item in bundle.oracle_action_set.actions), failures,
        bundle_attributions, status,
    )
    return TrainingViews(sft, group, student, teacher)
