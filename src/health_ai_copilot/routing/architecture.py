"""Research-only Jev architecture and evidence-task intent routing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from .jev import JevClient, JevResult
from .task_intent import TaskIntentAssessment, parse_task_intent, task_intent_questions


class Architecture(StrEnum):
    SINGLE = "single"
    PARALLEL_TEAM = "parallel_team"


class WorkerRole(StrEnum):
    WHO = "who_global"
    CDC = "cdc_us"
    NHC = "nhc_china"
    LITERATURE = "literature"


WORKER_FAMILY: dict[WorkerRole, str] = {
    WorkerRole.WHO: "WHO",
    WorkerRole.CDC: "CDC",
    WorkerRole.NHC: "NHC",
    WorkerRole.LITERATURE: "LITERATURE",
}


@dataclass(frozen=True)
class ArchitectureDecision:
    architecture: Architecture
    worker_roles: tuple[WorkerRole, ...]
    architecture_probabilities: Mapping[str, float]
    worker_probabilities: Mapping[str, float]
    confidence: float
    latency_ms: int
    model: str
    input_tokens: int
    output_tokens: int | None
    fallback_reason: str | None = None
    task_intent: TaskIntentAssessment | None = None
    route_policy_version: str | None = None
    route_reasons: tuple[str, ...] = ()

    def metadata(self) -> dict[str, Any]:
        result = {
            "architecture": self.architecture.value,
            "worker_roles": [role.value for role in self.worker_roles],
            "architecture_probabilities": dict(self.architecture_probabilities),
            "worker_probabilities": dict(self.worker_probabilities),
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "fallback_reason": self.fallback_reason,
        }
        if self.task_intent is not None:
            result.update(self.task_intent.metadata())
        if self.route_policy_version is not None:
            result["route_policy_version"] = self.route_policy_version
            result["route_reasons"] = list(self.route_reasons)
        return result


class JevArchitectureRouter:
    """Use Jev probabilities for routing; keep execution policy in Python."""

    version = "jev-architecture-router-v1"

    def __init__(
        self,
        client: JevClient | None = None,
        *,
        team_threshold: float = 0.5,
        worker_threshold: float = 0.5,
        fallback_on_error: bool = True,
    ) -> None:
        _validate_threshold("team_threshold", team_threshold)
        _validate_threshold("worker_threshold", worker_threshold)
        self.client = client or JevClient()
        self.team_threshold = team_threshold
        self.worker_threshold = worker_threshold
        self.fallback_on_error = fallback_on_error

    async def route(self, *, question: str, data_classification: str) -> ArchitectureDecision:
        _require_research_safe_classification(data_classification)
        if not question.strip():
            raise ValueError("question must be non-empty")
        try:
            result = await self.client.evaluate(
                state={"question": question},
                questions=_architecture_questions(),
            )
            return self._decision(result)
        except Exception as exc:
            if not self.fallback_on_error:
                raise
            return self._fallback_decision(type(exc).__name__)

    @staticmethod
    def _fallback_decision(reason: str) -> ArchitectureDecision:
        return ArchitectureDecision(
            architecture=Architecture.PARALLEL_TEAM,
            worker_roles=tuple(WorkerRole),
            architecture_probabilities={},
            worker_probabilities={},
            confidence=0.0,
            latency_ms=0,
            model="fallback-team",
            input_tokens=0,
            output_tokens=0,
            fallback_reason=reason,
        )

    def _decision(self, result: JevResult) -> ArchitectureDecision:
        architecture_answer = _typed_answer(result, "architecture", "choice")
        route_probabilities = _probabilities(architecture_answer.get("probabilities"), ("single", "parallel_team"))
        team_probability = route_probabilities[Architecture.PARALLEL_TEAM.value]
        architecture = (
            Architecture.PARALLEL_TEAM
            if team_probability >= self.team_threshold
            else Architecture.SINGLE
        )
        worker_probabilities = {
            role.value: _noul_probability(_typed_answer(result, role.value, "noul"))
            for role in WorkerRole
        }
        roles: tuple[WorkerRole, ...] = ()
        if architecture == Architecture.PARALLEL_TEAM:
            chosen = tuple(role for role in WorkerRole if worker_probabilities[role.value] >= self.worker_threshold)
            if chosen:
                roles = chosen
            else:
                # Conservative deterministic tie-break when the team route was chosen
                # but no individual worker crossed the configured threshold.
                roles = (max(WorkerRole, key=lambda role: worker_probabilities[role.value]),)
        confidence = _probability(architecture_answer.get("confidence", max(route_probabilities.values())))
        return ArchitectureDecision(
            architecture=architecture,
            worker_roles=roles,
            architecture_probabilities=route_probabilities,
            worker_probabilities=worker_probabilities,
            confidence=confidence,
            latency_ms=result.latency_ms,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )


class JevTaskIntentRouter(JevArchitectureRouter):
    """Classify evidence-task intent and apply an explicit Python route policy."""

    version = "jev-task-intent-router-v1"
    policy_version = "task-intent-policy-v1"

    async def route(self, *, question: str, data_classification: str) -> ArchitectureDecision:
        _require_research_safe_classification(data_classification)
        if not question.strip():
            raise ValueError("question must be non-empty")
        questions = _architecture_questions()
        questions.update(task_intent_questions())
        try:
            result = await self.client.evaluate(
                state={"question": question},
                questions=questions,
            )
            return self._decision_with_task_intent(result)
        except Exception as exc:
            if not self.fallback_on_error:
                raise
            fallback = self._fallback_decision(type(exc).__name__)
            return replace(
                fallback,
                route_policy_version=self.policy_version,
                route_reasons=("jev_error_team_fallback",),
            )

    def _decision_with_task_intent(self, result: JevResult) -> ArchitectureDecision:
        baseline = self._decision(result)
        intent = parse_task_intent(result)
        facets = intent.facet_probabilities
        route_reasons: list[str] = []
        if baseline.architecture == Architecture.PARALLEL_TEAM:
            route_reasons.append("architecture_probability")
        force_team_facets = (
            "needs_independent_sources",
            "requires_cross_source_comparison",
            "requires_conflict_review",
        )
        route_reasons.extend(
            name for name in force_team_facets if facets[name] >= self.team_threshold
        )
        architecture = (
            Architecture.PARALLEL_TEAM
            if route_reasons
            else Architecture.SINGLE
        )
        roles = list(baseline.worker_roles)
        requires_multiple_workers = any(
            facets[name] >= self.team_threshold
            for name in (
                "needs_independent_sources",
                "requires_cross_source_comparison",
                "requires_conflict_review",
            )
        )
        if (
            architecture == Architecture.PARALLEL_TEAM
            and requires_multiple_workers
            and len(roles) < 2
        ):
            role_order = {role: index for index, role in enumerate(WorkerRole)}
            ranked = sorted(
                WorkerRole,
                key=lambda role: (-baseline.worker_probabilities[role.value], role_order[role]),
            )
            for role in ranked:
                if role not in roles:
                    roles.append(role)
                if len(roles) >= 2:
                    break
        return replace(
            baseline,
            architecture=architecture,
            worker_roles=tuple(roles),
            task_intent=intent,
            route_policy_version=self.policy_version,
            route_reasons=tuple(route_reasons),
        )


def _architecture_questions() -> dict[str, dict[str, Any]]:
    roles = {
        WorkerRole.WHO: "WHO global guidance",
        WorkerRole.CDC: "CDC or United States public-health guidance",
        WorkerRole.NHC: "China National Health Commission guidance",
        WorkerRole.LITERATURE: "independent scientific literature beyond public-health guidance",
    }
    questions: dict[str, dict[str, Any]] = {
        "architecture": {
            "type": "choice",
            "instructions": (
                "Choose whether this request benefits from parallel research by multiple "
                "source-specialized workers. Decide workflow only. Do not answer the medical question."
            ),
            "criteria": {
                "single": "One concise answer or one source family is likely enough.",
                "parallel_team": "The request needs independent sources, multiple jurisdictions, "
                "comparison, conflicting guidance, or several distinct evidence strands.",
            },
        }
    }
    for role, description in roles.items():
        questions[role.value] = {
            "type": "noul",
            "instructions": f"Should the research team include a worker for {description}?",
            "criteria": {
                "true": f"This source family is relevant to the question: {description}.",
                "false": f"This source family is not needed for the question: {description}.",
            },
        }
    return questions


def _typed_answer(result: JevResult, name: str, expected_type: str) -> Mapping[str, Any]:
    answer = result.answers.get(name)
    if not isinstance(answer, Mapping) or answer.get("type") != expected_type:
        raise ValueError(f"Jev returned an invalid {expected_type} answer for {name}")
    return answer


def _probabilities(value: Any, expected: tuple[str, ...]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("Jev architecture probabilities must be an object")
    result = {name: _probability(value.get(name)) for name in expected}
    if sum(result.values()) <= 0:
        raise ValueError("Jev returned empty architecture probabilities")
    return result


def _noul_probability(answer: Mapping[str, Any]) -> float:
    if "noul" not in answer:
        raise ValueError("Jev did not return a noul probability")
    return _probability(answer["noul"])


def _probability(value: Any) -> float:
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("Jev probability must be between 0 and 1")
    return float(value)


def _validate_threshold(name: str, threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _require_research_safe_classification(value: str) -> None:
    if value not in {"public", "synthetic"}:
        raise ValueError("Jev adapters accept only explicitly classified public or synthetic data")


__all__ = [
    "WORKER_FAMILY",
    "Architecture",
    "ArchitectureDecision",
    "JevArchitectureRouter",
    "JevTaskIntentRouter",
    "WorkerRole",
]
