"""Research-only Jev gate for choosing Single or a heterogeneous team."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from .jev import JevClient, JevResult


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
    output_tokens: int
    fallback_reason: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {
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
                fallback_reason=type(exc).__name__,
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
        raise ValueError("Jev did not return architecture probabilities")
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
    "Architecture",
    "ArchitectureDecision",
    "JevArchitectureRouter",
    "WORKER_FAMILY",
    "WorkerRole",
]
