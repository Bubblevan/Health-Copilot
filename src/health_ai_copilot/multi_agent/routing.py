"""A small, inspectable product router and strict Lead-plan parser."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..routing.jev import JevClient, JevConfig, JevError
from .contracts import LeadPlan, RouteMode, TriageDecision, WorkerRole

_SYNTHETIC_KEY = re.compile(r"SYNKEY-[A-Z0-9]+", re.IGNORECASE)
_MEMORY_CUES = (
    "my history", "patient history", "previous", "prior", "last time", "before",
    "compared", "change", "trend", "timeline", "months", "weeks", "since",
    "上次", "以前", "之前", "最近", "变化", "对比", "病史", "检查记录",
)
_EVIDENCE_CUES = (
    "guideline", "evidence", "research", "medical knowledge", "according to",
    "recommendation", "what is", "what are", "public health", "指南", "证据", "医学知识",
)
_CARE_CUES = (
    "urgent", "emergency", "risk", "should i", "what should", "seek care", "symptom",
    "chest pain", "shortness of breath", "紧急", "风险", "怎么办", "是否就医", "症状",
)


@dataclass(frozen=True)
class RouteDecision:
    mode: RouteMode
    reason: str
    predicted_capabilities: tuple[WorkerRole, ...]
    unique_key_count: int
    triage_decision: TriageDecision | None = None
    thresholds: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "predicted_capabilities": [role.value for role in self.predicted_capabilities],
            "unique_key_count": self.unique_key_count,
            "triage_decision": self.triage_decision.to_dict() if self.triage_decision else None,
            "thresholds": dict(self.thresholds),
        }


class LocalTriageProvider:
    """Deterministic question-only triage used as the Jev fallback and comparator."""

    version = "local-triage-v1"

    def __init__(
        self,
        router: MedicalRouter | None = None,
        *,
        trained_model: Mapping[str, Any] | None = None,
    ) -> None:
        self.router = router or MedicalRouter()
        self.trained_model = dict(trained_model) if trained_model is not None else None
        if self.trained_model is not None:
            self.version = str(self.trained_model.get("version", "local-trained-logreg-v1"))

    async def triage(self, query: str, observable_context: str) -> TriageDecision:
        del observable_context  # The fitted local model uses only user-visible query text.
        if self.trained_model is not None:
            scores = _score_local_triage_model(self.trained_model, query)
            return TriageDecision(
                need_patient_context=scores["need_patient_context"],
                need_external_evidence=scores["need_external_evidence"],
                need_care_analysis=scores["need_care_analysis"],
                complexity=scores["complexity"],
                provider="local",
                provider_version=self.version,
            )
        route = self.router.decide(query)
        selected = set(route.predicted_capabilities)
        normalized = query.casefold()
        connectors = sum(normalized.count(token) for token in (
            " and ", " also ", " then ", " as well as ", "并且", "同时", "以及", "然后", "分别",
        ))
        complexity = 0.12
        if connectors or len(_SYNTHETIC_KEY.findall(query.upper())) > 1:
            complexity = 0.52
        if len(selected) >= 2:
            complexity = max(complexity, 0.68)
        if len(selected) >= 3 or (connectors >= 2 and len(selected) >= 2):
            complexity = 0.92
        return TriageDecision(
            need_patient_context=0.82 if WorkerRole.PATIENT_CONTEXT in selected else 0.12,
            need_external_evidence=0.82 if WorkerRole.EVIDENCE in selected else 0.12,
            need_care_analysis=0.82 if WorkerRole.CARE in selected else 0.12,
            complexity=complexity,
            provider="local",
            provider_version=self.version,
        )


_LOCAL_MODEL_TOKEN = re.compile(r"(?u)\b[\w-]+\b")


def _score_local_triage_model(model: Mapping[str, Any], query: str) -> dict[str, float]:
    """Evaluate a small, JSON-serializable TF-IDF/logistic triage model."""
    vocabulary = model.get("vocabulary")
    idf = model.get("idf")
    classifiers = model.get("classifiers")
    if not isinstance(vocabulary, Mapping) or not isinstance(idf, list):
        raise TypeError("LOCAL_TRIAGE_MODEL_INVALID_VECTORIZER")
    if not isinstance(classifiers, Mapping):
        raise TypeError("LOCAL_TRIAGE_MODEL_INVALID_CLASSIFIERS")
    tokens = [token.casefold() for token in _LOCAL_MODEL_TOKEN.findall(query)]
    features = set(tokens)
    features.update(f"{left} {right}" for left, right in pairwise(tokens))
    indices = [int(vocabulary[item]) for item in sorted(features) if item in vocabulary]
    norm = math.sqrt(sum(float(idf[index]) ** 2 for index in indices)) or 1.0
    result: dict[str, float] = {}
    for label in ("need_patient_context", "need_external_evidence", "need_care_analysis", "complexity"):
        row = classifiers.get(label)
        if not isinstance(row, Mapping):
            raise TypeError(f"LOCAL_TRIAGE_MODEL_CLASSIFIER_MISSING:{label}")
        coefficients = row.get("coefficients")
        if not isinstance(coefficients, list) or len(coefficients) != len(idf):
            raise ValueError(f"LOCAL_TRIAGE_MODEL_COEFFICIENT_SHAPE:{label}")
        logit = float(row.get("intercept", 0.0))
        logit += sum(
            (float(idf[index]) / norm) * float(coefficients[index])
            for index in indices
        )
        result[label] = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
    return result


JEV_TRIAGE_QUESTIONS: dict[str, dict[str, Any]] = {
    "need_patient_context": {
        "type": "noul",
        "instructions": (
            "Based only on the user request and the supplied observable context, is longitudinal "
            "patient-specific history needed to answer? Classify workflow need only; do not answer "
            "the medical question."
        ),
        "criteria": {
            "true": "The request requires prior patient records, personal history, or change over time.",
            "false": "The request can be handled without patient-specific longitudinal records.",
        },
    },
    "need_external_evidence": {
        "type": "noul",
        "instructions": (
            "Based only on the user request and supplied observable context, is external medical "
            "evidence needed to answer? This is routing only; do not provide medical advice."
        ),
        "criteria": {
            "true": "The request asks for or depends on medical evidence, guidance, or source-backed facts.",
            "false": "The request can be handled without external medical retrieval.",
        },
    },
    "need_care_analysis": {
        "type": "noul",
        "instructions": (
            "Based only on the user request and supplied observable context, is risk, answerability, "
            "or next-step care analysis needed? Do not diagnose or answer the medical question."
        ),
        "criteria": {
            "true": "The request asks what to do, seeks risk triage, or describes a potentially urgent symptom.",
            "false": "No risk, answerability, or next-step care analysis is requested.",
        },
    },
    "complexity": {
        "type": "score",
        "instructions": (
            "Rate workflow complexity from the request and observable context only. Consider the number "
            "of independent aspects, cross-capability dependencies, temporal comparisons, and synthesis. "
            "Do not answer the medical question."
        ),
        "criteria": [
            "Score 0 / LOW: one atomic aspect, one capability, no dependency or comparison.",
            "Score 1 / MEDIUM: multiple aspects or a comparison, but a short mostly independent workflow.",
            "Score 2 / HIGH: several aspects, cross-capability synthesis, temporal reasoning, or dependencies.",
        ],
    },
}


class JevTriageProvider:
    """One batched Jev routing request with automatic failover to local triage."""

    version = "jev-medical-triage-v1"

    def __init__(
        self,
        client: JevClient | None = None,
        *,
        local_fallback: LocalTriageProvider | None = None,
        dotenv_path: Path | None = None,
    ) -> None:
        self.local_fallback = local_fallback or LocalTriageProvider()
        self.client: JevClient | None = client
        self.configuration_error: str | None = None
        if self.client is None:
            try:
                config = JevConfig.from_env(dotenv_path=dotenv_path) if dotenv_path else None
                self.client = JevClient(config)
            except JevError as exc:
                self.configuration_error = type(exc).__name__

    async def triage(self, query: str, observable_context: str) -> TriageDecision:
        if self.client is None:
            local = await self.local_fallback.triage(query, observable_context)
            return replace(
                local,
                provider="local_fallback",
                fallback_reason=self.configuration_error or "JEV_UNAVAILABLE",
            )
        try:
            result = await self.client.evaluate(
                state={"query": query, "observable_context": observable_context},
                questions=JEV_TRIAGE_QUESTIONS,
            )
            probabilities = {
                name: _parse_noul(result.answers.get(name))
                for name in (
                    "need_patient_context", "need_external_evidence", "need_care_analysis",
                )
            }
            score_answer = result.answers.get("complexity")
            if not isinstance(score_answer, dict) or score_answer.get("type") != "score":
                raise ValueError("JEV_COMPLEXITY_SCORE_INVALID")
            score = score_answer.get("score")
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not 0.0 <= float(score) <= 2.0):
                raise ValueError("JEV_COMPLEXITY_SCORE_OUT_OF_RANGE")
            return TriageDecision(
                **probabilities,
                complexity=float(score) / 2.0,
                provider="jev",
                provider_version=self.version,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=int(result.output_tokens or 0),
                latency_ms=float(result.latency_ms),
                cost_usd=result.cost_usd,
            )
        except Exception as exc:  # noqa: BLE001 - Jev failures degrade to local policy.
            local = await self.local_fallback.triage(query, observable_context)
            reason = type(exc).__name__
            if isinstance(exc, JevError) and str(exc):
                reason = f"{reason}:{str(exc)[:96]}"
            return replace(
                local,
                provider="local_fallback",
                fallback_reason=reason,
            )


def _parse_noul(answer: Any) -> float:
    if not isinstance(answer, Mapping) or answer.get("type") != "noul":
        raise ValueError("JEV_NOUL_ANSWER_INVALID")
    value = answer.get("noul")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("JEV_NOUL_PROBABILITY_INVALID")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("JEV_NOUL_PROBABILITY_OUT_OF_RANGE")
    return float(value)


def route_from_triage(
    triage: TriageDecision,
    *,
    thresholds: dict[str, float] | None = None,
) -> RouteDecision:
    limits = {
        "need_patient_context": 0.5,
        "need_external_evidence": 0.5,
        "need_care_analysis": 0.5,
        "complexity": 0.5,
    }
    limits.update(thresholds or {})
    rows = (
        (triage.need_patient_context, WorkerRole.PATIENT_CONTEXT, "need_patient_context"),
        (triage.need_external_evidence, WorkerRole.EVIDENCE, "need_external_evidence"),
        (triage.need_care_analysis, WorkerRole.CARE, "need_care_analysis"),
    )
    capabilities = tuple(role for probability, role, name in rows
                         if probability >= limits[name])
    reasons = [f"{name}_at_or_above_threshold" for probability, _role, name in rows
               if probability >= limits[name]]
    if not capabilities:
        capabilities = (WorkerRole.EVIDENCE,)
        reasons.append("empty_capability_set_fallback_to_evidence")
    is_low_complexity = triage.complexity < limits["complexity"]
    mode = RouteMode.SINGLE if len(capabilities) == 1 and is_low_complexity else RouteMode.TEAM
    reason = "single_capability_low_complexity_fast_path" if mode == RouteMode.SINGLE else (
        "multi_capability_or_nonlow_complexity_team"
    )
    return RouteDecision(
        mode, reason, capabilities, 0, triage,
        tuple(sorted((name, float(value)) for name, value in limits.items())),
    )


class MedicalRouter:
    """Route simple lookups quickly and delegate multi-source composition."""

    def decide(self, query: str) -> RouteDecision:
        normalized = query.casefold()
        key_count = len(set(_SYNTHETIC_KEY.findall(query.upper())))
        memory = any(cue in normalized for cue in _MEMORY_CUES)
        evidence = any(cue in normalized for cue in _EVIDENCE_CUES)
        care = any(cue in normalized for cue in _CARE_CUES)
        if key_count >= 2:
            memory = evidence = True
        if not memory and not evidence:
            evidence = True
        capabilities = tuple(
            role for active, role in (
                (memory, WorkerRole.PATIENT_CONTEXT),
                (evidence, WorkerRole.EVIDENCE),
                (care, WorkerRole.CARE),
            ) if active
        )
        if len(capabilities) >= 2:
            return RouteDecision(
                RouteMode.TEAM,
                "query combines multiple observable capability cues",
                capabilities,
                key_count,
            )
        reason = "single-source or direct education request"
        if key_count <= 1 and not care:
            reason = "single-key/simple fast path"
        return RouteDecision(RouteMode.SINGLE, reason, capabilities, key_count)


def deterministic_fallback_plan(decision: RouteDecision, query: str) -> LeadPlan:
    """Harness fallback used only when model planning fails or is malformed."""
    roles = decision.predicted_capabilities or (WorkerRole.EVIDENCE,)
    tasks = tuple(
        (role, _objective_for(role, query))
        for role in roles[:3]
    )
    if len(tasks) < 2:
        tasks = (
            (WorkerRole.PATIENT_CONTEXT, _objective_for(WorkerRole.PATIENT_CONTEXT, query)),
            (WorkerRole.EVIDENCE, _objective_for(WorkerRole.EVIDENCE, query)),
        )
    return LeadPlan(RouteMode.TEAM, tasks, source="harness_fallback")


def parse_lead_plan(content: str) -> LeadPlan:
    """Parse the untrusted model plan; IDs, budgets, and status remain harness-owned."""
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("LEAD_PLAN_INVALID_JSON") from exc
    if not isinstance(payload, dict) or set(payload) - {"mode", "tasks"}:
        raise ValueError("LEAD_PLAN_INVALID_SHAPE")
    try:
        mode = RouteMode(str(payload["mode"]).upper())
    except (KeyError, ValueError) as exc:
        raise ValueError("LEAD_PLAN_INVALID_MODE") from exc
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) > 3:
        raise ValueError("LEAD_PLAN_TASK_LIMIT")
    tasks: list[tuple[WorkerRole, str]] = []
    for row in raw_tasks:
        if not isinstance(row, dict) or set(row) != {"worker", "objective"}:
            raise ValueError("LEAD_PLAN_INVALID_TASK")
        try:
            role = WorkerRole(str(row["worker"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("LEAD_PLAN_UNKNOWN_WORKER") from exc
        objective = row["objective"]
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("LEAD_PLAN_EMPTY_OBJECTIVE")
        tasks.append((role, objective.strip()))
    try:
        return LeadPlan(mode, tuple(tasks), source="model")
    except ValueError as exc:
        raise ValueError(f"LEAD_PLAN_INVALID:{exc}") from exc


def _objective_for(role: WorkerRole, query: str) -> str:
    objectives = {
        WorkerRole.PATIENT_CONTEXT: "查找与当前问题相关的患者既往记录及时间变化。",
        WorkerRole.EVIDENCE: "检索与问题相关的医学知识和可追溯外部证据。",
        WorkerRole.CARE: "依据已经观察到的上下文评估风险、可回答性和下一步行动。",
    }
    del query  # Original query is provided separately to each worker.
    return objectives[role]
