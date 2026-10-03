"""A small, inspectable product router and strict Lead-plan parser."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .contracts import LeadPlan, RouteMode, WorkerRole

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "predicted_capabilities": [role.value for role in self.predicted_capabilities],
            "unique_key_count": self.unique_key_count,
        }


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
