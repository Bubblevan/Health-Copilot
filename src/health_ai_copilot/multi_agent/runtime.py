"""Routed Single/Team execution with harness-owned budgets and provenance."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from threading import RLock
from time import monotonic
from typing import Any
from uuid import uuid4

from ..research.integration.contracts import IntegrationEpisode
from .contracts import (
    Citation,
    LeadPlan,
    MedicalAgentRequest,
    MedicalAgentResponse,
    RouteMode,
    WorkerReport,
    WorkerRole,
    WorkerStatus,
)
from .providers import ModelProvider, ModelReply
from .routing import (
    MedicalRouter,
    RouteDecision,
    deterministic_fallback_plan,
    parse_lead_plan,
)
from .shared_context import SharedContext
from .skills import SkillContext, SkillRegistry, SkillResult

_VALUE = re.compile(r"SYNVAL-[0-9A-F]{10}", re.IGNORECASE)
_URGENT_MARKERS = (
    "chest pain", "shortness of breath", "stroke", "unconscious",
    "呼吸困难", "胸痛", "意识丧失", "大出血",
)
_ROLE_PROMPTS = {
    WorkerRole.PATIENT_CONTEXT: (
        "你是 PatientContextAgent。只负责患者纵向状态、时间顺序、变化和个人上下文。"
        "只使用 Harness 提供的患者状态观察；不要虚构病史、来源或证据 ID。"
        "问题要求合成 SYNVAL token 时，逐字保留观察到的相关 token。"
    ),
    WorkerRole.EVIDENCE: (
        "你是 EvidenceAgent。只负责检索到的医学知识与来源支持的事实。"
        "只使用 Harness 提供的观察，不要补入未观察到的医学事实、来源或证据 ID。"
        "问题要求合成 SYNVAL token 时，逐字保留观察到的相关 token。"
    ),
    WorkerRole.CARE: (
        "你是 CareAgent。只负责风险、可回答性和下一步行动；不得创造患者历史或医学证据。"
        "把可能紧急的症状提示线下急救，不做诊断、处方或用药调整。"
    ),
}
_SINGLE_PROMPT = (
    "你是 Health-Copilot 的 Strong Single Agent。你可使用全部患者状态、时间线、"
    "医学检索、外部证据、风险与可回答性能力。只能依据当前问题和 Harness 实际观察；"
    "不得虚构病史、证据、引文或来源 ID。若合成研究问题中要求精确 token，逐字保留。"
    "不要给出诊断或药物剂量调整。只返回面向用户的直接答案，不展示推理过程。"
)
_LEAD_PLAN_PROMPT = (
    "你是 Medical Lead。只做任务拆解和 worker 分配。选择能带来互补信息的 specialist；"
    "简单问题应返回 single 且 tasks 为空。复杂问题可用 patient_context、evidence、care。"
    "每个角色最多一个任务，总任务不超过 3 个，不要重复同一子问题。"
    '只返回 JSON：{"mode":"single|team","tasks":[{"worker":"patient_context|evidence|care",'
    '"objective":"..."}]}。不要生成任何 ID、预算、source_id 或 evidence_id。'
)
_LEAD_FINAL_PROMPT = (
    "你是 Medical Lead。综合已完成 WorkerReport 与 Evidence Ledger，直接回答用户。"
    "上下文里的 worker status 是权威状态；failed/timed_out worker 的内容不存在，"
    "不能把失败 worker 当成空字符串或当成完成。只引用观察到的事实，不得生成 source/evidence ID。"
    "如果问题或报告包含合成 SYNVAL token，答案必须逐字保留所有相关 token。"
    "若证据不够，明确说明局限；有紧急症状时提示及时线下急救。不要诊断、处方或调整药量。"
    "只输出给用户的答案，不展示推理过程。"
)


@dataclass(frozen=True)
class RuntimeConfig:
    max_model_turns_per_worker: int = 3
    max_tool_calls_per_worker: int = 2
    max_worker_calls: int = 3
    worker_timeout_seconds: float = 120.0
    planning_timeout_seconds: float = 60.0
    synthesis_timeout_seconds: float = 120.0
    max_worker_output_tokens: int = 128
    max_lead_output_tokens: int = 192
    max_single_output_tokens: int = 192

    def __post_init__(self) -> None:
        if self.max_model_turns_per_worker > 3 or self.max_model_turns_per_worker < 1:
            raise ValueError("worker model turns must be between one and three")
        if self.max_tool_calls_per_worker > 2 or self.max_tool_calls_per_worker < 0:
            raise ValueError("worker tool calls must be between zero and two")
        if self.max_worker_calls > 3 or self.max_worker_calls < 1:
            raise ValueError("total worker calls must be between one and three")


@dataclass(frozen=True)
class RuntimeExecution:
    response: MedicalAgentResponse
    route_decision: RouteDecision
    plan: LeadPlan | None
    worker_reports: tuple[WorkerReport, ...]
    trace: dict[str, Any]
    provider_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    worker_wave_wall_ms: float
    sequential_worker_latency_ms: float
    partial_failure_recovered: bool
    trajectory: dict[str, Any] = field(default_factory=dict)


class AggregateObservability:
    """Process-local aggregate counters suitable for a lightweight API endpoint."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._rows: list[dict[str, Any]] = []

    def record(self, execution: RuntimeExecution) -> None:
        with self._lock:
            self._rows.append({
                "route_mode": execution.response.route_mode.value,
                "latency_ms": execution.response.latency_ms,
                "worker_count": len(execution.response.workers_used),
                "provider_calls": execution.provider_calls,
                "tool_calls": execution.tool_calls,
                "tokens": execution.input_tokens + execution.output_tokens,
                "error": any(report.status in {WorkerStatus.FAILED, WorkerStatus.TIMED_OUT}
                              for report in execution.worker_reports),
            })

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self._rows)
        latencies = sorted(float(row["latency_ms"]) for row in rows)
        count = len(rows)

        def percentile(p: float) -> float:
            if not latencies:
                return 0.0
            index = min(len(latencies) - 1, max(0, int((len(latencies) - 1) * p)))
            return round(latencies[index], 3)

        team = sum(row["route_mode"] == RouteMode.TEAM.value for row in rows)
        return {
            "requests": count,
            "single_requests": count - team,
            "team_requests": team,
            "team_ratio": round(team / count, 6) if count else 0.0,
            "avg_workers": round(sum(row["worker_count"] for row in rows) / count, 4) if count else 0.0,
            "avg_latency_ms": round(sum(latencies) / count, 3) if count else 0.0,
            "p50_latency_ms": percentile(0.50),
            "p95_latency_ms": percentile(0.95),
            "provider_calls": sum(row["provider_calls"] for row in rows),
            "tool_calls": sum(row["tool_calls"] for row in rows),
            "tokens": sum(row["tokens"] for row in rows),
            "error_rate": round(sum(bool(row["error"]) for row in rows) / count, 6) if count else 0.0,
        }


class MedicalAgentRuntime:
    """The API-facing coordinator; LLM content cannot mutate harness state."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        router: MedicalRouter | None = None,
        skill_registry: SkillRegistry | None = None,
        config: RuntimeConfig | None = None,
        observability: AggregateObservability | None = None,
    ) -> None:
        self.provider = provider
        self.router = router or MedicalRouter()
        self.skills = skill_registry or SkillRegistry()
        self.config = config or RuntimeConfig()
        self.observability = observability or AggregateObservability()

    async def respond(self, request: MedicalAgentRequest) -> MedicalAgentResponse:
        return (await self.execute(request)).response

    async def execute(self, request: MedicalAgentRequest) -> RuntimeExecution:
        started = monotonic()
        trace_id = request.request_id or f"trace-{uuid4().hex}"
        shared = SharedContext(trace_id, request.query)
        shared._harness_timeline("request_started")
        route = self.router.decide(request.query)
        shared._harness_timeline("route_decision", route=route.to_dict())
        total_provider_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_tool_calls = 0
        plan: LeadPlan | None = None
        reports: list[WorkerReport] = []
        wave_wall_ms = 0.0
        sequential_worker_ms = 0.0
        partial_failure_recovered = False
        trajectory: dict[str, Any] = {
            "query": request.query,
            "conversation_context": list(request.conversation_context),
            "worker_inputs": [],
            "lead_synthesis_input": None,
            "final_answer": None,
        }

        try:
            if route.mode == RouteMode.SINGLE:
                answer, citations, flags, counts, single_input = await self._run_single(
                    request, shared, worker_id=f"single-{uuid4().hex[:12]}",
                )
                total_provider_calls += counts[0]
                total_input_tokens += counts[1]
                total_output_tokens += counts[2]
                total_tool_calls += counts[3]
                trajectory["single_input"] = single_input
                actual_mode = RouteMode.SINGLE
            else:
                plan, plan_reply, plan_input = await self._plan(request, route, shared)
                total_provider_calls += int(plan_input.get("provider_attempted", False))
                if plan_reply:
                    total_input_tokens += plan_reply.input_tokens
                    total_output_tokens += plan_reply.output_tokens
                trajectory["lead_plan_input"] = plan_input
                shared._harness_set_plan(plan)
                if plan.mode == RouteMode.SINGLE:
                    answer, citations, flags, counts, single_input = await self._run_single(
                        request, shared, worker_id=f"single-{uuid4().hex[:12]}",
                    )
                    total_provider_calls += counts[0]
                    total_input_tokens += counts[1]
                    total_output_tokens += counts[2]
                    total_tool_calls += counts[3]
                    trajectory["single_input"] = single_input
                    actual_mode = RouteMode.SINGLE
                else:
                    reports, wave_wall_ms = await self._run_plan(
                        request, plan, shared, trajectory,
                    )
                    total_tool_calls += sum(report.tool_calls for report in reports)
                    total_provider_calls += sum(report.provider_calls for report in reports)
                    total_input_tokens += sum(report.input_tokens for report in reports)
                    total_output_tokens += sum(report.output_tokens for report in reports)
                    sequential_worker_ms = sum(report.latency_ms for report in reports)
                    synthesis, synthesis_reply, synthesis_input = await self._synthesize(
                        request, shared,
                    )
                    total_provider_calls += int(synthesis_input.get("provider_attempted", False))
                    if synthesis_reply:
                        total_input_tokens += synthesis_reply.input_tokens
                        total_output_tokens += synthesis_reply.output_tokens
                    trajectory["lead_synthesis_input"] = synthesis_input
                    answer = synthesis
                    actual_mode = RouteMode.TEAM
                    if not any(
                        report.status in {WorkerStatus.COMPLETE, WorkerStatus.PARTIAL}
                        for report in reports
                    ):
                        partial_failure_recovered = True
                        shared._harness_timeline("team_fallback_single")
                        answer, _, single_flags, counts, single_input = await self._run_single(
                            request, shared, worker_id=f"fallback-single-{uuid4().hex[:12]}",
                        )
                        total_provider_calls += counts[0]
                        total_input_tokens += counts[1]
                        total_output_tokens += counts[2]
                        total_tool_calls += counts[3]
                        flags = set(single_flags)
                        flags.add("team_fallback_single")
                        trajectory["single_fallback_input"] = single_input
                    else:
                        flags = set()
                        partial_failure_recovered = any(
                            report.status in {WorkerStatus.FAILED, WorkerStatus.TIMED_OUT}
                            for report in reports
                        )
                    citations = self._verified_citations(shared)

            flags = set(flags)
            flags.update(self._harness_safety_flags(request.query))
            for flag in sorted(flags):
                shared._harness_risk_flag(flag)
            answer = self._verify_answer(answer, request.query, bool(citations), flags)
            latency_ms = max(0, int((monotonic() - started) * 1000))
            response = MedicalAgentResponse(
                answer=answer,
                route_mode=actual_mode,
                workers_used=tuple(
                    report.role.value for report in reports
                    if report.status in {WorkerStatus.COMPLETE, WorkerStatus.PARTIAL}
                ),
                citations=self._verified_citations(shared),
                safety_flags=tuple(sorted(flags)),
                trace_id=trace_id,
                latency_ms=latency_ms,
            )
            trajectory["final_answer"] = answer
            shared._harness_timeline("verification", passed=True, flag_count=len(flags))
            shared._harness_timeline("request_completed", route_mode=actual_mode.value)
        except Exception as exc:  # noqa: BLE001 - all request-level failures map to a safe response
            shared._harness_timeline("request_error", error=type(exc).__name__)
            flags = set(self._harness_safety_flags(request.query))
            flags.add("runtime_degraded_safe_response")
            response = MedicalAgentResponse(
                answer="当前请求未能可靠完成。请稍后重试；如有紧急症状，请及时联系当地急救服务。",
                route_mode=RouteMode.TEAM if route.mode == RouteMode.TEAM else RouteMode.SINGLE,
                workers_used=tuple(report.role.value for report in reports),
                citations=self._verified_citations(shared),
                safety_flags=tuple(sorted(flags)),
                trace_id=trace_id,
                latency_ms=max(0, int((monotonic() - started) * 1000)),
            )

        execution = RuntimeExecution(
            response=response,
            route_decision=route,
            plan=plan,
            worker_reports=tuple(reports),
            trace=shared.to_dict(),
            provider_calls=total_provider_calls,
            tool_calls=total_tool_calls,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            worker_wave_wall_ms=wave_wall_ms,
            sequential_worker_latency_ms=sequential_worker_ms,
            partial_failure_recovered=partial_failure_recovered,
            trajectory=trajectory,
        )
        self.observability.record(execution)
        return execution

    async def _plan(
        self, request: MedicalAgentRequest, decision: RouteDecision, shared: SharedContext,
    ) -> tuple[LeadPlan, ModelReply | None, dict[str, Any]]:
        user_prompt = json.dumps({
            "query": request.query,
            "conversation_context": list(request.conversation_context),
            "worker_capabilities": {
                role.value: list(self.skills.allowed_for(role)) for role in WorkerRole
            },
            "router_hint": decision.to_dict(),
        }, ensure_ascii=False, separators=(",", ":"))
        trace_id = shared.trace_id
        shared._harness_timeline("lead_plan_started")
        reply: ModelReply | None = None
        call_id = f"provider-{uuid4().hex}"
        try:
            shared._harness_timeline("provider_call_started", call_id=call_id, stage="plan")
            reply = await asyncio.wait_for(
                self.provider.complete(
                    system_prompt=_LEAD_PLAN_PROMPT,
                    user_prompt=user_prompt,
                    max_output_tokens=80,
                    timeout_seconds=self.config.planning_timeout_seconds,
                    json_mode=True,
                ),
                timeout=self.config.planning_timeout_seconds,
            )
            shared._harness_record_call(
                trace_id=trace_id, call_id=call_id, stage="plan", status="complete",
                input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
            )
            plan = parse_lead_plan(reply.content)
            if plan.mode == RouteMode.TEAM and len(plan.tasks) < 2:
                plan = deterministic_fallback_plan(decision, request.query)
            return plan, reply, {
                "system_prompt": _LEAD_PLAN_PROMPT, "user_prompt": user_prompt,
                "provider_attempted": True,
            }
        except Exception as exc:  # noqa: BLE001 - malformed plans and provider errors share fallback
            if reply is None:
                shared._harness_record_call(
                    trace_id=trace_id, call_id=call_id, stage="plan", status="failed",
                    error=type(exc).__name__,
                )
            shared._harness_timeline("lead_plan_fallback", error=type(exc).__name__)
            return (
                deterministic_fallback_plan(decision, request.query),
                reply,
                {"system_prompt": _LEAD_PLAN_PROMPT, "user_prompt": user_prompt,
                 "fallback": type(exc).__name__, "provider_attempted": True},
            )

    async def _run_plan(
        self,
        request: MedicalAgentRequest,
        plan: LeadPlan,
        shared: SharedContext,
        trajectory: dict[str, Any],
    ) -> tuple[list[WorkerReport], float]:
        if len(plan.tasks) > self.config.max_worker_calls:
            raise ValueError("HARNESS_WORKER_CALL_BUDGET_EXCEEDED")
        specialist_roles = tuple(role for role, _ in plan.tasks if role != WorkerRole.CARE)
        task_rows = []
        for role, objective in plan.tasks:
            depends = specialist_roles if role == WorkerRole.CARE else ()
            task_rows.append(shared._harness_register_task(role, objective, depends))
        dependencies_by_role = {task.role: task.depends_on for task in task_rows}
        wave_one = [task for task in task_rows if not task.depends_on]
        wave_two = [task for task in task_rows if task.depends_on]
        reports: list[WorkerReport] = []

        async def run_wave(tasks) -> tuple[list[WorkerReport], float]:
            if not tasks:
                return [], 0.0
            started = monotonic()
            wave_reports = await asyncio.gather(*(
                self._run_worker(
                    request, task, shared,
                    dependencies=tuple(
                        report for report in reports
                        if report.role in dependencies_by_role.get(task.role, ())
                    ),
                    trajectory=trajectory,
                )
                for task in tasks
            ))
            return list(wave_reports), (monotonic() - started) * 1000

        first, first_wall = await run_wave(wave_one)
        reports.extend(first)
        second, second_wall = await run_wave(wave_two)
        reports.extend(second)
        shared._harness_timeline(
            "worker_waves_completed", wave_count=1 + int(bool(wave_two)),
            worker_calls=len(reports), wall_ms=round(first_wall + second_wall, 3),
        )
        return reports, first_wall + second_wall

    async def _run_worker(
        self,
        request: MedicalAgentRequest,
        task,
        shared: SharedContext,
        *,
        dependencies: tuple[WorkerReport, ...],
        trajectory: dict[str, Any],
    ) -> WorkerReport:
        started = monotonic()
        role = task.role
        shared._harness_worker_status(task.worker_id, WorkerStatus.RUNNING.value)
        dependency_text = tuple(
            f"{report.role.value} status={report.status.value}: {report.answer_text or '[no answer]'}"
            for report in dependencies
        )
        skill_names = self._skills_for_worker(role, request.query, request.episode)
        if len(skill_names) > self.config.max_tool_calls_per_worker:
            raise ValueError("HARNESS_WORKER_TOOL_BUDGET_EXCEEDED")
        skill_results: list[SkillResult] = []
        for skill_name in skill_names:
            result = await self.skills.execute(
                role,
                skill_name,
                SkillContext(
                    query=request.query,
                    role=role,
                    episode=request.episode,
                    resources=request.resources,
                    observed_context=dependency_text,
                    patient_id=(request.patient_id or (
                        request.episode.subject_id if request.episode else None
                    )),
                    as_of_time=(request.as_of_time or (
                        request.episode.decision_time if request.episode else None
                    )),
                ),
            )
            skill_results.append(result)
            shared._harness_timeline(
                "skill_call", worker_id=task.worker_id, role=role.value,
                skill=skill_name, success=result.error is None,
            )
            if result.tool_calls:
                tool_call_id = f"tool-{uuid4().hex}"
                shared._harness_timeline(
                    "tool_call", worker_id=task.worker_id, tool_call_id=tool_call_id,
                    tool_id=result.tool_id,
                )
                for source in result.sources:
                    shared._harness_add_evidence(
                        source_id=source.source_id,
                        worker_id=task.worker_id,
                        role=role.value,
                        tool_call_id=tool_call_id,
                        tool_id=result.tool_id or source.tool_id,
                        excerpt=source.excerpt[:1200],
                        input_hash=result.input_hash,
                        output_hash=result.output_hash,
                        resource_versions=result.resource_versions,
                    )

        observations = [
            {"skill": result.skill_name, "status": "error" if result.error else "observed",
             "error": result.error,
             "output": result.output[:5000]}
            for result in skill_results
        ]
        user_payload = {
            "query": request.query,
            "objective": task.objective,
            "conversation_context": list(request.conversation_context),
            "observations": observations,
            "dependency_reports": [
                {"role": report.role.value, "status": report.status.value,
                 "answer_text": report.answer_text, "error": report.error}
                for report in dependencies
            ],
        }
        user_prompt = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
        trajectory.setdefault("worker_inputs", []).append({
            "worker_id": task.worker_id,
            "role": role.value,
            "system_prompt": _ROLE_PROMPTS[role],
            "user_prompt": user_prompt,
        })
        call_count = 0
        input_tokens = 0
        output_tokens = 0
        answer = ""
        error = next((result.error for result in skill_results if result.error), None)
        status = WorkerStatus.COMPLETE
        model_turns = 0
        reply_received = False
        call_id = f"provider-{uuid4().hex}"
        try:
            model_turns = 1
            call_count = 1
            shared._harness_timeline(
                "provider_call_started", call_id=call_id, stage="worker",
                worker_id=task.worker_id, role=role.value,
            )
            reply = await asyncio.wait_for(
                self.provider.complete(
                    system_prompt=_ROLE_PROMPTS[role],
                    user_prompt=user_prompt,
                    max_output_tokens=self.config.max_worker_output_tokens,
                    timeout_seconds=self.config.worker_timeout_seconds,
                ),
                timeout=self.config.worker_timeout_seconds,
            )
            reply_received = True
            input_tokens = reply.input_tokens
            output_tokens = reply.output_tokens
            answer = _strip_nonanswer(reply.content)
            shared._harness_record_call(
                trace_id=shared.trace_id, call_id=call_id, stage="worker", status="complete",
                worker_id=task.worker_id, role=role.value,
                input_tokens=input_tokens, output_tokens=output_tokens,
            )
            if not answer:
                raise ValueError("WORKER_EMPTY_ANSWER")
            if error:
                status = WorkerStatus.PARTIAL
        except TimeoutError:
            shared._harness_record_call(
                trace_id=shared.trace_id, call_id=call_id, stage="worker", status="timed_out",
                worker_id=task.worker_id, role=role.value,
            )
            status = WorkerStatus.TIMED_OUT
            error = "WORKER_TIMEOUT"
            answer = _skill_fallback(skill_results)
            if answer:
                status = WorkerStatus.PARTIAL
        except Exception as exc:  # noqa: BLE001 - provider SDK errors are vendor-specific
            if not reply_received:
                shared._harness_record_call(
                    trace_id=shared.trace_id, call_id=call_id, stage="worker", status="failed",
                    worker_id=task.worker_id, role=role.value, error=type(exc).__name__,
                )
            status = WorkerStatus.FAILED
            error = error or f"WORKER_PROVIDER_ERROR:{type(exc).__name__}"
            answer = _skill_fallback(skill_results)
            if answer:
                status = WorkerStatus.PARTIAL

        citations = tuple(
            citation for citation in self._verified_citations(shared)
            if citation.worker_id == task.worker_id
        )
        source_ids = tuple(citation.source_id for citation in citations)
        report = WorkerReport(
            worker_id=task.worker_id,
            role=role,
            objective=task.objective,
            status=status,
            answer_text=answer,
            observed_facts=tuple(dict.fromkeys(_VALUE.findall(answer))),
            observed_evidence_ids=source_ids,
            citations=citations,
            tool_calls=sum(result.tool_calls for result in skill_results),
            provider_calls=call_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(monotonic() - started) * 1000,
            error=error,
            model_turns=model_turns,
        )
        shared._harness_add_report(report)
        return report

    async def _run_single(
        self,
        request: MedicalAgentRequest,
        shared: SharedContext,
        *,
        worker_id: str,
    ) -> tuple[str, tuple[Citation, ...], set[str], tuple[int, int, int, int], dict[str, Any]]:
        shared._harness_timeline("single_agent_started", worker_id=worker_id)
        normalized = request.query.casefold()
        patient_skill = "TimelineCompareSkill" if any(
            cue in normalized for cue in ("compare", "change", "trend", "before", "变化", "对比")
        ) else "PatientStateLookupSkill"
        source_skills = await asyncio.gather(*(
            self.skills.execute(
                WorkerRole.PATIENT_CONTEXT if name == patient_skill else WorkerRole.EVIDENCE,
                name,
                SkillContext(
                    query=request.query,
                    role=WorkerRole.PATIENT_CONTEXT if name == patient_skill
                    else WorkerRole.EVIDENCE,
                    episode=request.episode,
                    resources=request.resources,
                    patient_id=(request.patient_id or (
                        request.episode.subject_id if request.episode else None
                    )),
                    as_of_time=(request.as_of_time or (
                        request.episode.decision_time if request.episode else None
                    )),
                ),
            ) for name in (patient_skill, "ExternalEvidenceSearchSkill")
        ))
        observed_context = tuple(result.output for result in source_skills if result.output)
        care_context = SkillContext(
            query=request.query,
            role=WorkerRole.CARE,
            episode=request.episode,
            resources=request.resources,
            observed_context=observed_context,
            patient_id=request.patient_id,
            as_of_time=request.as_of_time,
        )
        care_results = await asyncio.gather(*(
            self.skills.execute(WorkerRole.CARE, name, care_context)
            for name in ("RiskAssessmentSkill", "AnswerabilitySkill")
        ))
        skill_results = [*source_skills, *care_results]
        for result in skill_results:
            shared._harness_timeline(
                "skill_call", worker_id=worker_id, role="single", skill=result.skill_name,
                success=result.error is None,
            )
            if result.tool_calls:
                call_id = f"tool-{uuid4().hex}"
                shared._harness_timeline(
                    "tool_call", worker_id=worker_id, tool_call_id=call_id,
                    tool_id=result.tool_id,
                )
                for source in result.sources:
                    shared._harness_add_evidence(
                        source_id=source.source_id,
                        worker_id=worker_id,
                        role="single",
                        tool_call_id=call_id,
                        tool_id=result.tool_id or source.tool_id,
                        excerpt=source.excerpt[:1200],
                        input_hash=result.input_hash,
                        output_hash=result.output_hash,
                        resource_versions=result.resource_versions,
                    )
        user_payload = {
            "query": request.query,
            "conversation_context": list(request.conversation_context),
            "harness_observations": [
                {"skill": row.skill_name, "status": "error" if row.error else "observed",
                 "error": row.error, "output": row.output[:6000]}
                for row in skill_results
            ],
            "hospital_knowledge_search": "disabled; no hospital corpus is loaded",
        }
        user_prompt = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
        input_record = {"system_prompt": _SINGLE_PROMPT, "user_prompt": user_prompt}
        flags: set[str] = set()
        provider_attempted = False
        reply: ModelReply | None = None
        input_token_count = 0
        output_token_count = 0
        call_id = f"provider-{uuid4().hex}"
        try:
            provider_attempted = True
            shared._harness_timeline("provider_call_started", call_id=call_id, stage="single")
            reply = await asyncio.wait_for(
                self.provider.complete(
                    system_prompt=_SINGLE_PROMPT,
                    user_prompt=user_prompt,
                    max_output_tokens=self.config.max_single_output_tokens,
                    timeout_seconds=self.config.synthesis_timeout_seconds,
                ),
                timeout=self.config.synthesis_timeout_seconds,
            )
            input_token_count = reply.input_tokens
            output_token_count = reply.output_tokens
            shared._harness_record_call(
                trace_id=shared.trace_id, call_id=call_id, stage="single", status="complete",
                worker_id=worker_id, input_tokens=input_token_count,
                output_tokens=output_token_count,
            )
            answer = _strip_nonanswer(reply.content)
            if not answer:
                raise ValueError("SINGLE_AGENT_EMPTY_ANSWER")
            counts = (1, reply.input_tokens, reply.output_tokens,
                      sum(result.tool_calls for result in skill_results))
        except Exception as exc:  # noqa: BLE001 - provider SDK errors are vendor-specific
            if provider_attempted and reply is None:
                shared._harness_record_call(
                    trace_id=shared.trace_id, call_id=call_id, stage="single", status="failed",
                    worker_id=worker_id, error=type(exc).__name__,
                )
            answer = _skill_fallback(skill_results) or "INSUFFICIENT_EVIDENCE"
            flags.add(f"single_model_degraded:{type(exc).__name__}")
            counts = (
                int(provider_attempted), input_token_count, output_token_count,
                sum(result.tool_calls for result in skill_results),
            )
        shared._harness_timeline("single_agent_completed", worker_id=worker_id)
        citations = self._verified_citations(shared)
        return answer, citations, flags, counts, input_record

    async def _synthesize(
        self, request: MedicalAgentRequest, shared: SharedContext,
    ) -> tuple[str, ModelReply | None, dict[str, Any]]:
        lead_view = shared.lead_view()
        prompt_data = {
            "query": request.query,
            "conversation_context": list(request.conversation_context),
            "worker_status": lead_view["worker_status"],
            "worker_reports": lead_view["worker_reports"],
            "evidence_ledger": [
                {"source_id": item["source_id"], "role": item["role"],
                 "tool_id": item["tool_id"], "excerpt": item["excerpt"][:1200]}
                for item in lead_view["evidence_ledger"]
            ],
            "risk_flags": lead_view["risk_flags"],
        }
        user_prompt = json.dumps(prompt_data, ensure_ascii=False, separators=(",", ":"))
        shared._harness_timeline("lead_synthesis_started")
        reply: ModelReply | None = None
        call_id = f"provider-{uuid4().hex}"
        try:
            shared._harness_timeline("provider_call_started", call_id=call_id, stage="synthesis")
            reply = await asyncio.wait_for(
                self.provider.complete(
                    system_prompt=_LEAD_FINAL_PROMPT,
                    user_prompt=user_prompt,
                    max_output_tokens=self.config.max_lead_output_tokens,
                    timeout_seconds=self.config.synthesis_timeout_seconds,
                ),
                timeout=self.config.synthesis_timeout_seconds,
            )
            shared._harness_record_call(
                trace_id=shared.trace_id, call_id=call_id, stage="synthesis", status="complete",
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
            )
            answer = _strip_nonanswer(reply.content)
            if not answer:
                raise ValueError("LEAD_SYNTHESIS_EMPTY")
            shared._harness_timeline("lead_synthesis_completed", fallback=False)
            return answer, reply, {
                "system_prompt": _LEAD_FINAL_PROMPT, "user_prompt": user_prompt,
                "provider_attempted": True,
            }
        except Exception as exc:  # noqa: BLE001 - provider SDK errors are vendor-specific
            if reply is None:
                shared._harness_record_call(
                    trace_id=shared.trace_id, call_id=call_id, stage="synthesis", status="failed",
                    error=type(exc).__name__,
                )
            completed = [
                f"{report.role.value}: {report.answer_text}"
                for report in lead_view["worker_reports"]
                if report["status"] in {WorkerStatus.COMPLETE.value, WorkerStatus.PARTIAL.value}
                and report["answer_text"]
            ]
            statuses = [
                f"{report['role']} status={report['status']} error={report.get('error') or 'none'}"
                for report in lead_view["worker_reports"]
            ]
            fallback = "\n".join([*statuses, *completed]) or "INSUFFICIENT_EVIDENCE"
            shared._harness_timeline("lead_synthesis_completed", fallback=True,
                                     error=type(exc).__name__)
            return fallback, None, {
                "system_prompt": _LEAD_FINAL_PROMPT, "user_prompt": user_prompt,
                "fallback": type(exc).__name__, "provider_attempted": True,
            }

    def _skills_for_worker(
        self, role: WorkerRole, query: str, episode: IntegrationEpisode | None,
    ) -> tuple[str, ...]:
        normalized = query.casefold()
        if role == WorkerRole.PATIENT_CONTEXT:
            if (episode and episode.patient_state_ref) or self.skills.memory_provider_configured:
                name = ("TimelineCompareSkill" if any(
                    cue in normalized for cue in ("compare", "change", "trend", "before", "变化", "对比")
                ) else "PatientStateLookupSkill")
                return (name,)
            return ()
        if role == WorkerRole.EVIDENCE:
            rows: list[str] = []
            if (episode and episode.external_world_ref) or self.skills.external_evidence_provider_configured:
                rows.append("ExternalEvidenceSearchSkill")
            if any(cue in normalized for cue in ("hospital", "clinic", "慧宜", "医院", "门诊")):
                rows.append("HospitalKnowledgeSearchSkill")
            return tuple(rows[:self.config.max_tool_calls_per_worker])
        return ("RiskAssessmentSkill", "AnswerabilitySkill")

    @staticmethod
    def _verified_citations(shared: SharedContext) -> tuple[Citation, ...]:
        payload = shared.to_dict()
        ledger = {item["evidence_id"] for item in payload["evidence_ledger"]}
        citations = []
        seen: set[str] = set()
        for row in payload["citations"]:
            citation = Citation(
                evidence_id=row["evidence_id"],
                source_id=row["source_id"],
                source=row["source"],
                excerpt=row["excerpt"],
                worker_id=row["worker_id"],
                tool_id=row["tool_id"],
            )
            if citation.evidence_id in ledger and citation.evidence_id not in seen:
                citations.append(citation)
                seen.add(citation.evidence_id)
        return tuple(citations)

    @staticmethod
    def _harness_safety_flags(query: str) -> set[str]:
        normalized = query.casefold()
        if any(marker in normalized for marker in _URGENT_MARKERS):
            return {"urgent_symptom_requires_human_care"}
        return set()

    @staticmethod
    def _verify_answer(
        answer: str, query: str, has_citations: bool, flags: set[str],
    ) -> str:
        verified = answer.strip()
        if not verified:
            verified = "INSUFFICIENT_EVIDENCE"
        if "urgent_symptom_requires_human_care" in flags:
            caution = "如出现紧急症状，请立即联系当地急救或前往急诊。"
            if caution not in verified:
                verified = f"{verified}\n\n{caution}"
        if not has_citations and any(marker in query.casefold() for marker in _VALUE_MARKERS):
            # Keep synthetic benchmark answers intact; grounding is reported by the evaluator.
            return verified
        return verified


_VALUE_MARKERS = ("synkey-",)


def _strip_nonanswer(text: str) -> str:
    answer = text.strip()
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.IGNORECASE | re.DOTALL).strip()
    if answer.startswith("```"):
        answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer, flags=re.IGNORECASE).strip()
    return answer


def _skill_fallback(results: list[SkillResult]) -> str:
    pieces = [result.output for result in results if result.output]
    if pieces:
        return " ".join(pieces)
    return ""
