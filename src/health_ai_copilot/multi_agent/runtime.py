"""Routed Single/Team execution with harness-owned budgets and provenance."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field, replace
from threading import RLock
from time import monotonic
from typing import Any
from uuid import uuid4

from ..research.integration.contracts import IntegrationEpisode
from .contracts import (
    Citation,
    CoverageLedger,
    CoverageStatus,
    LeadPlan,
    MedicalAgentRequest,
    MedicalAgentResponse,
    PlannedAspect,
    RouteMode,
    TaskAspect,
    TaskAssignment,
    TaskLedger,
    TriageDecision,
    WorkerArtifact,
    WorkerReport,
    WorkerRole,
    WorkerStatus,
)
from .mdagents_style import ClinicalReasoningSkill
from .orchestration import (
    artifact_from_worker_output,
    coverage_judgment_prompt,
    fallback_aspects,
    harness_task_ledger,
    make_coverage_ledger,
    parse_aspect_plan,
    parse_coverage_judgment,
    task_ledger_with_coverage,
)
from .providers import ModelProvider, ModelReply
from .routing import (
    LocalTriageProvider,
    MedicalRouter,
    RouteDecision,
    deterministic_fallback_plan,
    parse_lead_plan,
    route_from_triage,
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
_MVP2_LEAD_PLAN_PROMPT = (
    "你是 Medical Lead，只负责把用户请求拆成所有需要独立回答的 aspect，并给每个 aspect "
    "分配一个已允许的 worker。每个 aspect 对应一个不同的用户交付结果；同一对象或结果的语义改写、"
    "定位线索、限定语和重复的返回要求都合并到同一个 aspect。相同实体键最多一个 aspect；"
    "不同键各自要求独立结果时，才分别建立 aspect。保留所有独立交付结果，不漏项，不重复，不回答问题。"
    "例如，同一键的“返回结果、给出该键索引的项目、返回该项目的值”合并为一个 aspect；"
    "明确要求分别返回 KEY-A 和 KEY-B 的值时建立两个 aspect。"
    "只输出一行有效 JSON，不要 Markdown；顶层只能有 aspects。每项严格包含 aspect、worker、"
    "expected_output 三个字符串字段。worker 只能是 patient_context、evidence、care。"
    '格式示例：{"aspects":[{"aspect":"比较患者两次记录","worker":"patient_context",'
    '"expected_output":"给出两次记录及变化"}]}。最多 12 个 aspect。'
    "不要输出 ID、状态、预算、mode、tasks、source_id、evidence_id 或事实答案。"
)
_MVP2_ROLE_PROMPTS = {
    WorkerRole.PATIENT_CONTEXT: (
        "你是 PatientContextAgent。只根据当前患者记录回答分配给你的 aspect。"
        "只输出有效 JSON，不要 Markdown；且只含 findings、facts、unresolved、answer_fragment 四个字段。"
        "前三项是字符串数组，answer_fragment 是字符串。facts 只列回答该 aspect 必需的观察原文，逐字复制，"
        "不可改写、不可生成 ID。findings 和 unresolved 简短填写；answer_fragment 最多 2 句、45 个英文单词"
        "（或 60 个汉字），不要重复列出 facts。不得补造患者历史；信息不足时说明具体缺项。"
    ),
    WorkerRole.EVIDENCE: (
        "你是 EvidenceAgent。只根据当前外部医学证据回答分配给你的 aspect。"
        "只输出有效 JSON，不要 Markdown；且只含 findings、facts、unresolved、answer_fragment 四个字段。"
        "前三项是字符串数组，answer_fragment 是字符串。facts 只列回答该 aspect 必需的观察原文，逐字复制，"
        "不可改写、不可生成 ID。findings 和 unresolved 简短填写；answer_fragment 最多 2 句、45 个英文单词"
        "（或 60 个汉字），不要重复列出 facts。不得补入未观察到的医学事实或来源；信息不足时说明具体缺项。"
    ),
    WorkerRole.CARE: (
        "你是 CareAgent。只负责分配给你的风险、可回答性和下一步行动 aspect。"
        "只输出有效 JSON，不要 Markdown；且只含 findings、facts、unresolved、answer_fragment 四个字段。"
        "前三项是字符串数组，answer_fragment 是字符串。facts 只列回答该 aspect 必需的观察原文，逐字复制，"
        "不可改写、不可生成 ID。findings 和 unresolved 简短填写；answer_fragment 最多 2 句、45 个英文单词"
        "（或 60 个汉字），不要重复列出 facts。"
        "不做诊断、处方或药物剂量调整；紧急症状提示及时线下急救。"
    ),
}
_MVP2_COVERAGE_PROMPT = (
    "你是受限的任务覆盖检查器，不提供医学判断。逐项比较每个 aspect 的 expected_output 与对应 WorkerArtifact，"
    "verified facts 不必逐字复述 aspect；若 answer_fragment 或 verified facts 中有可核验的对象/键到结果映射，"
    "且直接给出该 aspect 请求的结果，就标 COVERED。同一条答案可以覆盖多个重叠 aspect；不要仅因答案简短、"
    "或同一 worker 同时回答多个 aspect 而降为 PARTIAL。有相关事实但缺少所请求的对象或结果时标 PARTIAL；"
    "没有可用的相关内容时标 MISSING。"
    '只返回有效 JSON：{"statuses":["COVERED","PARTIAL"]}。每个输入 aspect 必须且只能有一个状态，'
    "顺序与输入 aspects 一致。不要输出说明、ID 或其他字段。"
)
_MVP2_FINAL_PROMPT = (
    "你是 Medical Lead finalizer。输入包含 TaskLedger、WorkerArtifacts、CoverageLedger、EvidenceLedger 和安全标记。"
    "按 TaskLedger 逐项回答用户提出的所有 aspect；不得为了简洁删除已由 Harness 核验的 facts，"
    "若多份 WorkerArtifact/repair 重复同一已核验事实，最终答案中将该事实写一次；保留所有不同 facts，"
    "同一精确 token 不要重复输出。"
    "只将 ArtifactFact.text 当作已验证事实，只引用 EvidenceLedger 中存在的来源。不要生成来源或事实 ID。"
    "若 CoverageLedger 标为 PARTIAL/MISSING，诚实指出未解决部分，不把它说成已完成。"
    "只给用户最终答案，不展示推理；不诊断、不处方、不调整药物剂量，有紧急症状时提示及时线下急救。"
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
    max_repair_workers: int = 2
    max_coverage_output_tokens: int = 64

    def __post_init__(self) -> None:
        if self.max_model_turns_per_worker > 3 or self.max_model_turns_per_worker < 1:
            raise ValueError("worker model turns must be between one and three")
        if self.max_tool_calls_per_worker > 2 or self.max_tool_calls_per_worker < 0:
            raise ValueError("worker tool calls must be between zero and two")
        if self.max_worker_calls > 3 or self.max_worker_calls < 1:
            raise ValueError("total worker calls must be between one and three")
        if self.max_repair_workers < 0 or self.max_repair_workers > 2:
            raise ValueError("repair wave must use at most two workers")


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
    task_ledger: TaskLedger | None = None
    coverage_ledger: CoverageLedger | None = None
    triage_decision: TriageDecision | None = None
    repair_wave: dict[str, Any] = field(default_factory=dict)


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
        triage_provider=None,
        triage_thresholds: dict[str, float] | None = None,
        clinical_reasoning_skill: ClinicalReasoningSkill | None = None,
    ) -> None:
        self.provider = provider
        self.router = router or MedicalRouter()
        self.skills = skill_registry or SkillRegistry()
        self.config = config or RuntimeConfig()
        self.observability = observability or AggregateObservability()
        self.triage_provider = triage_provider
        self.triage_thresholds = dict(triage_thresholds or {})
        self.clinical_reasoning_skill = clinical_reasoning_skill

    async def respond(self, request: MedicalAgentRequest) -> MedicalAgentResponse:
        return (await self.execute(request)).response

    async def execute(self, request: MedicalAgentRequest) -> RuntimeExecution:
        if self.clinical_reasoning_skill is not None:
            return await self._execute_clinical_reasoning(request)
        if self.triage_provider is not None:
            return await self._execute_mvp2(request)
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

    async def _execute_clinical_reasoning(
        self, request: MedicalAgentRequest,
    ) -> RuntimeExecution:
        """Adapt the opt-in MDAgents-style sub-runtime to the stable API execution contract."""
        result = await self.clinical_reasoning_skill.run(request)
        response = result.response
        complexity = result.complexity
        route = RouteDecision(
            mode=response.route_mode,
            reason=f"mdagents_style_complexity:{complexity.value}",
            predicted_capabilities=(WorkerRole.CARE,),
            unique_key_count=0,
        )
        events = [
            {
                "sequence": event.sequence,
                "event_type": event.event_type.value,
                **dict(event.fields),
            }
            for event in result.trace_events
        ]
        trajectory = {
            "runtime_version": "MDAGENTS_STYLE_V1",
            "complexity": complexity.value,
            "parsed_option": result.parsed_option,
            "raw_model_output": result.raw_model_output,
            "failure_reason": result.failure_reason,
            "token_usage_known": (
                result.input_tokens is not None and result.output_tokens is not None
            ),
        }
        execution = RuntimeExecution(
            response=response,
            route_decision=route,
            plan=None,
            worker_reports=(),
            trace={
                "runtime_version": "MDAGENTS_STYLE_V1",
                "events": events,
                "shared_context": result.shared_context,
            },
            provider_calls=result.provider_calls,
            tool_calls=result.tool_calls,
            input_tokens=int(result.input_tokens or 0),
            output_tokens=int(result.output_tokens or 0),
            worker_wave_wall_ms=result.latency_ms,
            sequential_worker_latency_ms=result.latency_ms,
            partial_failure_recovered=False,
            trajectory=trajectory,
        )
        self.observability.record(execution)
        return execution

    async def _execute_mvp2(self, request: MedicalAgentRequest) -> RuntimeExecution:
        """MVP2 route with harness-owned tasks, artifacts, coverage, and one repair wave."""
        started = monotonic()
        trace_id = request.request_id or f"trace-{uuid4().hex}"
        shared = SharedContext(trace_id, request.query)
        shared._harness_timeline("request_started", runtime_version="MA_MVP2")
        observable_context = self._triage_observable_context(request)
        triage_provider = self.triage_provider or LocalTriageProvider()
        try:
            triage = await triage_provider.triage(request.query, observable_context)
        except Exception as exc:  # noqa: BLE001 - routing always has a deterministic fallback.
            triage = await LocalTriageProvider().triage(request.query, observable_context)
            triage = replace(triage, provider="local_fallback", fallback_reason=type(exc).__name__)
        route = route_from_triage(triage, thresholds=self.triage_thresholds)
        shared._harness_timeline(
            "triage_decision", decision=triage.to_dict(), route=route.to_dict(),
        )
        counts = [int(triage.provider == "jev"), triage.input_tokens,
                  triage.output_tokens, 0]
        reports: list[WorkerReport] = []
        plan: LeadPlan | None = None
        task_ledger: TaskLedger | None = None
        coverage_ledger: CoverageLedger | None = None
        trajectory: dict[str, Any] = {
            "runtime_version": "MA_MVP2",
            "query": request.query,
            "conversation_context": list(request.conversation_context),
            "observable_context": observable_context,
            "triage_decision": triage.to_dict(),
            "worker_inputs": [],
            "lead_synthesis_input": None,
            "final_answer": None,
        }
        total_worker_wall = 0.0
        sequential_worker_ms = 0.0
        partial_failure_recovered = False
        repair_wave: dict[str, Any] = {"invoked": False, "worker_ids": [], "aspect_ids": []}
        flags: set[str] = set()
        actual_mode = route.mode

        try:
            if route.mode == RouteMode.SINGLE:
                answer, citations, single_flags, single_counts, single_input = await self._run_single(
                    request, shared, worker_id=f"single-{uuid4().hex[:12]}",
                )
                counts = [counts[index] + single_counts[index] for index in range(4)]
                trajectory["single_input"] = single_input
                flags.update(single_flags)
                actual_mode = RouteMode.SINGLE
            else:
                (plan, planned, plan_reply, plan_input) = await self._plan_mvp2(
                    request, route, shared, trajectory,
                )
                counts[0] += int(plan_input.get("provider_attempted", False))
                if plan_reply:
                    counts[1] += plan_reply.input_tokens
                    counts[2] += plan_reply.output_tokens
                trajectory["lead_plan_input"] = plan_input
                shared._harness_set_plan(plan)
                reports, task_ledger, wave_ms = await self._run_mvp2_plan(
                    request, plan, planned, shared, trajectory,
                )
                total_worker_wall += wave_ms
                counts[0] += sum(report.provider_calls for report in reports)
                counts[1] += sum(report.input_tokens for report in reports)
                counts[2] += sum(report.output_tokens for report in reports)
                counts[3] += sum(report.tool_calls for report in reports)
                sequential_worker_ms += sum(report.latency_ms for report in reports)

                coverage_ledger, coverage_reply, coverage_input = await self._judge_coverage(
                    task_ledger, reports, shared,
                )
                trajectory.setdefault("coverage_judgments", []).append(coverage_input)
                counts[0] += int(coverage_input.get("provider_attempted", False))
                if coverage_reply:
                    counts[1] += coverage_reply.input_tokens
                    counts[2] += coverage_reply.output_tokens
                task_ledger = task_ledger_with_coverage(task_ledger, coverage_ledger)
                shared._harness_set_task_ledger(task_ledger)
                shared._harness_set_coverage_ledger(coverage_ledger)

                missing = [item for item in coverage_ledger.items
                           if item.status in {CoverageStatus.MISSING, CoverageStatus.PARTIAL}]
                if missing and self.config.max_repair_workers:
                    repair_reports, task_ledger, repair_wave, repair_wall = (
                        await self._run_mvp2_repair_wave(
                            request, task_ledger, coverage_ledger, reports, shared, trajectory,
                        )
                    )
                    if repair_reports:
                        reports.extend(repair_reports)
                        total_worker_wall += repair_wall
                        sequential_worker_ms += sum(report.latency_ms for report in repair_reports)
                        counts[0] += sum(report.provider_calls for report in repair_reports)
                        counts[1] += sum(report.input_tokens for report in repair_reports)
                        counts[2] += sum(report.output_tokens for report in repair_reports)
                        counts[3] += sum(report.tool_calls for report in repair_reports)
                        coverage_ledger, coverage_reply, coverage_input = await self._judge_coverage(
                            task_ledger, reports, shared,
                        )
                        trajectory.setdefault("coverage_judgments", []).append(coverage_input)
                        counts[0] += int(coverage_input.get("provider_attempted", False))
                        if coverage_reply:
                            counts[1] += coverage_reply.input_tokens
                            counts[2] += coverage_reply.output_tokens
                        task_ledger = task_ledger_with_coverage(task_ledger, coverage_ledger)
                        shared._harness_set_task_ledger(task_ledger)
                        shared._harness_set_coverage_ledger(coverage_ledger)
                shared._harness_timeline("repair_wave", **repair_wave)

                # Safety signals derived from the original request must be visible
                # to the finalizer, not appended to the shared context afterward.
                flags.update(self._harness_safety_flags(request.query))
                for flag in sorted(flags):
                    shared._harness_risk_flag(flag)

                answer, synthesis_reply, synthesis_input = await self._synthesize_mvp2(
                    request, shared, task_ledger, coverage_ledger,
                )
                counts[0] += int(synthesis_input.get("provider_attempted", False))
                if synthesis_reply:
                    counts[1] += synthesis_reply.input_tokens
                    counts[2] += synthesis_reply.output_tokens
                trajectory["lead_synthesis_input"] = synthesis_input
                actual_mode = RouteMode.TEAM
                if not any(report.status in {WorkerStatus.COMPLETE, WorkerStatus.PARTIAL}
                           for report in reports):
                    partial_failure_recovered = True
                    shared._harness_timeline("team_fallback_single")
                    answer, _, single_flags, single_counts, single_input = await self._run_single(
                        request, shared, worker_id=f"fallback-single-{uuid4().hex[:12]}",
                    )
                    counts = [counts[index] + single_counts[index] for index in range(4)]
                    flags.update(single_flags)
                    flags.add("team_fallback_single")
                    trajectory["single_fallback_input"] = single_input
                    actual_mode = RouteMode.SINGLE
                else:
                    partial_failure_recovered = any(
                        report.status in {WorkerStatus.FAILED, WorkerStatus.TIMED_OUT}
                        for report in reports
                    )
                citations = self._verified_citations(shared)

            flags.update(self._harness_safety_flags(request.query))
            for flag in sorted(flags):
                shared._harness_risk_flag(flag)
            answer = self._verify_answer(answer, request.query, bool(citations), flags)
            response = MedicalAgentResponse(
                answer=answer,
                route_mode=actual_mode,
                workers_used=tuple(dict.fromkeys(
                    report.role.value for report in reports
                    if report.status in {WorkerStatus.COMPLETE, WorkerStatus.PARTIAL}
                )),
                citations=self._verified_citations(shared),
                safety_flags=tuple(sorted(flags)),
                trace_id=trace_id,
                latency_ms=max(0, int((monotonic() - started) * 1000)),
            )
            trajectory["final_answer"] = answer
            shared._harness_timeline(
                "finalization", task_ledger=task_ledger.to_dict() if task_ledger else None,
                coverage_ledger=coverage_ledger.to_dict() if coverage_ledger else None,
                safety_flags=sorted(flags),
            )
            shared._harness_timeline("request_completed", route_mode=actual_mode.value)
        except Exception as exc:  # noqa: BLE001 - preserve the safe request-level degradation.
            shared._harness_timeline("request_error", error=type(exc).__name__)
            flags = self._harness_safety_flags(request.query) | {"runtime_degraded_safe_response"}
            response = MedicalAgentResponse(
                answer="当前请求未能可靠完成。请稍后重试；如有紧急症状，请及时联系当地急救服务。",
                route_mode=actual_mode,
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
            provider_calls=counts[0],
            tool_calls=counts[3],
            input_tokens=counts[1],
            output_tokens=counts[2],
            worker_wave_wall_ms=total_worker_wall,
            sequential_worker_latency_ms=sequential_worker_ms,
            partial_failure_recovered=partial_failure_recovered,
            trajectory=trajectory,
            task_ledger=task_ledger,
            coverage_ledger=coverage_ledger,
            triage_decision=triage,
            repair_wave=repair_wave,
        )
        self.observability.record(execution)
        return execution

    @staticmethod
    def _triage_observable_context(request: MedicalAgentRequest) -> str:
        state: dict[str, Any] = {
            "conversation_context": list(request.conversation_context),
        }
        if request.episode is not None:
            observable = request.episode.observable_state
            state.update({
                "history_exists": observable.history_exists,
                "history_length_bucket": observable.history_length_bucket,
                "history_time_span": observable.history_time_span,
                "available_personal_state_types": list(observable.available_personal_state_types),
                "available_external_source_families": list(observable.available_external_source_families),
                "available_tool_ids": list(observable.available_tool_ids),
                "available_worker_capabilities": list(observable.available_worker_capabilities),
                "budget_class": observable.budget_class,
                "deadline_class": observable.deadline_class,
            })
        return json.dumps(state, ensure_ascii=False, separators=(",", ":"))

    async def _plan_mvp2(
        self, request: MedicalAgentRequest, route: RouteDecision, shared: SharedContext,
        trajectory: dict[str, Any],
    ) -> tuple[LeadPlan, tuple[PlannedAspect, ...], ModelReply | None, dict[str, Any]]:
        allowed = route.predicted_capabilities or (WorkerRole.EVIDENCE,)
        user_prompt = json.dumps({
            "query": request.query,
            "conversation_context": list(request.conversation_context),
            "triage_decision": route.triage_decision.to_dict() if route.triage_decision else {},
            "allowed_workers": [role.value for role in allowed],
            "worker_capabilities": {
                role.value: list(self.skills.allowed_for(role)) for role in allowed
            },
        }, ensure_ascii=False, separators=(",", ":"))
        shared._harness_timeline("lead_plan_started", schema="aspects-only-v1")
        reply: ModelReply | None = None
        call_id = f"provider-{uuid4().hex}"
        source = "model_aspects"
        try:
            shared._harness_timeline("provider_call_started", call_id=call_id, stage="plan")
            reply = await asyncio.wait_for(self.provider.complete(
                system_prompt=_MVP2_LEAD_PLAN_PROMPT,
                user_prompt=user_prompt,
                max_output_tokens=max(160, self.config.max_lead_output_tokens),
                timeout_seconds=self.config.planning_timeout_seconds,
                json_mode=True,
            ), timeout=self.config.planning_timeout_seconds)
            shared._harness_record_call(
                trace_id=shared.trace_id, call_id=call_id, stage="plan", status="complete",
                input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
            )
            planned = parse_aspect_plan(reply.content, allowed_workers=allowed)
        except Exception as exc:  # noqa: BLE001 - invalid plans fall back to a bounded harness plan.
            source = "harness_fallback"
            if reply is None:
                shared._harness_record_call(
                    trace_id=shared.trace_id, call_id=call_id, stage="plan", status="failed",
                    error=type(exc).__name__,
                )
            shared._harness_timeline("lead_plan_fallback", error=type(exc).__name__)
            planned = fallback_aspects(request.query, allowed)

        # Every triaged role receives at least one aspect; the Lead cannot silently drop a worker.
        present = {item.worker for item in planned}
        for role in allowed:
            if role not in present:
                planned += (PlannedAspect(
                    aspect=request.query[:1200],
                    worker=role,
                    expected_output=(
                        f"Address the part of the original request requiring {role.value}; "
                        "preserve verified facts and state any gap."
                    ),
                ),)
        objectives = []
        for role in dict.fromkeys(item.worker for item in planned):
            role_aspects = [item for item in planned if item.worker == role]
            objective = "\n".join(
                f"Aspect: {item.aspect}\nExpected: {item.expected_output}" for item in role_aspects
            )
            objectives.append((role, objective[:4000]))
        plan = LeadPlan(RouteMode.TEAM, tuple(objectives), source=source)
        return plan, planned, reply, {
            "system_prompt": _MVP2_LEAD_PLAN_PROMPT,
            "user_prompt": user_prompt,
            "provider_attempted": True,
            "source": source,
        }

    async def _run_mvp2_plan(
        self,
        request: MedicalAgentRequest,
        plan: LeadPlan,
        planned: tuple[PlannedAspect, ...],
        shared: SharedContext,
        trajectory: dict[str, Any],
    ) -> tuple[list[WorkerReport], TaskLedger, float]:
        roles = tuple(role for role, _objective in plan.tasks)
        specialists = tuple(role for role in roles if role != WorkerRole.CARE)
        tasks = [shared._harness_register_task(
            role.value, objective, specialists if role == WorkerRole.CARE else (),
        ) for role, objective in plan.tasks]
        worker_ids = {task.role: task.worker_id for task in tasks}
        ledger, aspect_ids_by_role = harness_task_ledger(
            shared.request_id, planned, worker_ids,
        )
        tasks = [
            replace(task, aspect_ids=aspect_ids_by_role.get(task.role, ()))
            for task in tasks
        ]
        for task in tasks:
            shared._harness_set_task_aspects(task.worker_id, task.aspect_ids)
        shared._harness_set_task_ledger(ledger)
        aspect_by_id = {item.aspect_id: item for item in ledger.aspects}
        dependencies_by_role = {task.role: task.depends_on for task in tasks}
        reports: list[WorkerReport] = []

        async def run_wave(wave_tasks) -> tuple[list[WorkerReport], float]:
            if not wave_tasks:
                return [], 0.0
            wave_started = monotonic()
            wave_reports = await asyncio.gather(*(
                self._run_worker(
                    request,
                    task,
                    shared,
                    dependencies=tuple(report for report in reports
                                       if report.role in dependencies_by_role.get(task.role, ())),
                    trajectory=trajectory,
                    mvp2=True,
                    aspect_items=tuple(aspect_by_id[item] for item in task.aspect_ids),
                ) for task in wave_tasks
            ))
            return list(wave_reports), (monotonic() - wave_started) * 1000

        wave_one = [task for task in tasks if not task.depends_on]
        wave_two = [task for task in tasks if task.depends_on]
        first, first_ms = await run_wave(wave_one)
        reports.extend(first)
        second, second_ms = await run_wave(wave_two)
        reports.extend(second)
        shared._harness_timeline(
            "worker_waves_completed", wave_count=1 + int(bool(wave_two)),
            worker_calls=len(reports), wall_ms=round(first_ms + second_ms, 3),
        )
        return reports, ledger, first_ms + second_ms

    async def _judge_coverage(
        self,
        ledger: TaskLedger,
        reports: list[WorkerReport],
        shared: SharedContext,
    ) -> tuple[CoverageLedger, ModelReply | None, dict[str, Any]]:
        user_prompt = coverage_judgment_prompt(ledger, tuple(reports))
        call_id = f"provider-{uuid4().hex}"
        reply: ModelReply | None = None
        judge = "harness-fallback"
        statuses: tuple[CoverageStatus, ...] | None = None
        try:
            shared._harness_timeline("coverage_judgment_started", aspect_count=len(ledger.aspects))
            shared._harness_timeline("provider_call_started", call_id=call_id, stage="coverage_judgment")
            reply = await asyncio.wait_for(self.provider.complete(
                system_prompt=_MVP2_COVERAGE_PROMPT,
                user_prompt=user_prompt,
                max_output_tokens=self.config.max_coverage_output_tokens,
                timeout_seconds=self.config.synthesis_timeout_seconds,
                json_mode=True,
            ), timeout=self.config.synthesis_timeout_seconds)
            statuses = parse_coverage_judgment(reply.content, len(ledger.aspects))
            by_worker = {report.worker_id: report for report in reports}
            corrected: list[CoverageStatus] = []
            for aspect, status in zip(ledger.aspects, statuses, strict=True):
                assigned = [item for item in ledger.assignments if item.aspect_id == aspect.aspect_id]
                has_output = any(
                    item.worker_id in by_worker
                    and by_worker[item.worker_id].status in {WorkerStatus.COMPLETE, WorkerStatus.PARTIAL}
                    and bool(by_worker[item.worker_id].answer_text.strip())
                    for item in assigned
                )
                corrected.append(status if has_output else CoverageStatus.MISSING)
            statuses = tuple(corrected)
            judge = f"bounded-model:{reply.model or 'configured-provider'}"
            shared._harness_record_call(
                trace_id=shared.trace_id, call_id=call_id, stage="coverage_judgment", status="complete",
                input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - invalid judge output uses the harness coverage gate.
            if reply is None:
                shared._harness_record_call(
                    trace_id=shared.trace_id, call_id=call_id, stage="coverage_judgment",
                    status="failed", error=type(exc).__name__,
                )
            shared._harness_timeline("coverage_judgment_fallback", error=type(exc).__name__)
        coverage = make_coverage_ledger(ledger, reports, statuses=statuses, judge=judge)
        shared._harness_timeline("coverage_judgment_completed", coverage=coverage.to_dict())
        return coverage, reply, {
            "system_prompt": _MVP2_COVERAGE_PROMPT,
            "user_prompt": user_prompt,
            "provider_attempted": True,
            "judge": judge,
        }

    async def _run_mvp2_repair_wave(
        self,
        request: MedicalAgentRequest,
        ledger: TaskLedger,
        coverage: CoverageLedger,
        prior_reports: list[WorkerReport],
        shared: SharedContext,
        trajectory: dict[str, Any],
    ) -> tuple[list[WorkerReport], TaskLedger, dict[str, Any], float]:
        deficient = {item.aspect_id for item in coverage.items
                     if item.status in {CoverageStatus.MISSING, CoverageStatus.PARTIAL}}
        assignments = [item for item in ledger.assignments if item.aspect_id in deficient]
        counts: dict[WorkerRole, int] = {}
        for item in assignments:
            counts[item.worker] = counts.get(item.worker, 0) + 1
        role_order = {role: index for index, role in enumerate(WorkerRole)}
        chosen = sorted(counts, key=lambda role: (-counts[role], role_order[role]))[
            :self.config.max_repair_workers
        ]
        if not chosen:
            return [], ledger, {"invoked": False, "worker_ids": [],
                                "aspect_ids": sorted(deficient)}, 0.0
        aspect_by_id = {item.aspect_id: item for item in ledger.aspects}
        repair_tasks = []
        repair_assignments = list(ledger.assignments)
        for role in chosen:
            target_ids = tuple(item.aspect_id for item in assignments if item.worker == role)
            target_aspects = [aspect_by_id[item] for item in target_ids]
            objective = "REPAIR ONLY these missing or partial aspects; do not redo covered work:\n" + (
                "\n".join(f"- [{item.aspect_id}] {item.aspect}: {item.expected_output}"
                          for item in target_aspects)
            )
            task = shared._harness_register_task(
                role.value, objective[:4000], (), target_ids,
            )
            repair_tasks.append(task)
            for aspect_id in target_ids:
                repair_assignments.append(TaskAssignment(
                    assignment_id=f"assignment-{uuid4().hex[:12]}",
                    aspect_id=aspect_id,
                    worker_id=task.worker_id,
                    worker=role,
                    objective=f"Repair missing aspect: {aspect_by_id[aspect_id].aspect}",
                ))
        updated_ledger = TaskLedger(ledger.request_id, ledger.aspects, tuple(repair_assignments))
        shared._harness_set_task_ledger(updated_ledger)
        task_aspects = {task.worker_id: tuple(aspect_by_id[item] for item in task.aspect_ids)
                        for task in repair_tasks}
        started = monotonic()
        repair_reports = list(await asyncio.gather(*(
            self._run_worker(
                request, task, shared, dependencies=tuple(prior_reports),
                trajectory=trajectory, mvp2=True, aspect_items=task_aspects[task.worker_id],
            ) for task in repair_tasks
        )))
        wall = (monotonic() - started) * 1000
        detail = {
            "invoked": True,
            "wave_count": 1,
            "worker_ids": [task.worker_id for task in repair_tasks],
            "workers": [task.role.value for task in repair_tasks],
            "aspect_ids": sorted(deficient),
            "worker_limit": self.config.max_repair_workers,
        }
        shared._harness_timeline("targeted_repair_completed", **detail, wall_ms=round(wall, 3))
        return repair_reports, updated_ledger, detail, wall

    async def _synthesize_mvp2(
        self,
        request: MedicalAgentRequest,
        shared: SharedContext,
        ledger: TaskLedger,
        coverage: CoverageLedger,
    ) -> tuple[str, ModelReply | None, dict[str, Any]]:
        view = shared.lead_view()
        prompt_data = {
            "original_query": request.query,
            "conversation_context": list(request.conversation_context),
            "task_ledger": ledger.to_dict(),
            "worker_artifacts": view["worker_artifacts"],
            "coverage_ledger": coverage.to_dict(),
            "evidence_ledger": view["evidence_ledger"],
            "safety_flags": view["risk_flags"],
        }
        user_prompt = json.dumps(prompt_data, ensure_ascii=False, separators=(",", ":"))
        call_id = f"provider-{uuid4().hex}"
        reply: ModelReply | None = None
        try:
            shared._harness_timeline("finalization_started", input_fields=list(prompt_data))
            shared._harness_timeline("provider_call_started", call_id=call_id, stage="finalization")
            reply = await asyncio.wait_for(self.provider.complete(
                system_prompt=_MVP2_FINAL_PROMPT,
                user_prompt=user_prompt,
                max_output_tokens=self.config.max_lead_output_tokens,
                timeout_seconds=self.config.synthesis_timeout_seconds,
            ), timeout=self.config.synthesis_timeout_seconds)
            answer = _strip_nonanswer(reply.content)
            if not answer:
                raise ValueError("MVP2_FINALIZER_EMPTY")
            shared._harness_record_call(
                trace_id=shared.trace_id, call_id=call_id, stage="finalization", status="complete",
                input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
            )
            shared._harness_timeline("finalization_completed", fallback=False)
            return answer, reply, {
                "system_prompt": _MVP2_FINAL_PROMPT,
                "user_prompt": user_prompt,
                "provider_attempted": True,
            }
        except Exception as exc:  # noqa: BLE001 - fallback retains verified artifact facts.
            if reply is None:
                shared._harness_record_call(
                    trace_id=shared.trace_id, call_id=call_id, stage="finalization", status="failed",
                    error=type(exc).__name__,
                )
            parts = []
            for artifact in view["worker_artifacts"]:
                parts.extend(fact["text"] for fact in artifact.get("facts", ()))
                if artifact.get("answer_fragment"):
                    parts.append(artifact["answer_fragment"])
            fallback = "\n".join(dict.fromkeys(parts)) or "INSUFFICIENT_EVIDENCE"
            shared._harness_timeline("finalization_completed", fallback=True,
                                     error=type(exc).__name__)
            return fallback, None, {
                "system_prompt": _MVP2_FINAL_PROMPT,
                "user_prompt": user_prompt,
                "fallback": type(exc).__name__,
                "provider_attempted": True,
            }

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
        mvp2: bool = False,
        aspect_items: tuple[TaskAspect, ...] = (),
    ) -> WorkerReport:
        started = monotonic()
        role = task.role
        shared._harness_worker_status(task.worker_id, WorkerStatus.RUNNING.value)
        dependency_text = tuple(
            json.dumps({
                "role": report.role.value,
                "status": report.status.value,
                "worker_artifact": report.artifact.to_dict() if report.artifact else None,
                "answer_fragment": report.answer_text or "[no answer]",
            }, ensure_ascii=False, separators=(",", ":"))
            if mvp2 else
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
                ({"role": report.role.value, "status": report.status.value,
                  "worker_artifact": report.artifact.to_dict() if report.artifact else None,
                  "answer_text": report.answer_text, "error": report.error}
                 if mvp2 else
                 {"role": report.role.value, "status": report.status.value,
                  "answer_text": report.answer_text, "error": report.error})
                for report in dependencies
            ],
        }
        if mvp2:
            user_payload["assigned_aspects"] = [
                {"aspect": item.aspect, "expected_output": item.expected_output}
                for item in aspect_items
            ]
            user_payload["artifact_contract"] = {
                "findings": "string array",
                "facts": "string array; each item must be an exact quote from observations",
                "unresolved": "string array",
                "answer_fragment": "string",
            }
        user_prompt = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
        system_prompt = _MVP2_ROLE_PROMPTS[role] if mvp2 else _ROLE_PROMPTS[role]
        trajectory.setdefault("worker_inputs", []).append({
            "worker_id": task.worker_id,
            "role": role.value,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
        call_count = 0
        input_tokens = 0
        output_tokens = 0
        answer = ""
        raw_output = ""
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
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=self.config.max_worker_output_tokens,
                    timeout_seconds=self.config.worker_timeout_seconds,
                    json_mode=mvp2,
                ),
                timeout=self.config.worker_timeout_seconds,
            )
            reply_received = True
            input_tokens = reply.input_tokens
            output_tokens = reply.output_tokens
            raw_output = reply.content
            answer = _mvp2_answer_fragment(reply.content) if mvp2 else _strip_nonanswer(reply.content)
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
        artifact: WorkerArtifact | None = None
        if mvp2:
            artifact = artifact_from_worker_output(
                worker_id=task.worker_id,
                role=role,
                status=status,
                raw_output=raw_output,
                skill_results=skill_results,
                evidence_entries=shared._harness_evidence_for_worker(task.worker_id),
                aspect_ids=task.aspect_ids,
            )
            if not artifact.answer_fragment and answer:
                artifact = replace(
                    artifact,
                    answer_fragment=answer,
                    findings=artifact.findings or (answer[:1200],),
                )
            elif artifact.answer_fragment:
                answer = artifact.answer_fragment
        source_ids = tuple(citation.source_id for citation in citations)
        report = WorkerReport(
            worker_id=task.worker_id,
            role=role,
            objective=task.objective,
            status=status,
            answer_text=answer,
            observed_facts=tuple(dict.fromkeys([
                *_VALUE.findall(answer),
                *(token for fact in (artifact.facts if artifact else ())
                  for token in _VALUE.findall(fact.text)),
            ])),
            observed_evidence_ids=source_ids,
            citations=citations,
            tool_calls=sum(result.tool_calls for result in skill_results),
            provider_calls=call_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(monotonic() - started) * 1000,
            error=error,
            model_turns=model_turns,
            raw_output=raw_output,
            artifact=artifact,
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


def _mvp2_answer_fragment(text: str) -> str:
    cleaned = _strip_nonanswer(text)
    try:
        payload = json.loads(cleaned)
    except (TypeError, json.JSONDecodeError):
        return "" if cleaned.startswith(("{", "[")) else cleaned
    if not isinstance(payload, dict) or set(payload) != {
        "findings", "facts", "unresolved", "answer_fragment",
    }:
        return ""
    answer = payload.get("answer_fragment")
    if (not isinstance(payload.get("findings"), list)
            or not isinstance(payload.get("facts"), list)
            or not isinstance(payload.get("unresolved"), list)
            or not isinstance(answer, str)):
        return ""
    if any(not isinstance(item, str) for key in ("findings", "facts", "unresolved")
           for item in payload[key]):
        return ""
    return answer.strip()


def _skill_fallback(results: list[SkillResult]) -> str:
    pieces = [result.output for result in results if result.output]
    if pieces:
        return " ".join(pieces)
    return ""
