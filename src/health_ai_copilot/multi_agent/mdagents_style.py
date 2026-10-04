"""Opt-in MDAgents-style clinical reasoning behind the existing harness boundary.

This module implements the algorithmic pattern independently; it does not import
or vendor the pinned MDAgents repository.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from time import monotonic
from typing import Any
from uuid import uuid4

from ..runtime.budget import BudgetDenied, RunBudgetConfig, RunBudgetState
from ..runtime.provider import ProviderUsage
from ..runtime.trace import RunTrace, TraceEvent, TraceEventType
from ..safety import route_question
from .contracts import MedicalAgentRequest, MedicalAgentResponse, RouteMode, WorkerRole
from .providers import ModelProvider, ModelReply
from .shared_context import SharedContext
from .skills import SkillContext, SkillRegistry


class Complexity(StrEnum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    SAFETY_ROUTED = "safety_routed"
    FAILED = "failed"


@dataclass(frozen=True)
class MDAgentsStyleConfig:
    model_name: str = "qwen3-8b-local"
    max_output_tokens: int = 1024
    max_provider_calls: int = 32
    max_tool_executions: int = 2
    max_total_tokens: int = 80_000
    deadline_ms: float = 20 * 60 * 1000
    provider_timeout_seconds: float = 180.0
    max_intermediate_specialists: int = 5
    max_advanced_teams: int = 2
    max_specialists_per_team: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.max_provider_calls <= 64:
            raise ValueError("max_provider_calls must be between 1 and 64")
        if not 1 <= self.max_intermediate_specialists <= 8:
            raise ValueError("max_intermediate_specialists must be between 1 and 8")
        if not 1 <= self.max_advanced_teams <= 4:
            raise ValueError("max_advanced_teams must be between 1 and 4")
        if not 1 <= self.max_specialists_per_team <= 5:
            raise ValueError("max_specialists_per_team must be between 1 and 5")


@dataclass(frozen=True)
class MDAgentsStyleExecution:
    response: MedicalAgentResponse
    complexity: Complexity
    parsed_option: str | None
    raw_model_output: str
    provider_calls: int
    tool_calls: int
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    failure_reason: str | None
    trace_events: tuple[TraceEvent, ...]
    shared_context: dict[str, Any] = field(default_factory=dict)


class ClinicalReasoningSkill:
    """Product-facing opt-in facade for the adaptive clinical reasoning sub-runtime."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        skill_registry: SkillRegistry | None = None,
        config: MDAgentsStyleConfig | None = None,
    ) -> None:
        self.orchestrator = MDAgentsStyleOrchestrator(
            provider, skill_registry=skill_registry, config=config,
        )

    async def run(self, request: MedicalAgentRequest) -> MDAgentsStyleExecution:
        return await self.orchestrator.run(request)

    async def execute(self, request: MedicalAgentRequest) -> MedicalAgentResponse:
        return (await self.run(request)).response


class MDAgentsStyleOrchestrator:
    """Complexity triage, dynamic specialists, bounded collaboration, and moderation."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        skill_registry: SkillRegistry | None = None,
        config: MDAgentsStyleConfig | None = None,
    ) -> None:
        self.provider = provider
        self.skills = skill_registry or SkillRegistry()
        self.config = config or MDAgentsStyleConfig()

    async def run(self, request: MedicalAgentRequest) -> MDAgentsStyleExecution:
        started = monotonic()
        trace_id = request.request_id or f"mdagents-{uuid4().hex}"

        # Keep the existing deterministic safety gate ahead of any mutable harness state.
        safety = route_question(request.query)
        if safety is not None:
            route = Complexity.SAFETY_ROUTED
            response = MedicalAgentResponse(
                answer=safety.message,
                route_mode=RouteMode.SINGLE,
                workers_used=(),
                citations=(),
                safety_flags=tuple(safety.safety_reasons),
                trace_id=trace_id,
                latency_ms=max(0, int((monotonic() - started) * 1000)),
            )
            return MDAgentsStyleExecution(
                response, route, None, "", 0, 0, 0, 0,
                (monotonic() - started) * 1000, "safety_routed", (),
            )

        shared = SharedContext(trace_id, request.query)
        trace = RunTrace()
        budget = RunBudgetState(RunBudgetConfig(
            max_provider_calls=self.config.max_provider_calls,
            max_tool_executions=self.config.max_tool_executions,
            deadline_ms=self.config.deadline_ms,
            max_total_tokens=self.config.max_total_tokens,
        ))
        query_hash = _digest(request.query)
        trace.emit(
            TraceEventType.RUN_START,
            run_id=trace_id,
            strategy="mdagents_style_v1",
            query_sha256=query_hash,
        )
        shared._harness_timeline("request_started", query_sha256=query_hash)
        complexity = Complexity.FAILED
        final_reply: ModelReply | None = None
        failure_reason: str | None = None
        workers: list[str] = []
        safety_flags: set[str] = set()

        try:
            harness_observations, harness_tool_calls = await self._run_harness_skills(
                request, shared, budget, trace, safety_flags,
            )
            question = _format_question(request, harness_observations)
            classifier_instruction = (
                "You are a medical expert who conducts initial assessment and your job is "
                "to decide the difficulty/complexity of the medical query."
            )
            classifier_initial_reply = await self._call(
                stage="complexity_classifier_init",
                system_prompt=classifier_instruction,
                user_prompt=classifier_instruction,
                budget=budget, trace=trace, shared=shared,
                max_output_tokens=self.config.max_output_tokens,
            )
            difficulty_prompt = (
                "Now, given the medical query as below, you need to decide the "
                "difficulty/complexity of it:\n"
                f"{question}.\n\n"
                "Please indicate the difficulty/complexity of the medical query among below options:\n"
                "1) basic: a single medical agent can output an answer.\n"
                "2) intermediate: number of medical experts with different expertise should dicuss "
                "and make final decision.\n"
                "3) advanced: multiple teams of clinicians from different departments need to "
                "collaborate with each other to make final decision."
            )
            classifier_history = [
                {"role": "system", "content": classifier_instruction},
                {"role": "user", "content": classifier_instruction},
                {"role": "assistant", "content": classifier_initial_reply.content},
                {"role": "user", "content": difficulty_prompt},
            ]
            triage = await self._call(
                stage="complexity_classifier",
                system_prompt=classifier_instruction,
                user_prompt=difficulty_prompt,
                messages=classifier_history,
                budget=budget, trace=trace, shared=shared,
                max_output_tokens=self.config.max_output_tokens,
            )
            complexity = _parse_complexity(triage.content)
            shared._harness_timeline(
                "complexity_decided", complexity=complexity.value,
                provider_call=budget.provider_calls_used,
            )
            trace.emit(TraceEventType.POLICY_DECISION, decision=complexity.value)

            if complexity is Complexity.BASIC:
                final_reply = await self._single(question, budget, trace, shared)
                workers.append("strong_single")
                mode = RouteMode.SINGLE
            elif complexity is Complexity.INTERMEDIATE:
                final_reply, specialists = await self._intermediate(
                    question, budget, trace, shared,
                )
                workers.extend(specialists)
                mode = RouteMode.TEAM
            else:
                final_reply, team_members = await self._advanced(
                    question, budget, trace, shared,
                )
                workers.extend(team_members)
                mode = RouteMode.TEAM
        except BudgetDenied as exc:
            failure_reason = str(exc)
            trace.emit(TraceEventType.BUDGET_DENIED, reason=failure_reason)
            mode = RouteMode.SINGLE if complexity is Complexity.BASIC else RouteMode.TEAM
        except Exception as exc:  # noqa: BLE001 - failures return a bounded, safe response
            failure_reason = _failure_code(exc)
            trace.emit(
                TraceEventType.HARNESS_DISPOSITION,
                disposition="abstain",
                reason=failure_reason,
            )
            mode = RouteMode.SINGLE if complexity is Complexity.BASIC else RouteMode.TEAM

        elapsed_ms = (monotonic() - started) * 1000
        raw_output = final_reply.content if final_reply else ""
        parsed = parse_medqa_option(raw_output) if raw_output else None
        if final_reply and raw_output.strip():
            answer = raw_output.strip()
        else:
            answer = "目前无法可靠完成这项分析，请由有资质的临床人员复核。"
        if "urgent_symptom_requires_human_care" in safety_flags:
            answer += "\n\n如出现紧急症状，请立即联系当地急救或前往急诊。"
        response = MedicalAgentResponse(
            answer=answer,
            route_mode=mode,
            workers_used=tuple(workers),
            citations=tuple(shared._citations),
            safety_flags=tuple(sorted(safety_flags)),
            trace_id=trace_id,
            latency_ms=max(0, int(elapsed_ms)),
        )
        shared._harness_timeline(
            "request_completed", complexity=complexity.value,
            status="failed" if failure_reason else "complete",
        )
        trace.close(status="failed" if failure_reason else "complete")
        return MDAgentsStyleExecution(
            response=response,
            complexity=complexity,
            parsed_option=parsed,
            raw_model_output=raw_output,
            provider_calls=budget.provider_calls_used,
            tool_calls=harness_tool_calls,
            input_tokens=budget.input_tokens_used,
            output_tokens=budget.output_tokens_used,
            latency_ms=elapsed_ms,
            failure_reason=failure_reason,
            trace_events=tuple(trace.events),
            shared_context=shared.to_dict(),
        )

    async def _run_harness_skills(
        self,
        request: MedicalAgentRequest,
        shared: SharedContext,
        budget: RunBudgetState,
        trace: RunTrace,
        safety_flags: set[str],
    ) -> tuple[tuple[str, ...], int]:
        patient_id = request.patient_id or (
            request.episode.subject_id if request.episode else None
        )
        as_of_time = request.as_of_time or (
            request.episode.decision_time if request.episode else None
        )
        specs: list[tuple[WorkerRole, str]] = []
        if request.episode is not None and request.resources is not None:
            specs.extend((
                (WorkerRole.PATIENT_CONTEXT, "PatientStateLookupSkill"),
                (WorkerRole.EVIDENCE, "ExternalEvidenceSearchSkill"),
            ))
        else:
            if self.skills.memory_provider_configured:
                specs.append((WorkerRole.PATIENT_CONTEXT, "PatientStateLookupSkill"))
            if self.skills.external_evidence_provider_configured:
                specs.append((WorkerRole.EVIDENCE, "ExternalEvidenceSearchSkill"))
        specs.extend((
            (WorkerRole.CARE, "RiskAssessmentSkill"),
            (WorkerRole.CARE, "AnswerabilitySkill"),
        ))

        observations: list[str] = []
        tool_calls_total = 0
        for role, name in specs:
            task = shared._harness_register_task(role.value, f"MDAgents harness skill: {name}")
            shared._harness_worker_status(task.worker_id, "running", phase="harness_skill")
            if role in {WorkerRole.PATIENT_CONTEXT, WorkerRole.EVIDENCE}:
                budget.guard_tool()
            trace.emit(TraceEventType.TOOL_START, tool_name=name, role=role.value)
            context = SkillContext(
                query=request.query,
                role=role,
                episode=request.episode,
                resources=request.resources,
                observed_context=request.conversation_context,
                patient_id=patient_id,
                as_of_time=as_of_time,
            )
            result = await self.skills.execute(role, name, context)
            tool_calls_total += result.tool_calls
            trace.emit(
                TraceEventType.TOOL_END,
                tool_name=name,
                status="error" if result.error else "complete",
            )
            shared._harness_worker_status(
                task.worker_id,
                "failed" if result.error else "complete",
                error=result.error,
                output_sha256=_digest(result.output) if result.output else None,
            )
            shared._harness_timeline(
                "skill_call", worker_id=task.worker_id, role=role.value,
                skill=name, success=result.error is None,
            )
            if result.tool_calls:
                tool_call_id = f"tool-{uuid4().hex}"
                shared._harness_timeline(
                    "tool_call", worker_id=task.worker_id,
                    tool_call_id=tool_call_id, tool_id=result.tool_id,
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
            if name == "RiskAssessmentSkill" and (
                "urgent_symptom_requires_human_care" in result.output
            ):
                safety_flags.add("urgent_symptom_requires_human_care")
                shared._harness_risk_flag("urgent_symptom_requires_human_care")
            if role in {WorkerRole.PATIENT_CONTEXT, WorkerRole.EVIDENCE} and result.output:
                observations.append(f"{name}: {result.output[:5000]}")
        shared._harness_timeline(
            "harness_skills_complete", tool_calls=tool_calls_total,
        )
        return tuple(observations), tool_calls_total

    async def _single(
        self, question: str, budget: RunBudgetState, trace: RunTrace, shared: SharedContext,
    ) -> ModelReply:
        task = shared._harness_register_task("care", "Strong single-agent clinical reasoning")
        shared._harness_worker_status(task.worker_id, "running", phase="single")
        reply = await self._call(
            stage="strong_single",
            system_prompt=(
                "You are a careful medical exam answerer. Solve the question from the listed choices. "
                "Return the best option and a concise rationale; do not invent evidence or cite sources."
            ),
            user_prompt=f"Question:\n{question}",
            budget=budget, trace=trace, shared=shared,
            max_output_tokens=self.config.max_output_tokens,
        )
        shared._harness_worker_status(task.worker_id, "complete", output_sha256=_digest(reply.content))
        return reply

    async def _intermediate(
        self, question: str, budget: RunBudgetState, trace: RunTrace, shared: SharedContext,
    ) -> tuple[ModelReply, list[str]]:
        recruitment = await self._call_json(
            stage="dynamic_recruitment",
            system_prompt=(
                "Recruit a diverse set of medical exam specialists whose perspectives fit the question. "
                "Use 3 to 5 distinct specialty names and one narrow focus per specialist. "
                "Return JSON only: {\"specialists\":[{\"name\":\"...\",\"focus\":\"...\"}]}"
            ),
            user_prompt=f"Question:\n{question}",
            budget=budget, trace=trace, shared=shared,
            max_output_tokens=256,
        )
        specialists = _parse_specialists(recruitment.content, self.config.max_intermediate_specialists)
        tasks = [
            shared._harness_register_task("care", f"Specialist {name}: {focus}")
            for name, focus in specialists
        ]
        for task in tasks:
            shared._harness_worker_status(task.worker_id, "running", phase="independent_analysis")
        independent = await _gather_calls([
            self._call(
                stage="specialist_analysis",
                system_prompt=(
                    f"You are the {name} specialist. Focus only on: {focus}. "
                    "Analyze the given multiple-choice medical question independently. "
                    "Return the best option and a concise rationale."
                ),
                user_prompt=f"Question:\n{question}",
                budget=budget, trace=trace, shared=shared,
                max_output_tokens=256,
            )
            for name, focus in specialists
        ])
        _raise_first_error(independent)
        initial = [x for x in independent if isinstance(x, ModelReply)]
        for task, reply in zip(tasks, initial, strict=True):
            shared._harness_worker_status(task.worker_id, "complete", output_sha256=_digest(reply.content))
        peer_view = _format_peer_view(specialists, initial)
        revisions = await _gather_calls([
            self._call(
                stage="collaborative_revision",
                system_prompt=(
                    f"You are the {name} specialist in a collaborative medical team. "
                    "Review the peer analyses, resolve disagreements against the question, and revise "
                    "your answer if needed. Return the best option and concise rationale."
                ),
                user_prompt=f"Question:\n{question}\n\nIndependent peer analyses:\n{peer_view}",
                budget=budget, trace=trace, shared=shared,
                max_output_tokens=224,
            )
            for name, _focus in specialists
        ])
        _raise_first_error(revisions)
        revised = [x for x in revisions if isinstance(x, ModelReply)]
        final = await self._moderate(question, _format_peer_view(specialists, revised), budget, trace, shared)
        return final, [name for name, _ in specialists]

    async def _advanced(
        self, question: str, budget: RunBudgetState, trace: RunTrace, shared: SharedContext,
    ) -> tuple[ModelReply, list[str]]:
        recruitment = await self._call_json(
            stage="multi_team_recruitment",
            system_prompt=(
                "Create two independent medical expert teams with different complementary perspectives. "
                "Each team has 2 or 3 distinct specialists, each with a narrow focus. "
                "Return JSON only: {\"teams\":[{\"name\":\"...\","
                "\"specialists\":[{\"name\":\"...\",\"focus\":\"...\"}]}]}"
            ),
            user_prompt=f"Question:\n{question}",
            budget=budget, trace=trace, shared=shared,
            max_output_tokens=320,
        )
        teams = _parse_teams(
            recruitment.content,
            max_teams=self.config.max_advanced_teams,
            max_specialists=self.config.max_specialists_per_team,
        )
        all_names = [member[0] for _team, members in teams for member in members]
        tasks = {
            name: shared._harness_register_task("care", f"Advanced specialist {name}: {focus}")
            for team, members in teams for name, focus in members
        }
        for task in tasks.values():
            shared._harness_worker_status(task.worker_id, "running", phase="team_analysis")
        call_specs = [
            (team_name, name, focus)
            for team_name, members in teams for name, focus in members
        ]
        analyses = await _gather_calls([
            self._call(
                stage="team_specialist_analysis",
                system_prompt=(
                    f"You are {name}, a specialist in team {team_name}. Focus on {focus}. "
                    "Solve the question independently and report the best option with concise reasoning."
                ),
                user_prompt=f"Question:\n{question}",
                budget=budget, trace=trace, shared=shared,
                max_output_tokens=256,
            )
            for team_name, name, focus in call_specs
        ])
        _raise_first_error(analyses)
        replies = [x for x in analyses if isinstance(x, ModelReply)]
        for (_team_name, name, _focus), reply in zip(call_specs, replies, strict=True):
            shared._harness_worker_status(tasks[name].worker_id, "complete", output_sha256=_digest(reply.content))
        team_views: list[tuple[str, str]] = []
        cursor = 0
        for team_name, members in teams:
            count = len(members)
            view = _format_peer_view(members, replies[cursor:cursor + count])
            cursor += count
            lead = await self._call(
                stage="team_synthesis",
                system_prompt=(
                    f"You moderate medical reasoning team {team_name}. Reconcile your specialists' "
                    "different conclusions and produce one team answer with a concise rationale."
                ),
                user_prompt=f"Question:\n{question}\n\nTeam specialist analyses:\n{view}",
                budget=budget, trace=trace, shared=shared,
                max_output_tokens=256,
            )
            team_views.append((team_name, lead.content))
        joined = "\n\n".join(f"Team {name}:\n{text}" for name, text in team_views)
        final = await self._moderate(question, joined, budget, trace, shared)
        return final, all_names

    async def _moderate(
        self,
        question: str,
        discussion: str,
        budget: RunBudgetState,
        trace: RunTrace,
        shared: SharedContext,
    ) -> ModelReply:
        return await self._call(
            stage="final_moderator",
            system_prompt=(
                "You are the medical team moderator. Weigh the specialists' arguments against the "
                "question and listed choices, resolve conflicts, and give one best option. "
                "Return the option and a concise rationale; do not add unsupported facts."
            ),
            user_prompt=f"Question:\n{question}\n\nTeam discussion:\n{discussion}",
            budget=budget, trace=trace, shared=shared,
            max_output_tokens=self.config.max_output_tokens,
        )

    async def _call_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        budget: RunBudgetState,
        trace: RunTrace,
        shared: SharedContext,
        max_output_tokens: int,
    ) -> ModelReply:
        reply = await self._call(
            stage=stage,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            budget=budget,
            trace=trace,
            shared=shared,
            max_output_tokens=max_output_tokens,
            json_mode=True,
        )
        _json_object(reply.content)
        return reply

    async def _call(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        budget: RunBudgetState,
        trace: RunTrace,
        shared: SharedContext,
        max_output_tokens: int,
        json_mode: bool = False,
        messages: list[dict[str, str]] | None = None,
    ) -> ModelReply:
        call_id = f"{stage}-{uuid4().hex[:12]}"
        prompt_for_hash = (
            json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
            if messages is not None else system_prompt + "\n" + user_prompt
        )
        prompt_hash = _digest(prompt_for_hash)
        remaining = budget.guard_provider()
        timeout = self.config.provider_timeout_seconds
        if remaining is not None:
            timeout = min(timeout, remaining)
        trace.emit(
            TraceEventType.PROVIDER_START,
            call_id=call_id,
            stage=stage,
            prompt_sha256=prompt_hash,
            model=self.config.model_name,
            max_output_tokens=max_output_tokens,
        )
        call_started = monotonic()
        try:
            complete_messages = getattr(self.provider, "complete_messages", None)
            if messages is not None and callable(complete_messages):
                reply = await complete_messages(
                    messages=messages,
                    max_output_tokens=max_output_tokens,
                    timeout_seconds=timeout,
                    json_mode=json_mode,
                )
            elif messages is not None:
                flattened_history = "\n\n".join(
                    f"[{message['role']}]\n{message['content']}" for message in messages
                )
                reply = await self.provider.complete(
                    system_prompt=system_prompt,
                    user_prompt=flattened_history,
                    max_output_tokens=max_output_tokens,
                    timeout_seconds=timeout,
                    json_mode=json_mode,
                )
            else:
                reply = await self.provider.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=max_output_tokens,
                    timeout_seconds=timeout,
                    json_mode=json_mode,
                )
        except Exception as exc:
            budget.record_usage(None)
            failure = _failure_code(exc)
            trace.emit(TraceEventType.PROVIDER_ERROR, call_id=call_id, stage=stage, reason=failure)
            shared._harness_record_call(
                call_id=call_id, stage=stage, status="error", reason=failure,
                prompt_sha256=prompt_hash,
            )
            raise
        total = reply.input_tokens + reply.output_tokens
        budget.record_usage(ProviderUsage(
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            total_tokens=total,
        ))
        latency = (monotonic() - call_started) * 1000
        trace.emit(
            TraceEventType.PROVIDER_END,
            call_id=call_id,
            stage=stage,
            model=reply.model or self.config.model_name,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            response_sha256=_digest(reply.content),
            latency_ms=round(latency, 3),
        )
        shared._harness_record_call(
            call_id=call_id, stage=stage, status="complete",
            model=reply.model or self.config.model_name,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            latency_ms=round(latency, 3),
            prompt_sha256=prompt_hash,
            response_sha256=_digest(reply.content),
        )
        return reply


def parse_medqa_option(text: str) -> str | None:
    """Deterministically extract a final A-D option from a model response."""
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"```(?:json)?|```", "", cleaned, flags=re.IGNORECASE).strip()
    patterns = (
        r"(?:final\s+answer|answer|correct\s+option|best\s+option|option)\s*(?:is|:|=)?\s*[\(\[]?([A-D])\b",
        r"\b([A-D])\s*(?:is|would be)\s+the\s+(?:correct|best)\s+(?:answer|option)\b",
    )
    hits: list[tuple[int, str]] = []
    for pattern in patterns:
        hits.extend((match.start(), match.group(1).upper()) for match in re.finditer(pattern, cleaned, re.IGNORECASE))
    if hits:
        return max(hits)[1]
    tail = cleaned[-160:]
    matches = list(re.finditer(r"(?:^|[\n.!?])\s*[\(\[]?([A-D])[\)\]]?\s*[.!]?$", tail, re.IGNORECASE))
    return matches[-1].group(1).upper() if matches else None


def _parse_complexity(text: str) -> Complexity:
    normalized = text.casefold()
    # Match the pinned reference parser's label precedence and numeric fallbacks.
    if "basic" in normalized or "1)" in normalized:
        return Complexity.BASIC
    if "intermediate" in normalized or "2)" in normalized:
        return Complexity.INTERMEDIATE
    if "advanced" in normalized or "3)" in normalized:
        return Complexity.ADVANCED
    if re.fullmatch(r"\s*([123])[.)]?\s*", normalized):
        return {"1": Complexity.BASIC, "2": Complexity.INTERMEDIATE, "3": Complexity.ADVANCED}[
            normalized.strip().rstrip(".)")
        ]
    raise ValueError("invalid_complexity_output")


def _parse_specialists(text: str, maximum: int) -> list[tuple[str, str]]:
    payload = _json_object(text)
    raw = payload.get("specialists")
    if not isinstance(raw, list):
        raise TypeError("invalid_specialist_recruitment")
    chosen: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = _clean_label(item.get("name"))
        focus = _clean_label(item.get("focus"))
        if name and focus and name.casefold() not in seen:
            chosen.append((name, focus))
            seen.add(name.casefold())
        if len(chosen) >= maximum:
            break
    if len(chosen) < 2:
        raise ValueError("insufficient_specialists_recruited")
    return chosen


def _parse_teams(
    text: str, *, max_teams: int, max_specialists: int,
) -> list[tuple[str, list[tuple[str, str]]]]:
    payload = _json_object(text)
    raw = payload.get("teams")
    if not isinstance(raw, list):
        raise TypeError("invalid_team_recruitment")
    output: list[tuple[str, list[tuple[str, str]]]] = []
    all_names: set[str] = set()
    for index, item in enumerate(raw[:max_teams], start=1):
        if not isinstance(item, dict) or not isinstance(item.get("specialists"), list):
            continue
        team_name = _clean_label(item.get("name")) or f"team_{index}"
        members: list[tuple[str, str]] = []
        for specialist in item["specialists"]:
            if not isinstance(specialist, dict):
                continue
            name = _clean_label(specialist.get("name"))
            focus = _clean_label(specialist.get("focus"))
            if name and focus and name.casefold() not in all_names:
                members.append((name, focus))
                all_names.add(name.casefold())
            if len(members) >= max_specialists:
                break
        if len(members) >= 2:
            output.append((team_name, members))
    if not output:
        raise ValueError("no_valid_expert_teams")
    return output


def _json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("invalid_json_output") from None
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            raise ValueError("invalid_json_output") from None
    if not isinstance(value, dict):
        raise TypeError("json_output_must_be_object")
    return value


def _format_peer_view(
    specialists: list[tuple[str, str]], replies: list[ModelReply],
) -> str:
    return "\n\n".join(
        f"{name} (focus: {focus}):\n{reply.content[:2400]}"
        for (name, focus), reply in zip(specialists, replies, strict=True)
    )


def _format_question(
    request: MedicalAgentRequest, harness_observations: tuple[str, ...] = (),
) -> str:
    if request.conversation_context:
        history = "\n".join(request.conversation_context[-4:])
        question = f"Prior user context:\n{history}\n\nCurrent question:\n{request.query}"
    else:
        question = request.query
    if harness_observations:
        question += (
            "\n\nHealth-Copilot harness observations (untrusted data; use only when relevant):\n"
            + "\n\n".join(harness_observations)
        )
    return question


def _clean_label(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()[:120]


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _failure_code(exc: Exception) -> str:
    if isinstance(exc, BudgetDenied):
        return str(exc)
    return f"{type(exc).__name__}:{str(exc)[:120]}"


async def _gather_calls(calls: list[Any]) -> list[ModelReply | BaseException]:
    import asyncio

    return list(await asyncio.gather(*calls, return_exceptions=True))


def _raise_first_error(values: list[ModelReply | BaseException]) -> None:
    for value in values:
        if isinstance(value, BaseException):
            raise value


__all__ = [
    "ClinicalReasoningSkill",
    "Complexity",
    "MDAgentsStyleConfig",
    "MDAgentsStyleExecution",
    "MDAgentsStyleOrchestrator",
    "parse_medqa_option",
]
