"""Single top-level execution path composing safety, memory, retrieval, and reasoning."""

from __future__ import annotations

import asyncio
import copy
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from time import monotonic
from typing import Any

from ..providers.memory import MemoryProvider, MemoryResult
from ..providers.model import ModelProvider, ModelReply, ModelRequest
from ..providers.retrieval import RetrievalProvider, RetrievalResult
from ..reasoning.adaptive_mdt import AdaptiveMDTReasoner
from ..reasoning.base import ReasoningContext, ReasoningResult, ReasoningStrategy
from ..reasoning.single import SingleReasoner
from ..safety import route_question
from .budget import BudgetExceeded, BudgetLedger, BudgetLimits
from .contracts import (
    HarnessCitation,
    HarnessRequest,
    HarnessResponse,
    RuntimeResources,
)
from .profiles import MemoryMode, ModelVariant, ReasoningMode, RetrievalMode, SystemProfile
from .trace import ExecutionTrace, query_fingerprint
from .verification import parse_answer

ReasonerFactory = Callable[[ModelProvider], ReasoningStrategy]


@dataclass(frozen=True)
class HarnessConfig:
    budget: BudgetLimits = field(default_factory=BudgetLimits)
    abstention_message: str = "目前无法可靠完成这项分析，请由有资质的临床人员复核。"


class _BudgetedModelProvider:
    def __init__(self, provider: ModelProvider, ledger: BudgetLedger, trace: ExecutionTrace) -> None:
        self.provider = provider
        self.ledger = ledger
        self.trace = trace

    async def complete(self, request: ModelRequest) -> ModelReply:
        timeout, output_cap = self.ledger.reserve_provider(request.max_output_tokens)
        if request.timeout_seconds > timeout or request.max_output_tokens > output_cap:
            request = ModelRequest(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_output_tokens=output_cap,
                timeout_seconds=min(request.timeout_seconds, timeout),
                json_mode=request.json_mode,
                json_schema=request.json_schema,
            )
        self.trace.emit("model_call_started", model=request.model or "provider_default")
        try:
            reply = await asyncio.wait_for(self.provider.complete(request), timeout=timeout)
        except Exception as exc:
            self.ledger.release_output_reservation(output_cap)
            self.trace.emit("model_call_failed", error_code=type(exc).__name__)
            raise
        try:
            self.ledger.record_usage(
                reply.input_tokens,
                reply.output_tokens,
                reserved_output_tokens=output_cap,
            )
        except BudgetExceeded as exc:
            self.trace.emit("model_call_budget_exceeded", reason=str(exc))
            raise
        self.trace.emit(
            "model_call_completed",
            model=reply.model,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            latency_ms=round(reply.latency_ms, 3),
        )
        return reply


class HealthCopilotHarness:
    """Production and evaluation share this request → context → reason → verify path."""

    def __init__(
        self,
        *,
        model_providers: Mapping[ModelVariant, ModelProvider],
        retrieval_provider: RetrievalProvider | None = None,
        memory_provider: MemoryProvider | None = None,
        reasoner_factories: Mapping[ReasoningMode, ReasonerFactory] | None = None,
        config: HarnessConfig | None = None,
        trace_sink: Callable[[ExecutionTrace], None] | None = None,
    ) -> None:
        self.model_providers = dict(model_providers)
        self.retrieval_provider = retrieval_provider
        self.memory_provider = memory_provider
        self.reasoner_factories = dict(reasoner_factories or {
            ReasoningMode.SINGLE: SingleReasoner,
            ReasoningMode.ADAPTIVE_MDT: AdaptiveMDTReasoner,
        })
        self.config = config or HarnessConfig()
        self.trace_sink = trace_sink
        self._metrics_lock = RLock()
        self._metric_rows: deque[dict[str, Any]] = deque(maxlen=4096)
        self._metric_totals: dict[str, Any] = {
            "requests": 0,
            "abstentions": 0,
            "safety_routes": 0,
            "provider_calls": 0,
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "input_tokens_known": True,
            "output_tokens_known": True,
            "reasoning_modes": {},
        }
        self._trace_store: OrderedDict[str, dict[str, Any]] = OrderedDict()

    async def execute(self, profile: SystemProfile, request: HarnessRequest) -> HarnessResponse:
        started = monotonic()
        trace = ExecutionTrace(
            query_sha256=query_fingerprint(request.query),
            profile_id=profile.profile_id,
        )
        trace.emit("request_started", request_id_sha256=query_fingerprint(request.request_id))
        limits = _request_limits(self.config.budget, request.runtime_resources)
        ledger = BudgetLedger(limits)
        memory_result = MemoryResult()
        retrieval_result = RetrievalResult(evidence=())
        safety_flags: list[str] = []
        try:
            safety = route_question(request.query)
            if safety is not None:
                trace.emit("safety_routed", reason_codes=list(safety.safety_reasons))
                return self._response(
                    answer_text=safety.message,
                    request=request,
                    profile=profile,
                    trace=trace,
                    ledger=ledger,
                    started=started,
                    safety_flags=tuple(safety.safety_reasons),
                    citations=(),
                    final_status="safety_routed",
                )

            if profile.memory_mode is MemoryMode.READ:
                if self.memory_provider is None:
                    raise RuntimeError("memory_provider_not_configured")
                if request.subject_id is None or request.as_of_time is None:
                    raise RuntimeError("memory_context_identity_missing")
                timeout = ledger.reserve_tool()
                trace.emit("tool_call_started", tool="memory_read")
                try:
                    memory_result = await asyncio.wait_for(
                        self.memory_provider.read(
                            subject_id=request.subject_id,
                            query=request.query,
                            as_of_time=request.as_of_time,
                        ),
                        timeout=timeout,
                    )
                except Exception as exc:
                    trace.emit("tool_call_failed", tool="memory_read", error_code=type(exc).__name__)
                    raise
                trace.emit("memory_read_completed", facts=len(memory_result.facts))
                trace.emit("tool_call_completed", tool="memory_read")

            if profile.retrieval_mode is RetrievalMode.STANDARD:
                if self.retrieval_provider is None:
                    raise RuntimeError("common_medical_kb_not_ready")
                timeout = ledger.reserve_tool()
                trace.emit("tool_call_started", tool="external_retrieval")
                try:
                    retrieval_result = await asyncio.wait_for(
                        self.retrieval_provider.retrieve(
                            query=request.query,
                            context=_RuntimeContext(
                                request_id=request.request_id,
                                subject_id=request.subject_id,
                                as_of_time=request.as_of_time,
                            ),
                        ),
                        timeout=timeout,
                    )
                except Exception as exc:
                    trace.emit(
                        "tool_call_failed", tool="external_retrieval", error_code=type(exc).__name__,
                    )
                    raise
                trace.emit(
                    "retrieval_completed",
                    corpus_id=retrieval_result.corpus_id,
                    index_hash=retrieval_result.index_hash,
                    evidence_sha256=retrieval_result.evidence_sha256,
                    evidence_count=len(retrieval_result.evidence),
                )
                trace.emit("tool_call_completed", tool="external_retrieval")

            try:
                provider = self.model_providers[profile.model_variant]
                factory = self.reasoner_factories[profile.reasoning_mode]
            except KeyError as exc:
                raise RuntimeError(f"harness_component_not_configured:{exc.args[0]}") from None
            metered = _BudgetedModelProvider(provider, ledger, trace)
            strategy = factory(metered)
            context = ReasoningContext(
                query=request.query,
                conversation_context=request.conversation_context,
                patient_state=tuple(fact.text for fact in memory_result.facts),
                external_evidence=retrieval_result.evidence,
                answer_schema=request.answer_schema,
                runtime_metadata={
                    "request_id": request.request_id,
                    "profile_id": profile.profile_id,
                    "model_variant": profile.model_variant.value,
                },
            )
            trace.emit("reasoning_started", strategy=profile.reasoning_mode.value)
            result = await strategy.reason(context)
            for event in result.reasoning_events:
                trace.emit("reasoning_detail", **dict(event))
            if result.failure_reason is not None:
                failure_flag = f"reasoning_failure:{result.failure_reason}"
                safety_flags.append(failure_flag)
                trace.emit("reasoning_failed", error_code=result.failure_reason)
                return self._response(
                    answer_text=self.config.abstention_message,
                    request=request,
                    profile=profile,
                    trace=trace,
                    ledger=ledger,
                    started=started,
                    safety_flags=tuple(safety_flags),
                    citations=(),
                    final_status="abstained",
                )
            trace.emit("reasoning_completed", strategy=profile.reasoning_mode.value)
            citations = _verified_citations(result, retrieval_result, memory_result)
            known_ids = {item.evidence_id for item in retrieval_result.evidence}
            known_ids.update(item.fact_id for item in memory_result.facts)
            unverified_ids = set(result.citation_ids) - known_ids
            answer_text = result.answer_text
            for evidence_id in sorted(unverified_ids, key=len, reverse=True):
                answer_text = answer_text.replace(f"[{evidence_id}]", "")
            citation_flags = ("unverified_citation_removed",) if unverified_ids else ()
            return self._response(
                answer_text=answer_text,
                request=request,
                profile=profile,
                trace=trace,
                ledger=ledger,
                started=started,
                safety_flags=(*safety_flags, *result.safety_flags, *citation_flags),
                citations=citations,
                final_status="complete",
            )
        except BudgetExceeded as exc:
            safety_flags.append(str(exc))
            trace.emit("budget_denied", reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - fail closed without returning provider internals
            safety_flags.append(f"harness_failure:{type(exc).__name__}")
            trace.emit("harness_failed", error_code=type(exc).__name__)
        return self._response(
            answer_text=self.config.abstention_message,
            request=request,
            profile=profile,
            trace=trace,
            ledger=ledger,
            started=started,
            safety_flags=tuple(safety_flags),
            citations=(),
            final_status="abstained",
        )

    def _response(
        self,
        *,
        answer_text: str,
        request: HarnessRequest,
        profile: SystemProfile,
        trace: ExecutionTrace,
        ledger: BudgetLedger,
        started: float,
        safety_flags: tuple[str, ...],
        citations: tuple[HarnessCitation, ...],
        final_status: str,
    ) -> HarnessResponse:
        elapsed = (monotonic() - started) * 1000
        trace.emit(
            "request_completed",
            status=final_status,
            answer_sha256=query_fingerprint(answer_text),
            latency_ms=round(elapsed, 3),
        )
        if self.trace_sink is not None:
            try:
                self.trace_sink(trace)
            except Exception as exc:  # noqa: BLE001 - observability must not change the answer
                trace.emit("trace_sink_failed", error_code=type(exc).__name__)
        with self._metrics_lock:
            self._trace_store[trace.trace_id] = trace.to_dict()
            self._trace_store.move_to_end(trace.trace_id)
            while len(self._trace_store) > 2048:
                self._trace_store.popitem(last=False)
        response = HarnessResponse(
            answer_text=answer_text,
            parsed_answer=parse_answer(answer_text, request.answer_schema),
            model_variant=profile.model_variant.value,
            system_profile=profile.profile_id,
            retrieval_mode=profile.retrieval_mode.value,
            memory_mode=profile.memory_mode.value,
            reasoning_mode=profile.reasoning_mode.value,
            citations=citations,
            safety_flags=safety_flags,
            trace_id=trace.trace_id,
            provider_calls=ledger.provider_calls,
            tool_calls=ledger.tool_calls,
            input_tokens=ledger.input_tokens,
            output_tokens=ledger.output_tokens,
            latency_ms=elapsed,
        )
        with self._metrics_lock:
            row = {
                "latency_ms": elapsed,
                "provider_calls": response.provider_calls,
                "tool_calls": response.tool_calls,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "abstained": final_status == "abstained",
                "safety_routed": final_status == "safety_routed",
                "reasoning_mode": response.reasoning_mode,
            }
            self._metric_rows.append(row)
            totals = self._metric_totals
            totals["requests"] += 1
            totals["abstentions"] += int(row["abstained"])
            totals["safety_routes"] += int(row["safety_routed"])
            totals["provider_calls"] += response.provider_calls
            totals["tool_calls"] += response.tool_calls
            totals["reasoning_modes"][response.reasoning_mode] = (
                totals["reasoning_modes"].get(response.reasoning_mode, 0) + 1
            )
            for name in ("input_tokens", "output_tokens"):
                usage = getattr(response, name)
                if usage is None:
                    totals[f"{name}_known"] = False
                elif totals[f"{name}_known"]:
                    totals[name] += usage
        return response

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._metrics_lock:
            rows = list(self._metric_rows)
            totals = copy.deepcopy(self._metric_totals)
        latency = sorted(float(item["latency_ms"]) for item in rows)
        sample_count = len(rows)

        def percentile(p: float) -> float:
            if not latency:
                return 0.0
            index = min(len(latency) - 1, int((len(latency) - 1) * p))
            return round(latency[index], 3)

        return {
            "requests": totals["requests"],
            "latency_samples": sample_count,
            "abstentions": totals["abstentions"],
            "safety_routes": totals["safety_routes"],
            "avg_latency_ms_recent": round(sum(latency) / sample_count, 3) if sample_count else 0.0,
            "p50_latency_ms": percentile(0.50),
            "p95_latency_ms": percentile(0.95),
            "provider_calls": totals["provider_calls"],
            "tool_calls": totals["tool_calls"],
            "input_tokens": totals["input_tokens"] if totals["input_tokens_known"] else None,
            "output_tokens": totals["output_tokens"] if totals["output_tokens_known"] else None,
            "reasoning_modes": dict(sorted(totals["reasoning_modes"].items())),
        }

    def trace_for(self, trace_id: str) -> dict[str, Any] | None:
        """Return a bounded metadata-only trace for the API compatibility adapter."""
        with self._metrics_lock:
            value = self._trace_store.get(trace_id)
            return copy.deepcopy(value) if value is not None else None


@dataclass(frozen=True)
class _RuntimeContext:
    request_id: str
    subject_id: str | None
    as_of_time: Any


def _request_limits(base: BudgetLimits, resources: RuntimeResources | None) -> BudgetLimits:
    if resources is None:
        return base
    return BudgetLimits(
        max_provider_calls=min(base.max_provider_calls, resources.max_provider_calls)
        if resources.max_provider_calls is not None else base.max_provider_calls,
        max_tool_calls=min(base.max_tool_calls, resources.max_tool_calls)
        if resources.max_tool_calls is not None else base.max_tool_calls,
        max_input_tokens=_minimum_limit(base.max_input_tokens, resources.max_input_tokens),
        max_output_tokens=_minimum_limit(base.max_output_tokens, resources.max_output_tokens),
        deadline_ms=min(base.deadline_ms, resources.deadline_ms)
        if resources.deadline_ms is not None else base.deadline_ms,
    )


def _minimum_limit(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _verified_citations(
    result: ReasoningResult,
    retrieval: RetrievalResult,
    memory: MemoryResult,
) -> tuple[HarnessCitation, ...]:
    observed: dict[str, HarnessCitation] = {
        item.evidence_id: HarnessCitation(
            evidence_id=item.evidence_id,
            source_id=item.source_id,
            source=item.source,
            excerpt=item.excerpt,
            score=item.score,
        )
        for item in retrieval.evidence
    }
    observed.update({
        fact.fact_id: HarnessCitation(
            evidence_id=fact.fact_id,
            source_id=fact.source_id or fact.fact_id,
            source="patient_memory",
            excerpt=fact.text,
        )
        for fact in memory.facts
    })
    return tuple(observed[item] for item in dict.fromkeys(result.citation_ids) if item in observed)
