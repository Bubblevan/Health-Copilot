"""Recording and strict replay executors for public evaluation artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..contracts import Evidence
from .budget import BudgetDenied
from .context import RunContext
from .provider import (
    ProviderExecutor,
    ProviderFailure,
    ProviderFailureKind,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    provider_request_fingerprint,
)

if TYPE_CHECKING:
    from ..agent.messages import ToolCall
    from ..agent.tools import ToolResult


@dataclass(frozen=True)
class RecordedProviderExchange:
    request: ProviderRequest
    response: ProviderResponse
    request_fingerprint: str


class RecordingProviderExecutor:
    """Records successful provider exchanges while preserving one delegated call."""

    def __init__(self, delegate: ProviderExecutor) -> None:
        self.delegate = delegate
        self.exchanges: list[RecordedProviderExchange] = []

    def execute(self, request: ProviderRequest, runtime: RunContext) -> ProviderResponse:
        response = self.delegate.execute(request, runtime)
        self.exchanges.append(
            RecordedProviderExchange(request, response, provider_request_fingerprint(request))
        )
        return response


class ReplayProviderExecutor:
    """Returns a recorded response only when the full request contract matches."""

    def __init__(self, exchanges: Sequence[RecordedProviderExchange]) -> None:
        self._exchanges = list(exchanges)
        self.requests: list[ProviderRequest] = []

    def execute(self, request: ProviderRequest, runtime: RunContext) -> ProviderResponse:
        try:
            runtime.budget.guard_provider()
        except BudgetDenied as exc:
            raise ProviderFailure(ProviderFailureKind.BUDGET_DENIED) from exc
        self.requests.append(request)
        if not self._exchanges:
            raise ProviderFailure(ProviderFailureKind.REPLAY_MISMATCH)
        exchange = self._exchanges.pop(0)
        if provider_request_fingerprint(request) != exchange.request_fingerprint:
            raise ProviderFailure(ProviderFailureKind.REPLAY_MISMATCH)
        response = exchange.response
        normalized = ProviderResponse(
            call_id=request.call_id,
            kind=request.kind,
            model=request.model,
            content=response.content,
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason,
            usage=response.usage,
            latency_ms=response.latency_ms,
            provider_request_id=response.provider_request_id,
        )
        runtime.budget.record_usage(normalized.usage)
        return normalized

    @property
    def remaining_exchanges(self) -> int:
        return len(self._exchanges)


@dataclass(frozen=True)
class RecordedToolExchange:
    tool_name: str
    arguments: object
    result: ToolResult


class RecordingToolRunner:
    """Records tool outputs from a delegated runner; it does not alter dispatch."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.exchanges: list[RecordedToolExchange] = []

    def execute(self, call: ToolCall, runtime: RunContext) -> ToolResult:
        result = self.delegate.execute(call, runtime)
        self.exchanges.append(RecordedToolExchange(call.name, call.arguments, result))
        return result


class ReplayToolRunner:
    """Serves recorded tool observations and never owns a live registry or network client."""

    def __init__(self, exchanges: Sequence[RecordedToolExchange]) -> None:
        self._exchanges = list(exchanges)
        self.calls: list[ToolCall] = []

    def execute(self, call: ToolCall, runtime: RunContext) -> ToolResult:
        from ..agent.tools import ToolResult

        try:
            runtime.budget.guard_tool()
        except BudgetDenied as exc:
            return ToolResult.failure("runtime_budget_denied", str(exc))
        self.calls.append(call)
        if not self._exchanges:
            return ToolResult.failure("replay_mismatch", "no recorded tool exchange remains")
        exchange = self._exchanges.pop(0)
        if exchange.tool_name != call.name or _canonical_json(exchange.arguments) != _canonical_json(
            call.arguments
        ):
            return ToolResult.failure("replay_mismatch", "recorded tool exchange does not match")
        return exchange.result

    @property
    def remaining_exchanges(self) -> int:
        return len(self._exchanges)


def write_provider_exchanges(path: Path, exchanges: Sequence[RecordedProviderExchange]) -> None:
    _write_jsonl(path, (_provider_exchange_to_dict(exchange) for exchange in exchanges))


def read_provider_exchanges(path: Path) -> list[RecordedProviderExchange]:
    return [_provider_exchange_from_dict(item) for item in _read_jsonl(path)]


def write_tool_exchanges(path: Path, exchanges: Sequence[RecordedToolExchange]) -> None:
    _write_jsonl(path, (_tool_exchange_to_dict(exchange) for exchange in exchanges))


def read_tool_exchanges(path: Path) -> list[RecordedToolExchange]:
    return [_tool_exchange_from_dict(item) for item in _read_jsonl(path)]


def _provider_exchange_to_dict(exchange: RecordedProviderExchange) -> dict[str, Any]:
    return {
        "request": {
            "call_id": exchange.request.call_id,
            "kind": exchange.request.kind.value,
            "model": exchange.request.model,
            "messages": list(exchange.request.messages),
            "tools": list(exchange.request.tools),
            "response_format": exchange.request.response_format,
            "temperature": exchange.request.temperature,
            "max_output_tokens": exchange.request.max_output_tokens,
            "timeout_seconds": exchange.request.timeout_seconds,
            "metadata": dict(exchange.request.metadata),
        },
        "response": {
            "call_id": exchange.response.call_id,
            "kind": exchange.response.kind.value,
            "model": exchange.response.model,
            "content": exchange.response.content,
            "tool_calls": list(exchange.response.tool_calls),
            "finish_reason": exchange.response.finish_reason,
            "usage": asdict(exchange.response.usage) if exchange.response.usage else None,
            "latency_ms": exchange.response.latency_ms,
            "provider_request_id": exchange.response.provider_request_id,
        },
        "request_fingerprint": exchange.request_fingerprint,
    }


def _provider_exchange_from_dict(item: Mapping[str, Any]) -> RecordedProviderExchange:
    request_raw = item["request"]
    response_raw = item["response"]
    request = ProviderRequest(
        call_id=request_raw["call_id"],
        kind=request_raw["kind"],
        model=request_raw["model"],
        messages=tuple(request_raw["messages"]),
        tools=tuple(request_raw.get("tools", [])),
        response_format=request_raw.get("response_format"),
        temperature=request_raw.get("temperature"),
        max_output_tokens=request_raw.get("max_output_tokens"),
        timeout_seconds=request_raw.get("timeout_seconds"),
        metadata=request_raw.get("metadata", {}),
    )
    usage_raw = response_raw.get("usage")
    usage = ProviderUsage(**usage_raw) if usage_raw is not None else None
    response = ProviderResponse(
        call_id=response_raw["call_id"],
        kind=response_raw["kind"],
        model=response_raw["model"],
        content=response_raw.get("content"),
        tool_calls=tuple(response_raw.get("tool_calls", [])),
        finish_reason=response_raw.get("finish_reason"),
        usage=usage,
        latency_ms=response_raw.get("latency_ms"),
        provider_request_id=response_raw.get("provider_request_id"),
    )
    return RecordedProviderExchange(request, response, item["request_fingerprint"])


def _tool_exchange_to_dict(exchange: RecordedToolExchange) -> dict[str, Any]:
    result = exchange.result
    return {
        "tool_name": exchange.tool_name,
        "arguments": exchange.arguments,
        "result": {
            "ok": result.ok,
            "data": result.data,
            "error": asdict(result.error) if result.error else None,
            "observed_evidence": [asdict(item) for item in result.observed_evidence],
        },
    }


def _tool_exchange_from_dict(item: Mapping[str, Any]) -> RecordedToolExchange:
    from ..agent.tools import ToolError, ToolResult

    result_raw = item["result"]
    error_raw = result_raw.get("error")
    return RecordedToolExchange(
        tool_name=item["tool_name"],
        arguments=item["arguments"],
        result=ToolResult(
            ok=result_raw["ok"],
            data=result_raw.get("data"),
            error=ToolError(**error_raw) if error_raw else None,
            observed_evidence=tuple(
                Evidence(**evidence) for evidence in result_raw.get("observed_evidence", [])
            ),
        ),
    )


def _write_jsonl(path: Path, items: Sequence[Mapping[str, Any]] | Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            # Provider messages are opaque strings. Preserve nested mapping insertion
            # order so a replay reconstructs the exact next-turn prompt bytes.
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
