"""SDK-backed MCP client boundary and ToolSpec adapter."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from ..agent.tools import ToolResult, ToolSpec
from ..contracts import Evidence
from ..runtime.trace import TraceEventType, canonical_json_sha256
from .permissions import PermissionDecision, PermissionGuard
from .sandbox import (
    SandboxBackend,
    SandboxFailure,
    SandboxFailureKind,
    SandboxProfile,
    ensure_sandbox_backend,
)

MCP_PROTOCOL_VERSION = "2026-07-28"
MCP_SDK_VERSION = "2.2.0"


class McpFailureKind(StrEnum):
    DISCOVERY = "discovery"
    PROTOCOL_VERSION = "protocol_version"
    TRANSPORT = "transport"
    CATALOG = "catalog"
    INVALID_TOOL = "invalid_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    CALL_ERROR = "call_error"
    AUTHORIZATION = "authorization"
    SANDBOX = "sandbox"


class McpFailure(RuntimeError):
    def __init__(
        self,
        kind: McpFailureKind,
        message: str = "MCP operation failed",
        *,
        sandbox_failure_kind: SandboxFailureKind | None = None,
    ) -> None:
        self.kind = McpFailureKind(kind)
        self.sandbox_failure_kind = (
            SandboxFailureKind(sandbox_failure_kind)
            if sandbox_failure_kind is not None
            else None
        )
        super().__init__(message)


@dataclass(frozen=True)
class McpServerConfig:
    server_id: str
    transport: str
    protocol_version: str = MCP_PROTOCOL_VERSION
    endpoint_identity: str = ""
    endpoint: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    auth_mode: str = "unauthenticated_local"
    sandbox_profile_id: str | None = None
    tool_namespace: str | None = None

    def __post_init__(self) -> None:
        if not self.server_id.strip():
            raise ValueError("server_id must be non-empty")
        if self.transport not in {"stdio", "streamable_http", "inproc_test"}:
            raise ValueError("unsupported MCP transport")
        if self.protocol_version != MCP_PROTOCOL_VERSION:
            raise ValueError("M9 requires MCP protocol 2026-07-28")
        if not self.endpoint_identity.strip():
            raise ValueError("endpoint_identity must be a safe identity, not a credential")
        if self.transport == "streamable_http" and not self.endpoint:
            raise ValueError("streamable_http requires endpoint")
        if self.transport == "streamable_http":
            parsed = urlparse(self.endpoint or "")
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("streamable_http endpoint must have an http(s) scheme and host")
            if parsed.username or parsed.password:
                raise ValueError("streamable_http endpoint must not contain credentials")
            try:
                port = parsed.port
            except ValueError as exc:
                raise ValueError("streamable_http endpoint has an invalid port") from exc
            if port is not None and not (0 < port <= 65535):
                raise ValueError("streamable_http endpoint has an invalid port")
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio requires command")
        object.__setattr__(self, "args", tuple(self.args))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "transport": self.transport,
            "protocol_version": self.protocol_version,
            "endpoint_identity": self.endpoint_identity,
            "auth_mode": self.auth_mode,
            "sandbox_profile_id": self.sandbox_profile_id,
            "tool_namespace": self.tool_namespace,
        }


@dataclass(frozen=True)
class McpServerIdentity:
    server_id: str
    transport: str
    protocol_version: str
    endpoint_identity: str
    auth_mode: str
    sandbox_profile_id: str | None
    server_info_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "transport": self.transport,
            "protocol_version": self.protocol_version,
            "endpoint_identity": self.endpoint_identity,
            "auth_mode": self.auth_mode,
            "sandbox_profile_id": self.sandbox_profile_id,
            "server_info_hash": self.server_info_hash,
        }


@dataclass(frozen=True)
class McpToolCatalogSnapshot:
    server_id: str
    protocol_version: str
    tools: tuple[ToolSpec, ...]
    catalog_hash: str
    ttl_ms: int
    cache_scope: str
    obtained_at: float

    @classmethod
    def create(
        cls,
        *,
        server_id: str,
        protocol_version: str,
        tools: Sequence[ToolSpec],
        ttl_ms: int,
        cache_scope: str,
    ) -> McpToolCatalogSnapshot:
        ordered = tuple(sorted(tools, key=lambda item: item.name))
        payload = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
            }
            for tool in ordered
        ]
        return cls(
            server_id,
            protocol_version,
            ordered,
            canonical_json_sha256(payload),
            max(0, int(ttl_ms)),
            cache_scope,
            time.time(),
        )

    def is_fresh(self, now: float | None = None) -> bool:
        if self.ttl_ms <= 0:
            return False
        return (now or time.time()) < self.obtained_at + self.ttl_ms / 1000


def _run_sync(operation):
    """Run one async SDK operation from the synchronous Health-Copilot boundary."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(operation())

    result: list[Any] = []
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(operation()))
        except Exception as exc:  # noqa: BLE001 - propagate active-loop worker failure
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def _dump_model(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("MCP SDK result is not a structured object")


class McpClient:
    """Small domain client; AgentLoop never receives SDK objects."""

    def __init__(
        self,
        config: McpServerConfig,
        target: Any | None = None,
        *,
        sandbox_backend: SandboxBackend | None = None,
        sandbox_profile: SandboxProfile | None = None,
    ) -> None:
        self.config = config
        self.target = target
        self.sandbox_backend = sandbox_backend
        self.sandbox_profile = sandbox_profile
        self.identity = McpServerIdentity(
            config.server_id,
            config.transport,
            config.protocol_version,
            config.endpoint_identity,
            config.auth_mode,
            config.sandbox_profile_id,
        )
        self._catalog: McpToolCatalogSnapshot | None = None
        self.discovery_calls = 0
        self.list_calls = 0
        self.tool_calls = 0

    def _sdk_target(self):
        if self.config.transport == "streamable_http":
            if not self.config.endpoint:
                raise McpFailure(McpFailureKind.TRANSPORT, "MCP HTTP endpoint is missing")
            return self.config.endpoint
        if self.config.transport == "stdio":
            if not self.config.command:
                raise McpFailure(McpFailureKind.TRANSPORT, "MCP stdio command is missing")
            try:
                from mcp.client.stdio import StdioServerParameters
            except ImportError as exc:  # pragma: no cover - dependency is pinned in pyproject
                raise McpFailure(McpFailureKind.TRANSPORT, "official MCP SDK is unavailable") from exc
            command = [self.config.command, *self.config.args]
            if self.config.sandbox_profile_id is not None:
                if self.sandbox_backend is None or self.sandbox_profile is None:
                    raise SandboxFailure(
                        SandboxFailureKind.UNAVAILABLE,
                        "stdio MCP server requires an explicit sandbox backend",
                    )
                ensure_sandbox_backend(self.sandbox_profile, self.sandbox_backend)
                command = self.sandbox_backend.wrap_command(
                    command,
                    profile=self.sandbox_profile,
                )
            return StdioServerParameters(command=command[0], args=command[1:])
        if self.target is None:
            raise McpFailure(McpFailureKind.TRANSPORT, "in-process MCP target is missing")
        return self.target

    def _sdk_client(self):
        try:
            from mcp import Client
            from mcp_types import Implementation
        except ImportError as exc:  # pragma: no cover - dependency is pinned in pyproject
            raise McpFailure(McpFailureKind.TRANSPORT, "official MCP SDK is unavailable") from exc
        return Client(
            self._sdk_target(),
            mode=MCP_PROTOCOL_VERSION,
            client_info=Implementation(name="health-copilot", version="m9"),
        )

    async def _discover_async(self) -> dict[str, Any]:
        async with self._sdk_client() as client:
            result = await client.session.discover()
            payload = _dump_model(result)
            supported = payload.get("supportedVersions", [])
            if MCP_PROTOCOL_VERSION not in supported:
                raise McpFailure(McpFailureKind.PROTOCOL_VERSION, "MCP server does not support 2026-07-28")
            server_info = (payload.get("_meta") or {}).get("io.modelcontextprotocol/serverInfo")
            self.identity = McpServerIdentity(
                self.identity.server_id,
                self.identity.transport,
                MCP_PROTOCOL_VERSION,
                self.identity.endpoint_identity,
                self.identity.auth_mode,
                self.identity.sandbox_profile_id,
                canonical_json_sha256(server_info) if server_info else None,
            )
            return payload

    def discover(self) -> dict[str, Any]:
        self.discovery_calls += 1
        try:
            return _run_sync(self._discover_async)
        except McpFailure:
            raise
        except SandboxFailure as exc:
            raise McpFailure(
                McpFailureKind.SANDBOX,
                "MCP sandbox setup failed",
                sandbox_failure_kind=exc.kind,
            ) from exc
        except Exception as exc:  # SDK exceptions stay behind the typed boundary
            raise McpFailure(McpFailureKind.DISCOVERY, "MCP server discovery failed") from exc

    async def _list_tools_async(self) -> Any:
        async with self._sdk_client() as client:
            await client.session.discover()
            return await client.list_tools()

    def catalog(self, *, force: bool = False) -> McpToolCatalogSnapshot:
        if not force and self._catalog is not None and self._catalog.is_fresh():
            return self._catalog
        self.list_calls += 1
        try:
            result = _run_sync(self._list_tools_async)
            tools: list[ToolSpec] = []
            for item in result.tools:
                payload = _dump_model(item)
                name = payload.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise McpFailure(McpFailureKind.CATALOG, "MCP tool has no valid name")
                tools.append(
                    ToolSpec(
                        name=name,
                        description=str(payload.get("description") or ""),
                        input_schema=payload.get("inputSchema") or {"type": "object"},
                    )
                )
            self._catalog = McpToolCatalogSnapshot.create(
                server_id=self.config.server_id,
                protocol_version=MCP_PROTOCOL_VERSION,
                tools=tools,
                ttl_ms=getattr(result, "ttl_ms", 0),
                cache_scope=getattr(result, "cache_scope", "private"),
            )
            return self._catalog
        except McpFailure:
            raise
        except SandboxFailure as exc:
            raise McpFailure(
                McpFailureKind.SANDBOX,
                "MCP sandbox setup failed",
                sandbox_failure_kind=exc.kind,
            ) from exc
        except Exception as exc:
            raise McpFailure(McpFailureKind.CATALOG, "MCP tool catalog failed") from exc

    async def _call_async(self, name: str, arguments: dict[str, Any] | None) -> Any:
        async with self._sdk_client() as client:
            await client.session.discover()
            return await client.call_tool(name, arguments)

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        self.tool_calls += 1
        try:
            return _run_sync(lambda: self._call_async(name, arguments))
        except McpFailure:
            raise
        except SandboxFailure as exc:
            raise McpFailure(
                McpFailureKind.SANDBOX,
                "MCP sandbox setup failed",
                sandbox_failure_kind=exc.kind,
            ) from exc
        except Exception as exc:
            raise McpFailure(McpFailureKind.CALL_ERROR, "MCP tool call failed") from exc


def _validate_json_schema(schema: Mapping[str, Any], arguments: object) -> object:
    try:
        from jsonschema import validate
    except ImportError as exc:  # pragma: no cover
        raise ValueError("JSON Schema validation dependency is unavailable") from exc
    validate(arguments, dict(schema))
    return arguments


def _evidence_from_payload(value: object) -> tuple[Evidence, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    evidence: list[Evidence] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        source_id = item.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            continue
        evidence.append(
            Evidence(
                source_id=source_id,
                title=str(item.get("title") or ""),
                excerpt=str(item.get("excerpt") or ""),
                source_url=str(item.get("source_url") or ""),
                score=float(item.get("score") or 0.0),
            )
        )
    return tuple(evidence)


class McpToolAdapter:
    """Expose one MCP tool as the existing Health-Copilot Tool boundary."""

    def __init__(
        self,
        client: McpClient,
        tool_spec: ToolSpec,
        *,
        permission_guard: PermissionGuard,
        exposed_name: str | None = None,
    ) -> None:
        self.client = client
        self.remote_name = tool_spec.name
        self._spec = ToolSpec(
            name=exposed_name or f"mcp::{client.config.server_id}::{tool_spec.name}",
            description=tool_spec.description,
            input_schema=tool_spec.input_schema,
            capability=tool_spec.capability,
        )
        self.permission_guard = permission_guard

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate_arguments(self, arguments: object) -> dict[str, Any]:
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments must be an object")
        validated = _validate_json_schema(self._spec.input_schema, dict(arguments))
        if not isinstance(validated, dict):
            raise TypeError("arguments must be an object")
        return validated

    def execute(self, arguments: object) -> ToolResult:
        return self.execute_with_runtime(arguments, runtime=None)

    def execute_with_runtime(self, arguments: object, runtime=None) -> ToolResult:
        try:
            validated = self.validate_arguments(arguments)
        except Exception:  # noqa: BLE001 - adapter turns schema errors into observations
            return ToolResult.failure("invalid_arguments", "MCP tool arguments are invalid")
        outcome = self.permission_guard.check(
            server_id=self.client.config.server_id,
            tool_name=self.remote_name,
            arguments=validated,
            run_id=runtime.identity.run_id if runtime is not None else "direct-call",
            trace=runtime.trace if runtime is not None else None,
        )
        if outcome.decision != PermissionDecision.ALLOW:
            return ToolResult.failure("permission_denied", outcome.reason)
        if runtime is not None and runtime.trace is not None:
            runtime.trace.emit(
                TraceEventType.MCP_CALL_START,
                server_id=self.client.config.server_id,
                tool_name=self.remote_name,
                argument_hash=canonical_json_sha256(validated),
                protocol_version=MCP_PROTOCOL_VERSION,
                sandbox_profile=self.client.config.sandbox_profile_id,
            )
        try:
            result = self.client.call_tool(self.remote_name, validated)
            if getattr(result, "is_error", False):
                if runtime is not None and runtime.trace is not None:
                    runtime.trace.emit(
                        TraceEventType.MCP_CALL_ERROR,
                        server_id=self.client.config.server_id,
                        tool_name=self.remote_name,
                        failure_kind=McpFailureKind.CALL_ERROR.value,
                    )
                return ToolResult.failure("mcp_tool_call_error", "MCP tool call returned an error")
            structured = getattr(result, "structured_content", None)
            if structured is None:
                content = getattr(result, "content", ())
                text = next((getattr(item, "text", None) for item in content if hasattr(item, "text")), None)
                if isinstance(text, str):
                    try:
                        structured = json.loads(text)
                    except json.JSONDecodeError:
                        structured = text
            observed = _evidence_from_payload(
                structured.get("results", ()) if isinstance(structured, Mapping) else structured
            )
            if runtime is not None and runtime.trace is not None:
                runtime.trace.emit(
                    TraceEventType.MCP_CALL_END,
                    server_id=self.client.config.server_id,
                    tool_name=self.remote_name,
                    success=True,
                    observed_source_ids=[item.source_id for item in observed],
                )
            return ToolResult.success(structured, observed_evidence=observed)
        except McpFailure as exc:
            if runtime is not None and runtime.trace is not None:
                runtime.trace.emit(
                    TraceEventType.MCP_CALL_ERROR,
                    server_id=self.client.config.server_id,
                    tool_name=self.remote_name,
                    failure_kind=exc.kind.value,
                    sandbox_failure_kind=(
                        exc.sandbox_failure_kind.value
                        if exc.sandbox_failure_kind is not None
                        else None
                    ),
                )
            return ToolResult.failure(f"mcp_{exc.kind.value}", "MCP operation failed")


__all__ = [
    "MCP_PROTOCOL_VERSION",
    "MCP_SDK_VERSION",
    "McpClient",
    "McpFailure",
    "McpFailureKind",
    "McpServerConfig",
    "McpServerIdentity",
    "McpToolAdapter",
    "McpToolCatalogSnapshot",
]
