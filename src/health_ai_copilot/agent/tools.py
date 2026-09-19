"""Typed tool contracts and the deliberately small M1 registry."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..contracts import Evidence
from .messages import ToolCall


@dataclass(frozen=True)
class ToolCapability:
    """Narrow M3 association between the single search tool and reviewed scope."""

    tool_name: str
    scope_id: str
    scope_version: str
    domain: str
    topic_ids: tuple[str, ...]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    capability: ToolCapability | None = None


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str


@dataclass(frozen=True)
class ToolResult:
    """A structured success or structured error observation."""

    ok: bool
    data: Any = None
    error: ToolError | None = None
    observed_evidence: tuple[Evidence, ...] = ()

    @classmethod
    def success(
        cls, data: Any, *, observed_evidence: Sequence[Evidence] = ()
    ) -> "ToolResult":
        return cls(
            ok=True,
            data=data,
            observed_evidence=tuple(observed_evidence),
        )

    @classmethod
    def failure(cls, code: str, message: str) -> "ToolResult":
        return cls(ok=False, error=ToolError(code=code, message=message))

    @property
    def is_error(self) -> bool:
        return not self.ok

    @property
    def success_result(self) -> bool:
        return self.ok


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec:
        ...

    def validate_arguments(self, arguments: object) -> object:
        ...

    def execute(self, arguments: object) -> ToolResult:
        ...


class ToolRegistry:
    """Explicit name-to-tool registry; no discovery, scanning or plugin loading."""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        spec = _get_tool_spec(tool)
        name = spec.name
        if not name.strip():
            raise ValueError("tool name must be non-empty")
        if name in self._tools:
            raise ValueError(f"duplicate tool name: {name}")
        self._tools[name] = tool

    def lookup(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get(self, name: str) -> Tool:
        tool = self.lookup(name)
        if tool is None:
            raise KeyError(name)
        return tool

    def list_model_tool_specs(self) -> list[ToolSpec]:
        return [_get_tool_spec(self._tools[name]) for name in sorted(self._tools)]

    def execute(self, tool: str | ToolCall, arguments: object | None = None) -> ToolResult:
        """Validate and execute by name, converting all failures to observations."""
        if isinstance(tool, ToolCall):
            name = tool.name
            arguments = tool.arguments
        else:
            name = tool

        target = self.lookup(name)
        if target is None:
            return ToolResult.failure("unknown_tool", f"unknown tool: {name}")

        try:
            validator = getattr(target, "validate_arguments", None)
            validated = (
                validator(arguments)
                if callable(validator)
                else _validate_against_schema(_get_tool_spec(target).input_schema, arguments)
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary is fail-closed
            return ToolResult.failure("invalid_arguments", _safe_error_message(exc))

        try:
            result = target.execute(validated)
        except Exception:  # noqa: BLE001 - tool errors are observations
            return ToolResult.failure("tool_execution_error", "tool execution failed")
        if not isinstance(result, ToolResult):
            return ToolResult.failure("invalid_tool_result", "tool returned an invalid result")
        return result

    def execute_by_name(self, name: str, arguments: object) -> ToolResult:
        """Explicit name-based convenience API for callers outside the loop."""
        return self.execute(name, arguments)


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message or "tool arguments are invalid"


def _get_tool_spec(tool: Tool) -> ToolSpec:
    spec = getattr(tool, "spec", None)
    if isinstance(spec, ToolSpec):
        return spec
    name = getattr(tool, "name", None)
    description = getattr(tool, "description", None)
    input_schema = getattr(tool, "input_schema", None)
    if (
        isinstance(name, str)
        and isinstance(description, str)
        and isinstance(input_schema, Mapping)
    ):
        return ToolSpec(name, description, input_schema)
    raise TypeError("tool must expose spec or name/description/input_schema")


def _validate_against_schema(schema: Mapping[str, Any], arguments: object) -> object:
    """Validate the small object/string subset used by M1, not full JSON Schema."""
    if schema.get("type") == "object" and not isinstance(arguments, Mapping):
        raise TypeError("arguments must be an object")
    if not isinstance(arguments, Mapping):
        return arguments
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"missing required argument: {missing[0]}")
    if schema.get("additionalProperties") is False:
        unknown = [name for name in arguments if name not in properties]
        if unknown:
            raise ValueError(f"unknown argument: {unknown[0]}")
    for name, definition in properties.items():
        if name not in arguments or not isinstance(definition, Mapping):
            continue
        expected_type = definition.get("type")
        if expected_type == "string" and not isinstance(arguments[name], str):
            raise TypeError(f"argument '{name}' must be a string")
    return dict(arguments)


@dataclass(frozen=True)
class FunctionTool:
    """Small convenience implementation for tests and local tools."""

    spec: ToolSpec
    handler: Callable[[object], ToolResult]
    validator: Callable[[object], object]

    def validate_arguments(self, arguments: object) -> object:
        return self.validator(arguments)

    def execute(self, arguments: object) -> ToolResult:
        return self.handler(arguments)
