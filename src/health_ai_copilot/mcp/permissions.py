"""Harness-owned permission and approval policy for MCP tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4

from ..runtime.trace import TraceEventType, canonical_json_sha256


class CapabilityClass(StrEnum):
    READ = "read"
    WRITE_WORKSPACE = "write_workspace"
    NETWORK = "network"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    SENSITIVE_DATA = "sensitive_data"


class PermissionDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ApprovalScope(StrEnum):
    ONE_SHOT = "one_shot"
    RUN = "run"


@dataclass(frozen=True)
class CapabilityPolicy:
    """Trusted local classification; MCP metadata cannot create this object."""

    capability: CapabilityClass
    decision: PermissionDecision = PermissionDecision.DENY
    read_paths: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    network_origins: tuple[str, ...] = ()
    reason: str = "trusted local policy"

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability", CapabilityClass(self.capability))
        object.__setattr__(self, "decision", PermissionDecision(self.decision))
        for name in ("read_paths", "write_paths", "network_origins"):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.value,
            "decision": self.decision.value,
            "read_paths": list(self.read_paths),
            "write_paths": list(self.write_paths),
            "network_origins": list(self.network_origins),
            "reason": self.reason,
        }


class PermissionPolicy:
    """Default-deny mapping from trusted server/tool identity to capability."""

    def __init__(
        self,
        bindings: Mapping[tuple[str, str], CapabilityPolicy] = (),
        *,
        policy_id: str = "permission-policy-v1",
    ) -> None:
        self.policy_id = policy_id
        self._bindings = dict(bindings)
        if any(
            not isinstance(key, tuple)
            or len(key) != 2
            or not all(isinstance(item, str) and item.strip() for item in key)
            for key in self._bindings
        ):
            raise ValueError("permission binding keys must be (server_id, tool_name)")
        if not all(isinstance(value, CapabilityPolicy) for value in self._bindings.values()):
            raise TypeError("permission bindings must contain CapabilityPolicy values")

    @property
    def config_hash(self) -> str:
        payload = {
            "policy_id": self.policy_id,
            "bindings": [
                {"server_id": server_id, "tool_name": tool_name, "policy": policy.to_dict()}
                for (server_id, tool_name), policy in sorted(self._bindings.items())
            ],
        }
        return canonical_json_sha256(payload)

    def resolve(self, server_id: str, tool_name: str) -> CapabilityPolicy | None:
        return self._bindings.get((server_id, tool_name))

    def evaluate(self, server_id: str, tool_name: str) -> tuple[PermissionDecision, str, CapabilityPolicy | None]:
        policy = self.resolve(server_id, tool_name)
        if policy is None:
            return PermissionDecision.DENY, "unknown_server_or_tool", None
        return policy.decision, policy.reason, policy


@dataclass(frozen=True)
class ApprovalRequest:
    approval_request_id: str
    run_id: str
    server_id: str
    tool_name: str
    capability: CapabilityClass
    argument_hash: str
    resource_scope: Mapping[str, tuple[str, ...]]
    reason: str
    scope: ApprovalScope = ApprovalScope.ONE_SHOT

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        server_id: str,
        tool_name: str,
        capability: CapabilityClass,
        arguments: object,
        resource_scope: Mapping[str, Sequence[str]],
        reason: str,
        scope: ApprovalScope = ApprovalScope.ONE_SHOT,
    ) -> ApprovalRequest:
        return cls(
            approval_request_id=f"approval-{uuid4().hex}",
            run_id=run_id,
            server_id=server_id,
            tool_name=tool_name,
            capability=CapabilityClass(capability),
            argument_hash=canonical_json_sha256(arguments),
            resource_scope={key: tuple(value) for key, value in resource_scope.items()},
            reason=reason,
            scope=ApprovalScope(scope),
        )


@dataclass(frozen=True)
class ApprovalResult:
    approved: bool
    approval_request_id: str
    reason: str
    scope: ApprovalScope = ApprovalScope.ONE_SHOT


class ApprovalProvider(Protocol):
    def decide(self, request: ApprovalRequest) -> ApprovalResult:
        ...


class AlwaysApprove:
    def decide(self, request: ApprovalRequest) -> ApprovalResult:
        return ApprovalResult(True, request.approval_request_id, "test approval", request.scope)


class AlwaysDeny:
    def decide(self, request: ApprovalRequest) -> ApprovalResult:
        return ApprovalResult(False, request.approval_request_id, "test denial", request.scope)


class ScriptedApprovalProvider:
    def __init__(self, decisions: Sequence[bool]) -> None:
        self._decisions = list(decisions)

    def decide(self, request: ApprovalRequest) -> ApprovalResult:
        approved = self._decisions.pop(0) if self._decisions else False
        return ApprovalResult(approved, request.approval_request_id, "scripted decision", request.scope)


@dataclass(frozen=True)
class PermissionOutcome:
    decision: PermissionDecision
    capability: CapabilityClass | None
    reason: str
    approval: ApprovalResult | None = None
    approval_request_id: str | None = None


class PermissionGuard:
    """Evaluates local policy and approval before the MCP client is called."""

    def __init__(
        self,
        policy: PermissionPolicy,
        approval_provider: ApprovalProvider | None = None,
    ) -> None:
        self.policy = policy
        self.approval_provider = approval_provider

    def check(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: object,
        run_id: str,
        trace=None,
    ) -> PermissionOutcome:
        decision, reason, capability_policy = self.policy.evaluate(server_id, tool_name)
        capability = capability_policy.capability if capability_policy else None
        fields = {
            "server_id": server_id,
            "tool_name": tool_name,
            "capability": capability.value if capability else None,
            "argument_hash": canonical_json_sha256(arguments),
            "decision": decision.value,
            "policy_id": self.policy.policy_id,
        }
        if trace is not None:
            trace.emit(TraceEventType.PERMISSION_CHECK, **fields)
        if decision == PermissionDecision.DENY:
            return PermissionOutcome(decision, capability, reason)
        if decision == PermissionDecision.ALLOW:
            return PermissionOutcome(decision, capability, reason)
        if self.approval_provider is None or capability_policy is None:
            return PermissionOutcome(PermissionDecision.DENY, capability, "approval_provider_unavailable")
        request = ApprovalRequest.create(
            run_id=run_id,
            server_id=server_id,
            tool_name=tool_name,
            capability=capability_policy.capability,
            arguments=arguments,
            resource_scope={
                "read_paths": capability_policy.read_paths,
                "write_paths": capability_policy.write_paths,
                "network_origins": capability_policy.network_origins,
            },
            reason=reason,
        )
        if trace is not None:
            trace.emit(
                TraceEventType.APPROVAL_REQUESTED,
                server_id=server_id,
                tool_name=tool_name,
                capability=capability.value if capability else None,
                argument_hash=request.argument_hash,
                approval_request_id=request.approval_request_id,
            )
        result = self.approval_provider.decide(request)
        if result.approval_request_id != request.approval_request_id:
            return PermissionOutcome(PermissionDecision.DENY, capability, "approval_binding_mismatch")
        event = TraceEventType.APPROVAL_GRANTED if result.approved else TraceEventType.APPROVAL_DENIED
        if trace is not None:
            trace.emit(
                event,
                server_id=server_id,
                tool_name=tool_name,
                capability=capability.value if capability else None,
                argument_hash=request.argument_hash,
                approval_request_id=request.approval_request_id,
            )
        return PermissionOutcome(
            PermissionDecision.ALLOW if result.approved else PermissionDecision.DENY,
            capability,
            result.reason if result.approved else "approval_denied",
            approval=result,
            approval_request_id=request.approval_request_id,
        )


__all__ = [
    "AlwaysApprove",
    "AlwaysDeny",
    "ApprovalProvider",
    "ApprovalRequest",
    "ApprovalResult",
    "CapabilityClass",
    "CapabilityPolicy",
    "PermissionDecision",
    "PermissionGuard",
    "PermissionOutcome",
    "PermissionPolicy",
    "ScriptedApprovalProvider",
]
