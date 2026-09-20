"""Declarative runtime profiles; no instantiated Python components are allowed."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .components import config_hash


@dataclass(frozen=True)
class RuntimeProfile:
    """A named, JSON-compatible selection of trusted runtime components."""

    profile_id: str
    provider: str
    retriever: str
    policy: str | None = None
    verifier: str | None = None
    tool_set: tuple[str, ...] = ()
    trace: str = "metadata-jsonl-v1"
    mode: str = "m0"
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("profile_id", "provider", "retriever", "trace", "mode"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"profile {name} must be non-empty")
        for name in ("policy", "verifier"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"profile {name} must be non-empty or null")
        tools = tuple(self.tool_set)
        if any(not isinstance(item, str) or not item.strip() for item in tools):
            raise ValueError("profile tool_set must contain non-empty IDs")
        object.__setattr__(self, "tool_set", tools)
        if not isinstance(self.config, Mapping):
            raise TypeError("profile config must be a mapping")
        # This both validates that no object instances are hidden in a profile and
        # gives callers a convenient stable hash for profile-level configuration.
        config_hash(dict(self.config))

    @property
    def config_hash(self) -> str:
        return config_hash(self.to_dict(include_config=True))

    def to_dict(self, *, include_config: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "profile_id": self.profile_id,
            "provider": self.provider,
            "retriever": self.retriever,
            "policy": self.policy,
            "verifier": self.verifier,
            "tool_set": list(self.tool_set),
            "trace": self.trace,
            "mode": self.mode,
        }
        if include_config:
            value["config"] = dict(self.config)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeProfile:
        if not isinstance(value, Mapping):
            raise TypeError("runtime profile must be an object")
        return cls(
            profile_id=value["profile_id"],
            provider=value["provider"],
            retriever=value["retriever"],
            policy=value.get("policy"),
            verifier=value.get("verifier"),
            tool_set=tuple(value.get("tool_set", ())),
            trace=value.get("trace", "metadata-jsonl-v1"),
            mode=value.get("mode", "m0"),
            config=value.get("config", {}),
        )
