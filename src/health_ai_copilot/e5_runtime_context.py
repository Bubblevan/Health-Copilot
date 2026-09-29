"""Task-independent E5 capability metadata sourced from runtime configuration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256

from .retrieval_capabilities import RetrievalAction, retrieval_action_specs

_SOURCE_FAMILIES = frozenset({"public_health", "reviewed_guideline"})


@dataclass(frozen=True, slots=True)
class E5RuntimeCapabilityContext:
    """Capabilities and budgets available from environment, corpus, and policy."""

    available_source_families: tuple[str, ...]
    available_retrieval_actions: tuple[RetrievalAction, ...]
    remaining_provider_budget: int
    remaining_tool_budget: int
    remaining_token_budget: int
    deadline_remaining_ms: int | None
    external_corpus_identity: str | None
    capability_context_source: str = field(default="environment", init=False)

    def __post_init__(self) -> None:
        families = _normalize_strings("available_source_families", self.available_source_families)
        if not set(families).issubset(_SOURCE_FAMILIES):
            raise ValueError("runtime context contains a source family outside E5 v1")
        object.__setattr__(self, "available_source_families", families)

        actions = tuple(
            value if isinstance(value, RetrievalAction) else RetrievalAction(value)
            for value in self.available_retrieval_actions
        )
        if len(actions) != len(set(actions)):
            raise ValueError("available_retrieval_actions must not contain duplicates")
        actions = tuple(sorted(actions, key=lambda action: action.value))
        if RetrievalAction.OFF not in actions:
            raise ValueError("OFF must always be an available fallback action")
        if any(action != RetrievalAction.OFF for action in actions) and not families:
            raise ValueError("external actions require an active source family")
        specs = {spec.action: spec for spec in retrieval_action_specs()}
        for action in actions:
            if action != RetrievalAction.OFF and not set(specs[action].source_families).intersection(
                families
            ):
                raise ValueError(f"active source families cannot serve {action.value}")
        object.__setattr__(self, "available_retrieval_actions", actions)

        for name in (
            "remaining_provider_budget",
            "remaining_tool_budget",
            "remaining_token_budget",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.deadline_remaining_ms is not None and (
            type(self.deadline_remaining_ms) is not int or self.deadline_remaining_ms < 0
        ):
            raise ValueError("deadline_remaining_ms must be a non-negative integer or null")
        if self.external_corpus_identity is not None and not _is_sha256(
            self.external_corpus_identity
        ):
            raise ValueError("external_corpus_identity must be a SHA-256 digest or null")
        if "reviewed_guideline" in families and self.external_corpus_identity is None:
            raise ValueError("reviewed_guideline availability requires an active corpus identity")

    @classmethod
    def from_environment(
        cls,
        *,
        environment_source_families: Sequence[str],
        permission_actions: Sequence[RetrievalAction | str],
        budget_snapshot: dict[str, int | None],
        active_corpus_identity: str | None,
    ) -> E5RuntimeCapabilityContext:
        """Construct from trusted runtime planes; task/evaluator data are not accepted."""
        allowed_budget_fields = {
            "remaining_provider_budget",
            "remaining_tool_budget",
            "remaining_token_budget",
            "deadline_remaining_ms",
        }
        unknown_budget_fields = set(budget_snapshot) - allowed_budget_fields
        if unknown_budget_fields:
            raise ValueError(f"budget snapshot contains unsupported fields: {sorted(unknown_budget_fields)}")
        missing_budget_fields = allowed_budget_fields - set(budget_snapshot)
        if missing_budget_fields:
            raise ValueError(f"budget snapshot is missing fields: {sorted(missing_budget_fields)}")
        return cls(
            available_source_families=tuple(environment_source_families),
            available_retrieval_actions=tuple(permission_actions),
            remaining_provider_budget=budget_snapshot["remaining_provider_budget"],
            remaining_tool_budget=budget_snapshot["remaining_tool_budget"],
            remaining_token_budget=budget_snapshot["remaining_token_budget"],
            deadline_remaining_ms=budget_snapshot["deadline_remaining_ms"],
            external_corpus_identity=active_corpus_identity,
        )

    def to_policy_fields(self) -> dict[str, object]:
        """Return only the frozen ExecutionPolicyObservation capability fields."""
        return {
            "available_source_families": list(self.available_source_families),
            "available_retrieval_actions": [
                action.value for action in self.available_retrieval_actions
            ],
            "remaining_provider_budget": self.remaining_provider_budget,
            "remaining_tool_budget": self.remaining_tool_budget,
            "remaining_token_budget": self.remaining_token_budget,
            "deadline_remaining_ms": self.deadline_remaining_ms,
        }

    @property
    def provenance_sha256(self) -> str:
        """Hash the runtime capability provenance without task identity or labels."""
        import json

        payload = {
            "capability_context_source": self.capability_context_source,
            "external_corpus_identity": self.external_corpus_identity,
            **self.to_policy_fields(),
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_strings(name: str, values: Sequence[str]) -> tuple[str, ...]:
    items = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in items):
        raise ValueError(f"{name} must contain non-empty strings")
    return tuple(sorted(set(items)))


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


__all__ = ["E5RuntimeCapabilityContext"]
