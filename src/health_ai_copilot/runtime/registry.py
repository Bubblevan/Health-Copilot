"""Small explicit trusted component registry; never performs discovery or imports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .components import BuiltComponent, ComponentIdentity, ComponentKind, config_hash


class ComponentRegistryError(RuntimeError):
    """Base error for startup-time component selection failures."""


class DuplicateComponentError(ComponentRegistryError):
    pass


class UnknownComponentError(ComponentRegistryError):
    pass


class ComponentConstructionError(ComponentRegistryError):
    pass


@dataclass(frozen=True)
class ComponentBuildContext:
    """Inputs visible to a registered factory; factories remain source-trusted."""

    profile: Any
    cards: tuple[Any, ...]
    knowledge_scope: Any | None
    environment: Mapping[str, Any]
    instances: Mapping[tuple[ComponentKind, str], Any]


ComponentFactory = Callable[[ComponentBuildContext], BuiltComponent | Any]


@dataclass(frozen=True)
class ComponentRegistration:
    kind: ComponentKind
    component_id: str
    factory: ComponentFactory
    implementation: str
    version: str
    optional_dependency: str | None = None


class ComponentRegistry:
    """Explicit ``(kind, id) -> factory`` map with fail-closed construction."""

    def __init__(self) -> None:
        self._registrations: dict[tuple[ComponentKind, str], ComponentRegistration] = {}

    def register(
        self,
        kind: ComponentKind | str,
        component_id: str,
        factory: ComponentFactory,
        *,
        implementation: str | None = None,
        version: str = "1",
        optional_dependency: str | None = None,
    ) -> None:
        kind = ComponentKind(kind)
        if not isinstance(component_id, str) or not component_id.strip():
            raise ValueError("component_id must be non-empty")
        if not callable(factory):
            raise TypeError("component factory must be callable")
        key = (kind, component_id)
        if key in self._registrations:
            raise DuplicateComponentError(f"duplicate component ID: {kind.value}/{component_id}")
        self._registrations[key] = ComponentRegistration(
            kind,
            component_id,
            factory,
            implementation or f"{factory.__module__}.{factory.__qualname__}",
            version,
            optional_dependency,
        )

    def lookup(self, kind: ComponentKind | str, component_id: str) -> ComponentRegistration:
        key = (ComponentKind(kind), component_id)
        try:
            return self._registrations[key]
        except KeyError as exc:
            raise UnknownComponentError(
                f"unknown component: {key[0].value}/{component_id}"
            ) from exc

    def registered_ids(self) -> tuple[tuple[ComponentKind, str], ...]:
        return tuple(sorted(self._registrations, key=lambda item: (item[0].value, item[1])))

    def build(
        self,
        kind: ComponentKind | str,
        component_id: str,
        context: ComponentBuildContext,
    ) -> BuiltComponent:
        registration = self.lookup(kind, component_id)
        try:
            result = registration.factory(context)
            if isinstance(result, BuiltComponent):
                binding = result
            else:
                binding = BuiltComponent(result)
            if binding.instance is None:
                raise ValueError("factory returned no component instance")
            identity = binding.identity or ComponentIdentity(
                kind=registration.kind,
                component_id=registration.component_id,
                implementation=registration.implementation,
                version=registration.version,
                config_hash=config_hash(
                    context.profile.config.get(registration.kind.value, {})
                    if isinstance(context.profile.config, Mapping)
                    else {}
                ),
            )
            if identity.kind != registration.kind or identity.component_id != registration.component_id:
                raise ValueError("factory identity does not match its registration")
            return BuiltComponent(binding.instance, identity)
        except ComponentRegistryError:
            raise
        except Exception as exc:
            dependency = (
                f" optional dependency '{registration.optional_dependency}' is unavailable;"
                if registration.optional_dependency
                else ""
            )
            raise ComponentConstructionError(
                f"failed to construct {registration.kind.value}/{registration.component_id};"
                f"{dependency} {exc}"
            ) from exc
