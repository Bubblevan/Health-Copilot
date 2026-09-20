"""Execution-control primitives shared by live and replayable harnesses."""

from .components import (
    BuiltComponent,
    ComponentIdentity,
    ComponentKind,
    ComponentManifest,
    LearnedArtifactIdentity,
)
from .context import RunContext, RunIdentity, call_with_optional_runtime
from .failure_injection import (
    FailureInjectingProviderExecutor,
    FailureInjectingToolRunner,
    FailureInjectionPlan,
)
from .provider import (
    FakeProviderExecutor,
    OpenAICompatibleProviderExecutor,
    ProviderCallKind,
    ProviderExecutor,
    ProviderFailure,
    ProviderFailureKind,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    provider_request_fingerprint,
)
from .registry import (
    ComponentBuildContext,
    ComponentConstructionError,
    ComponentRegistry,
    ComponentRegistryError,
    DuplicateComponentError,
    UnknownComponentError,
)
from .replay import (
    RecordedProviderExchange,
    RecordedToolExchange,
    RecordingProviderExecutor,
    RecordingToolRunner,
    ReplayMetadata,
    ReplayProviderExecutor,
    ReplayToolRunner,
    read_provider_exchanges,
    read_tool_exchanges,
    write_provider_exchanges,
    write_tool_exchanges,
)

__all__ = [
    "BuiltComponent",
    "ComponentBuildContext",
    "ComponentConstructionError",
    "ComponentIdentity",
    "ComponentKind",
    "ComponentManifest",
    "ComponentRegistry",
    "ComponentRegistryError",
    "DuplicateComponentError",
    "FailureInjectingProviderExecutor",
    "FailureInjectingToolRunner",
    "FailureInjectionPlan",
    "FakeProviderExecutor",
    "LearnedArtifactIdentity",
    "OpenAICompatibleProviderExecutor",
    "ProviderCallKind",
    "ProviderExecutor",
    "ProviderFailure",
    "ProviderFailureKind",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderUsage",
    "RecordedProviderExchange",
    "RecordedToolExchange",
    "RecordingProviderExecutor",
    "RecordingToolRunner",
    "ReplayMetadata",
    "ReplayProviderExecutor",
    "ReplayToolRunner",
    "RunContext",
    "RunIdentity",
    "UnknownComponentError",
    "call_with_optional_runtime",
    "provider_request_fingerprint",
    "read_provider_exchanges",
    "read_tool_exchanges",
    "write_provider_exchanges",
    "write_tool_exchanges",
]


def __getattr__(name: str):
    """Lazy-load builder symbols to avoid retrieval/runtime import cycles."""

    if name in {
        "RuntimeBuilder",
        "RuntimeBuildConfig",
        "RuntimeBuildError",
        "RuntimeComponents",
        "RuntimeProfile",
        "default_component_registry",
        "default_runtime_profiles",
    }:
        from .builder import (
            RuntimeBuildConfig,
            RuntimeBuilder,
            RuntimeBuildError,
            RuntimeComponents,
            default_component_registry,
            default_runtime_profiles,
        )
        from .profile import RuntimeProfile

        return {
            "RuntimeBuilder": RuntimeBuilder,
            "RuntimeBuildConfig": RuntimeBuildConfig,
            "RuntimeBuildError": RuntimeBuildError,
            "RuntimeComponents": RuntimeComponents,
            "RuntimeProfile": RuntimeProfile,
            "default_component_registry": default_component_registry,
            "default_runtime_profiles": default_runtime_profiles,
        }[name]
    raise AttributeError(name)
