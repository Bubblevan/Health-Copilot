"""Execution-control primitives shared by live and replayable harnesses."""

from .context import RunContext, RunIdentity
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
from .replay import (
    RecordedProviderExchange,
    RecordedToolExchange,
    RecordingProviderExecutor,
    RecordingToolRunner,
    ReplayProviderExecutor,
    ReplayToolRunner,
    read_provider_exchanges,
    read_tool_exchanges,
    write_provider_exchanges,
    write_tool_exchanges,
)

__all__ = [
    "FailureInjectingProviderExecutor",
    "FailureInjectingToolRunner",
    "FailureInjectionPlan",
    "FakeProviderExecutor",
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
    "ReplayProviderExecutor",
    "ReplayToolRunner",
    "RunContext",
    "RunIdentity",
    "provider_request_fingerprint",
    "read_provider_exchanges",
    "read_tool_exchanges",
    "write_provider_exchanges",
    "write_tool_exchanges",
]
