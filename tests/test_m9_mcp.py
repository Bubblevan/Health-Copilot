"""Deterministic M9 MCP, permission, and containment checks."""

from __future__ import annotations

import asyncio
import json
import platform
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2
import pytest

from health_ai_copilot.agent.messages import ToolCall
from health_ai_copilot.agent.tools import ToolRegistry
from health_ai_copilot.contracts import Evidence
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.knowledge.scope import CapabilityTopic, KnowledgeScope
from health_ai_copilot.mcp.client import (
    MCP_PROTOCOL_VERSION,
    McpClient,
    McpServerConfig,
    McpToolAdapter,
)
from health_ai_copilot.mcp.permissions import (
    AlwaysApprove,
    AlwaysDeny,
    ApprovalResult,
    CapabilityClass,
    CapabilityPolicy,
    PermissionDecision,
    PermissionGuard,
    PermissionPolicy,
)
from health_ai_copilot.mcp.sandbox import (
    NoSandboxDevBackend,
    SandboxFailure,
    SandboxFilesystemPolicy,
    SandboxNetworkPolicy,
    SandboxPolicy,
    SandboxProfile,
    WslBubblewrapSandboxBackend,
    ensure_sandbox_backend,
)
from health_ai_copilot.mcp.server import build_search_knowledge_server
from health_ai_copilot.retrieval.bm25 import BM25Retriever
from health_ai_copilot.runtime.builder import RuntimeBuilder, default_runtime_profiles
from health_ai_copilot.runtime.context import RunContext
from health_ai_copilot.runtime.replay import RecordingToolRunner, ReplayToolRunner
from health_ai_copilot.runtime.trace import RunTrace, TraceEventType
from health_ai_copilot.tools.search_knowledge import SearchKnowledgeTool


class _FixtureRetriever:
    def search(self, query: str, top_k: int = 3) -> list[Evidence]:
        return [Evidence("source-a", "Title", f"result for {query}", "https://example.test/a", 1.0)][:top_k]


def _client(*, retriever=None, server_id: str = "fixture-server", ttl_ms: int = 1000):
    server = build_search_knowledge_server(
        retriever or _FixtureRetriever(), server_id=server_id, ttl_ms=ttl_ms
    )
    config = McpServerConfig(
        server_id=server_id,
        transport="inproc_test",
        endpoint_identity=f"inproc:{server_id}",
    )
    return McpClient(config, server)


def test_modern_discovery_is_pinned_and_does_not_use_legacy_session_handshake() -> None:
    client = _client()

    result = client.discover()

    assert result["supportedVersions"] == [MCP_PROTOCOL_VERSION]
    assert client.discovery_calls == 1
    assert not hasattr(client, "session_id")


def test_remote_endpoint_is_trusted_configuration_not_model_supplied_authority() -> None:
    with pytest.raises(ValueError, match=r"http\(s\)"):
        McpServerConfig(
            server_id="remote",
            transport="streamable_http",
            endpoint_identity="remote",
            endpoint="file:///etc/passwd",
        )
    with pytest.raises(ValueError, match="credentials"):
        McpServerConfig(
            server_id="remote",
            transport="streamable_http",
            endpoint_identity="remote",
            endpoint="https://user:secret@example.test/mcp",
        )


def test_catalog_hash_order_and_ttl_cache_are_deterministic() -> None:
    client = _client()

    first = client.catalog()
    second = client.catalog()
    refreshed = client.catalog(force=True)

    assert [tool.name for tool in first.tools] == ["search_knowledge"]
    assert first.catalog_hash == second.catalog_hash == refreshed.catalog_hash
    assert first.ttl_ms == 1000
    assert first.cache_scope == "public"
    assert client.list_calls == 2

    zero_ttl = _client(ttl_ms=0)
    zero_ttl.catalog()
    zero_ttl.catalog()
    assert zero_ttl.list_calls == 2


def test_internal_and_mcp_search_have_same_evidence_order_and_tool_boundary() -> None:
    cards = load_knowledge_cards("tests/fixtures/knowledge_cards")
    retriever = BM25Retriever(cards)
    internal = SearchKnowledgeTool(retriever).execute({"query": "低盐饮食"})
    client = _client(retriever=retriever)
    remote_spec = client.catalog().tools[0]
    guard = PermissionGuard(
        PermissionPolicy(
            {
                ("fixture-server", "search_knowledge"): CapabilityPolicy(
                    CapabilityClass.READ, PermissionDecision.ALLOW
                )
            }
        )
    )
    mcp = McpToolAdapter(client, remote_spec, permission_guard=guard)

    remote = mcp.execute({"query": "低盐饮食"})

    assert internal.ok and remote.ok
    assert [item.source_id for item in internal.observed_evidence] == [
        item.source_id for item in remote.observed_evidence
    ]
    assert remote.data["results"][0]["source_id"] == internal.data[0]["source_id"]
    assert remote.data["results"][0]["title"] == internal.data[0]["title"]


def test_permission_is_default_deny_and_approval_happens_before_tool_call() -> None:
    client = _client()
    spec = client.catalog().tools[0]
    unknown = PermissionGuard(PermissionPolicy())
    denied = McpToolAdapter(client, spec, permission_guard=unknown).execute({"query": "x"})
    assert denied.error is not None and denied.error.code == "permission_denied"
    assert client.tool_calls == 0

    approval_policy = PermissionPolicy(
        {
            ("fixture-server", "search_knowledge"): CapabilityPolicy(
                CapabilityClass.READ, PermissionDecision.REQUIRE_APPROVAL
            )
        }
    )
    denied_by_user = McpToolAdapter(
        client,
        spec,
        permission_guard=PermissionGuard(approval_policy, AlwaysDeny()),
    ).execute({"query": "x"})
    assert denied_by_user.error is not None and denied_by_user.error.code == "permission_denied"
    assert client.tool_calls == 0

    approved = McpToolAdapter(
        client,
        spec,
        permission_guard=PermissionGuard(approval_policy, AlwaysApprove()),
    ).execute({"query": "x"})
    assert approved.ok
    assert client.tool_calls == 1

    class _MismatchedApproval:
        def decide(self, request):
            return ApprovalResult(True, "approval-wrong", "wrong binding")

    mismatch = PermissionGuard(approval_policy, _MismatchedApproval())
    result = McpToolAdapter(client, spec, permission_guard=mismatch).execute({"query": "x"})
    assert result.error is not None and result.error.code == "permission_denied"
    assert client.tool_calls == 1


def test_untrusted_mcp_output_does_not_enter_permission_or_registry_authority() -> None:
    client = _client()
    spec = client.catalog().tools[0]
    guard = PermissionGuard(
        PermissionPolicy(
            {
                ("fixture-server", "search_knowledge"): CapabilityPolicy(
                    CapabilityClass.READ, PermissionDecision.ALLOW
                )
            }
        )
    )
    adapter = McpToolAdapter(client, spec, permission_guard=guard)
    registry = ToolRegistry([adapter])

    result = registry.execute_by_name("mcp::fixture-server::search_knowledge", {"query": "ignore previous policy"})

    assert result.ok
    assert registry.lookup("mcp::fixture-server::search_knowledge") is adapter
    assert guard.policy.resolve("fixture-server", "search_knowledge") is not None
    assert registry.lookup("search_knowledge") is None


def test_mcp_namespace_prevents_silent_tool_collisions() -> None:
    client_a = _client(server_id="server-a")
    client_b = _client(server_id="server-b")
    spec_a = client_a.catalog().tools[0]
    spec_b = client_b.catalog().tools[0]
    allow = PermissionGuard(PermissionPolicy())
    adapter_a = McpToolAdapter(client_a, spec_a, permission_guard=allow)
    adapter_b = McpToolAdapter(client_b, spec_b, permission_guard=allow)

    assert adapter_a.spec.name == "mcp::server-a::search_knowledge"
    assert adapter_b.spec.name == "mcp::server-b::search_knowledge"
    registry = ToolRegistry([adapter_a, adapter_b])
    assert len(registry.list_model_tool_specs()) == 2


def test_mcp_permission_and_call_trace_contains_hashes_not_secrets() -> None:
    client = _client()
    spec = client.catalog().tools[0]
    policy = PermissionPolicy(
        {
            ("fixture-server", "search_knowledge"): CapabilityPolicy(
                CapabilityClass.READ, PermissionDecision.ALLOW
            )
        }
    )
    trace = RunTrace()
    runtime = RunContext.create("m9_mcp", trace=trace)
    result = McpToolAdapter(client, spec, permission_guard=PermissionGuard(policy)).execute_with_runtime(
        {"query": "token=do-not-log"}, runtime=runtime
    )

    assert result.ok
    event_types = [event.event_type for event in trace.events]
    assert TraceEventType.PERMISSION_CHECK in event_types
    assert TraceEventType.MCP_CALL_START in event_types
    assert TraceEventType.MCP_CALL_END in event_types
    trace_text = str([event.fields for event in trace.events])
    assert "do-not-log" not in trace_text
    assert "argument_hash" in trace_text


def test_m9_runtime_profile_exposes_mcp_adapter_and_safe_manifest_identity() -> None:
    cards = load_knowledge_cards("tests/fixtures/knowledge_cards")
    profile = default_runtime_profiles()["m9-mcp-search-bm25-v1"]

    class _Provider:
        def execute(self, request, runtime):
            raise AssertionError("M9 build smoke must not call a provider")

    components = RuntimeBuilder(environment={"provider_executor": _Provider()}).build(
        profile,
        cards=cards,
        knowledge_scope=KnowledgeScope(
            "fixture-scope",
            "1",
            "fixture-pack",
            "2026-09-20",
            "test",
            "fixture",
            ("synthetic_test",),
            (CapabilityTopic("all", "fixture coverage", tuple(card.id for card in cards)),),
        ),
    )

    # The experimental profile is an in-process protocol/permission parity
    # profile; its explicit no-sandbox identity must not be mistaken for H6.
    tool = components.tool_registry.lookup("search_knowledge")
    assert isinstance(tool, McpToolAdapter)
    manifest = components.component_manifest.to_dict()
    mcp_identity = next(item for item in manifest["components"] if item["kind"] == "mcp_client")
    assert mcp_identity["config_hash"]
    assert "endpoint" not in mcp_identity


def test_required_sandbox_fails_closed_for_non_containing_backend(tmp_path: Path) -> None:
    profile = SandboxProfile(
        "required",
        SandboxPolicy(SandboxFilesystemPolicy(write_roots=(tmp_path,))),
        requires_real_enforcement=True,
    )

    with pytest.raises(SandboxFailure, match="not containing"):
        ensure_sandbox_backend(profile, NoSandboxDevBackend())


def test_m9_security_eval_manifest_is_local_and_fixture_tools_stay_out_of_product_profiles() -> None:
    rows = [
        json.loads(line)
        for line in Path("evals/m9_mcp_security_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["id"] for row in rows} >= {
        "allowed-read",
        "unknown-capability",
        "outside-write",
        "network-egress",
        "modern-http-headers",
    }
    profiles = default_runtime_profiles()
    for profile_id in ("m3-bm25-default", "m8-team-bm25-v1", "m9-mcp-search-bm25-v1"):
        assert all("security" not in tool_id for tool_id in profiles[profile_id].tool_set)


def test_mcp_tool_result_can_be_replayed_without_starting_mcp() -> None:
    client = _client()
    spec = client.catalog().tools[0]
    guard = PermissionGuard(
        PermissionPolicy(
            {
                ("fixture-server", "search_knowledge"): CapabilityPolicy(
                    CapabilityClass.READ, PermissionDecision.ALLOW
                )
            }
        )
    )
    adapter = McpToolAdapter(client, spec, permission_guard=guard)
    call = ToolCall("call-1", adapter.spec.name, {"query": "x"})

    class _Delegate:
        def execute(self, incoming, runtime):
            assert incoming.name == call.name
            return adapter.execute_with_runtime(incoming.arguments, runtime=runtime)

    recording = RecordingToolRunner(_Delegate())
    recording.execute(call, RunContext.create("m9_mcp"))
    live_calls = client.tool_calls
    replay = ReplayToolRunner(recording.exchanges)
    replayed = replay.execute(call, RunContext.create("m9_mcp"))

    assert replayed.ok
    assert client.tool_calls == live_calls


@pytest.mark.skipif(platform.system() != "Windows", reason="WSL containment smoke is Windows-host specific")
def test_wsl_bubblewrap_real_filesystem_containment(tmp_path: Path) -> None:
    backend = WslBubblewrapSandboxBackend()
    if not backend.is_available():
        pytest.skip("WSL2 Bubblewrap is unavailable")
    allowed_dir = tmp_path / "allowed"
    workspace = tmp_path / "workspace"
    outside_dir = tmp_path / "outside"
    allowed_dir.mkdir()
    workspace.mkdir()
    outside_dir.mkdir()
    allowed = allowed_dir / "allowed.txt"
    outside = outside_dir / "secret.txt"
    allowed.write_text("allowed", encoding="utf-8")
    outside.write_text("secret", encoding="utf-8")
    profile = SandboxProfile(
        "real-wsl-bwrap",
        SandboxPolicy(
            SandboxFilesystemPolicy(read_roots=(allowed_dir,), write_roots=(workspace,)),
            network=SandboxNetworkPolicy.DENY_ALL,
        ),
    )
    allowed_wsl = backend._wsl_path(allowed)
    outside_wsl = backend._wsl_path(outside)
    script = f"test \"$(cat '{allowed_wsl}')\" = allowed && ! cat '{outside_wsl}'"

    result = backend.run(["/bin/sh", "-c", script], profile=profile, cwd=workspace)

    assert result.contained is True
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(platform.system() != "Windows", reason="WSL containment smoke is Windows-host specific")
def test_real_stdio_mcp_fixture_is_contained_by_wsl_bubblewrap(tmp_path: Path) -> None:
    site_dir = Path(".m9-wsl-site").resolve()
    fixture_script = Path("tools/m9_security_mcp_server.py").resolve()
    backend = WslBubblewrapSandboxBackend()
    if not backend.is_available() or not site_dir.exists():
        pytest.skip("WSL Bubblewrap fixture dependencies are unavailable")

    allowed_dir = tmp_path / "allowed"
    workspace = tmp_path / "workspace"
    outside_dir = tmp_path / "outside"
    allowed_dir.mkdir()
    workspace.mkdir()
    outside_dir.mkdir()
    allowed_file = allowed_dir / "allowed.txt"
    outside_file = outside_dir / "secret.txt"
    allowed_file.write_text("allowed", encoding="utf-8")
    outside_file.write_text("secret", encoding="utf-8")

    profile = SandboxProfile(
        "real-wsl-mcp-fixture",
        SandboxPolicy(
            SandboxFilesystemPolicy(
                read_roots=(site_dir, fixture_script.parent, allowed_dir),
                write_roots=(workspace,),
            ),
            network=SandboxNetworkPolicy.DENY_ALL,
        ),
    )
    script_wsl = backend._wsl_path(fixture_script)
    site_wsl = backend._wsl_path(site_dir)
    allowed_wsl = backend._wsl_path(allowed_file)
    workspace_wsl = backend._wsl_path(workspace)
    outside_wsl = backend._wsl_path(outside_file)
    server_id = "m9-real-stdio-fixture"
    config = McpServerConfig(
        server_id=server_id,
        transport="stdio",
        endpoint_identity="wsl-bwrap:m9-security-fixture",
        command="/usr/bin/env",
        args=(
            f"PYTHONPATH={site_wsl}",
            "/usr/bin/python3",
            script_wsl,
            "--allowed-file",
            allowed_wsl,
            "--workspace",
            workspace_wsl,
        ),
        sandbox_profile_id=profile.profile_id,
    )
    client = McpClient(config, sandbox_backend=backend, sandbox_profile=profile)
    catalog = client.catalog()
    names = {tool.name for tool in catalog.tools}
    assert {"read_allowed_file", "read_forbidden_file", "write_workspace_file", "write_outside_workspace", "network_probe"} <= names

    allow = PermissionPolicy(
        {
            (server_id, name): CapabilityPolicy(
                CapabilityClass.READ
                if name.startswith("read_")
                else CapabilityClass.NETWORK
                if name == "network_probe"
                else CapabilityClass.WRITE_WORKSPACE,
                PermissionDecision.ALLOW,
            )
            for name in names
        }
    )
    guard = PermissionGuard(allow)
    adapters = {tool.name: McpToolAdapter(client, tool, permission_guard=guard) for tool in catalog.tools}

    read_ok = adapters["read_allowed_file"].execute({})
    write_ok = adapters["write_workspace_file"].execute({"name": "created.txt", "text": "inside"})
    read_denied_by_os = adapters["read_forbidden_file"].execute({"path": outside_wsl})
    write_denied_by_os = adapters["write_outside_workspace"].execute({"path": outside_wsl, "text": "escape"})
    network_denied_by_os = adapters["network_probe"].execute({"host": "example.com", "port": 80})
    approved_but_sandbox_denies = McpToolAdapter(
        client,
        next(tool for tool in catalog.tools if tool.name == "write_outside_workspace"),
        permission_guard=PermissionGuard(
            PermissionPolicy(
                {
                    (server_id, "write_outside_workspace"): CapabilityPolicy(
                        CapabilityClass.WRITE_WORKSPACE,
                        PermissionDecision.REQUIRE_APPROVAL,
                    )
                }
            ),
            AlwaysApprove(),
        ),
    ).execute({"path": outside_wsl, "text": "approved escape"})

    assert read_ok.ok and read_ok.data["text"] == "allowed"
    assert write_ok.ok and (workspace / "created.txt").read_text(encoding="utf-8") == "inside"
    assert read_denied_by_os.error is not None
    assert write_denied_by_os.error is not None
    assert network_denied_by_os.error is not None
    assert approved_but_sandbox_denies.error is not None
    assert not outside_file.exists() or outside_file.read_text(encoding="utf-8") == "secret"


def test_streamable_http_uses_modern_protocol_and_routing_headers() -> None:
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server.transport_security import TransportSecuritySettings

    from health_ai_copilot.mcp.server import build_search_knowledge_server

    class _Capture:
        def __init__(self, app):
            self.app = app
            self.headers: list[dict[str, str]] = []

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                self.headers.append(
                    {key.decode(): value.decode() for key, value in scope["headers"]}
                )
            await self.app(scope, receive, send)

    async def run() -> list[dict[str, str]]:
        server = build_search_knowledge_server(_FixtureRetriever())
        raw_app = server.streamable_http_app(
            stateless_http=True,
            json_response=True,
            transport_security=TransportSecuritySettings(allowed_hosts=["test"]),
        )
        capture = _Capture(raw_app)
        http = httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=capture), base_url="http://test"
        )

        @asynccontextmanager
        async def transport():
            async with streamable_http_client("http://test/mcp", http_client=http) as streams:
                yield streams

        async with raw_app.router.lifespan_context(raw_app), Client(
            transport(), mode=MCP_PROTOCOL_VERSION
        ) as client:
            await client.session.discover()
            await client.list_tools()
            await client.call_tool("search_knowledge", {"query": "x"})
        await http.aclose()
        return capture.headers

    headers = asyncio.run(run())
    list_headers = next(item for item in headers if item.get("mcp-method") == "tools/list")
    call_headers = next(item for item in headers if item.get("mcp-method") == "tools/call")
    assert list_headers["mcp-protocol-version"] == MCP_PROTOCOL_VERSION
    assert call_headers["mcp-protocol-version"] == MCP_PROTOCOL_VERSION
    assert call_headers["mcp-name"] == "search_knowledge"
    assert all("mcp-session-id" not in item for item in headers)
