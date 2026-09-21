"""M9.1 declarative graph, failure semantics, and security eval closeout."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from health_ai_copilot.eval.registry import default_eval_suite_registry
from health_ai_copilot.eval.system import EvaluationRunner
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.knowledge.scope import CapabilityTopic, KnowledgeScope
from health_ai_copilot.mcp.client import (
    McpClient,
    McpFailure,
    McpFailureKind,
    McpServerConfig,
)
from health_ai_copilot.mcp.sandbox import (
    NoSandboxDevBackend,
    SandboxFailure,
    SandboxFailureKind,
    SandboxFilesystemPolicy,
    SandboxPolicy,
    SandboxProfile,
    ensure_sandbox_backend,
)
from health_ai_copilot.runtime.builder import (
    RuntimeBuilder,
    RuntimeBuildError,
    default_component_registry,
    default_runtime_profiles,
)
from health_ai_copilot.runtime.components import ComponentKind
from health_ai_copilot.runtime.profile import RuntimeProfile
from health_ai_copilot.runtime.registry import UnknownComponentError

FROZEN_M3_HASH = "06ca629157cea96034f0a1d7ec0dbb7d32797d3543ebd40a508395605aaad2d7"
FROZEN_M8_HASH = "66a618302fd3f9fad02b90e3486e464bee10a183a5f9f40f5195b2731156da86"


class _NoCallProvider:
    def execute(self, request, runtime):
        raise AssertionError("M9.1 build test must not call a provider")


def _scope(cards) -> KnowledgeScope:
    return KnowledgeScope(
        "fixture-scope",
        "1",
        "fixture-pack",
        "2026-09-20",
        "test",
        "fixture",
        ("synthetic_test",),
        (CapabilityTopic("all", "fixture coverage", tuple(card.id for card in cards)),),
    )


def test_old_profile_hashes_and_canonical_dicts_remain_frozen() -> None:
    profiles = default_runtime_profiles()

    assert profiles["m3-bm25-default"].config_hash == FROZEN_M3_HASH
    assert profiles["m8-team-bm25-v1"].config_hash == FROZEN_M8_HASH
    assert "mcp_client" not in profiles["m3-bm25-default"].to_dict()
    assert "permission" not in profiles["m8-team-bm25-v1"].to_dict()


def test_runtime_profile_m9_fields_round_trip_and_reject_invalid_ids() -> None:
    profile = RuntimeProfile(
        "custom-m9",
        "provider",
        "retriever",
        mcp_client="mcp-client-v1",
        permission="permission-v1",
        sandbox="sandbox-v1",
        config={"mcp": {"transport": "inproc_test"}},
    )

    assert RuntimeProfile.from_dict(profile.to_dict()) == profile
    assert RuntimeProfile("old", "provider", "retriever").to_dict() == {
        "profile_id": "old",
        "provider": "provider",
        "retriever": "retriever",
        "policy": None,
        "verifier": None,
        "tool_set": [],
        "trace": "metadata-jsonl-v1",
        "mode": "m0",
        "config": {},
    }
    with pytest.raises(ValueError, match="mcp_client"):
        RuntimeProfile("bad", "provider", "retriever", mcp_client=" ")
    with pytest.raises(ValueError, match="JSON-compatible"):
        RuntimeProfile("bad-config", "provider", "retriever", config={"object": object()})


def test_m9_profile_declares_graph_and_manifest_uses_selected_ids() -> None:
    profiles = default_runtime_profiles()
    profile = profiles["m9-mcp-search-bm25-v1"]
    cards = load_knowledge_cards("tests/fixtures/knowledge_cards")

    components = RuntimeBuilder(environment={"provider_executor": _NoCallProvider()}).build(
        profile, cards=cards, knowledge_scope=_scope(cards)
    )
    identities = {
        item.kind: item.component_id for item in components.component_manifest.components
    }

    assert profile.mcp_client == "mcp-client-2026-07-28-v1"
    assert profile.permission == "permission-policy-v1"
    assert profile.sandbox == "no-sandbox-dev-v1"
    assert identities[ComponentKind.MCP_CLIENT] == profile.mcp_client
    assert identities[ComponentKind.PERMISSION] == profile.permission
    assert identities[ComponentKind.SANDBOX] == profile.sandbox
    assert profile.config_hash != "1216ac4f8ad75089f6105c5ed85475f93e512ebf2abebcbbe7cf1b8d347cff99"
    assert replace(profile, permission="permission-policy-v2").config_hash != profile.config_hash


def test_m9_required_graph_and_unknown_component_fail_closed() -> None:
    cards = load_knowledge_cards("tests/fixtures/knowledge_cards")
    scope = _scope(cards)
    profile = default_runtime_profiles()["m9-mcp-search-bm25-v1"]

    with pytest.raises(RuntimeBuildError, match="mcp_client"):
        RuntimeBuilder(environment={"provider_executor": _NoCallProvider()}).build(
            replace(profile, mcp_client=None), cards=cards, knowledge_scope=scope
        )
    with pytest.raises(UnknownComponentError):
        RuntimeBuilder(environment={"provider_executor": _NoCallProvider()}).build(
            replace(profile, permission="unknown-permission-v1"),
            cards=cards,
            knowledge_scope=scope,
        )

    ids = default_component_registry().registered_ids()
    assert (ComponentKind.SANDBOX, "bubblewrap-sandbox-v1") in ids
    assert (ComponentKind.SANDBOX, "bubblewrap-sandbox-wsl-v1") in ids


def test_missing_stdio_sandbox_is_typed_sandbox_failure_before_process_start(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "started.txt"
    client = McpClient(
        McpServerConfig(
            server_id="required-sandbox",
            transport="stdio",
            endpoint_identity="stdio:required-sandbox",
            command="python",
            args=("-c", f"from pathlib import Path; Path(r'{marker}').write_text('started')"),
            sandbox_profile_id="required-profile",
        )
    )

    with pytest.raises(McpFailure) as caught:
        client.catalog()

    assert caught.value.kind == McpFailureKind.SANDBOX
    assert caught.value.sandbox_failure_kind == SandboxFailureKind.UNAVAILABLE
    assert not marker.exists()


def test_non_containing_backend_is_normalized_to_sandbox_failure(tmp_path: Path) -> None:
    profile = SandboxProfile(
        "required",
        SandboxPolicy(SandboxFilesystemPolicy(write_roots=(tmp_path,))),
        requires_real_enforcement=True,
    )

    with pytest.raises(SandboxFailure) as caught:
        ensure_sandbox_backend(profile, NoSandboxDevBackend())

    assert caught.value.kind == SandboxFailureKind.UNAVAILABLE

    client = McpClient(
        McpServerConfig(
            server_id="non-containing",
            transport="stdio",
            endpoint_identity="stdio:non-containing",
            command="python",
            sandbox_profile_id=profile.profile_id,
        ),
        sandbox_backend=NoSandboxDevBackend(),
        sandbox_profile=profile,
    )
    with pytest.raises(McpFailure) as normalized:
        client.catalog()
    assert normalized.value.kind == McpFailureKind.SANDBOX
    assert normalized.value.sandbox_failure_kind == SandboxFailureKind.UNAVAILABLE


def test_m9_security_suite_is_registered_and_produces_standard_offline_bundle(
    tmp_path: Path,
) -> None:
    suite = default_eval_suite_registry().get("m9-mcp-security-v1")
    assert suite.target_kind.value == "security"
    assert suite.supported_execution_modes == ("offline",)

    runner = EvaluationRunner()
    spec = runner.prepare_run_spec(
        "m9-mcp-security-v1", execution_mode="offline", output_root=tmp_path / "runs"
    )
    bundle = runner.run(spec)
    metrics = __import__("json").loads((bundle / "metrics.json").read_text(encoding="utf-8"))

    assert {
        "run_spec.json",
        "run_manifest.json",
        "case_results.jsonl",
        "grader_results.jsonl",
        "failures.jsonl",
        "metrics.json",
        "report.md",
    } <= {path.name for path in bundle.iterdir() if path.is_file()}
    assert metrics["m9.security_control_pass_rate"]["numerator"] == 12
    assert metrics["m9.security_control_pass_rate"]["denominator"] == 12
    assert metrics["m9.permission_cases_passed"]["numerator"] == 4
    assert metrics["m9.protocol_cases_passed"]["numerator"] == 3
    assert metrics["m9.sandbox_contract_cases_passed"]["numerator"] == 5
    assert (bundle / "failures.jsonl").read_text(encoding="utf-8") == ""
