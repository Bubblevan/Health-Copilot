from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.rag_e5.b4_guidance import GuidanceCompletion
from tools.research.rag_e5 import continue_e5b4_smoke as smoke_resume


class FakeClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> GuidanceCompletion:
        self.prompts.append(prompt)
        return GuidanceCompletion(
            text="General public-health guidance.",
            finish_reason="stop",
            input_tokens=20,
            output_tokens=8,
            latency_ms=10.0,
            request_payload={"messages": [{"role": "user", "content": prompt}]},
        )


def test_smoke_continuation_uses_only_unfinished_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "e5b4"
    smoke_dir = private_root / "smoke"
    smoke_dir.mkdir(parents=True)
    plan = [
        {
            "case_id": "case-01",
            "action": action,
            "task_kind": "GUIDANCE_ONLY",
            "guidance_prompt": f"prompt-{action}",
            "guidance_prompt_sha256": f"sha-{action}",
            "evidence_aliases": [],
        }
        for action in ("OFF", "STANDARD", "STRONG")
    ]
    off_payload = {
        "action_label_for_smoke_only": "OFF",
        "case_id": "case-01",
        "prompt_sha256": "sha-OFF",
        "alias_map": [],
        "response": {"text": "offline smoke", "finish_reason": "stop"},
    }
    off_path = smoke_dir / "OFF.json"
    off_path.write_text(json.dumps(off_payload), encoding="utf-8")
    off_bytes_before = off_path.read_bytes()
    client = FakeClient()
    monkeypatch.setattr(
        smoke_resume,
        "_read_json",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
        if Path(path).exists()
        else {"runtime": {"server_url": "http://127.0.0.1:8081"}},
    )
    monkeypatch.setattr(smoke_resume, "verify_protocol_lock", lambda _lock: "lock-sha")
    monkeypatch.setattr(smoke_resume, "verify_frozen_code", lambda _lock: None)
    monkeypatch.setattr(smoke_resume, "LlamaServerClient", lambda _url: client)
    monkeypatch.setattr(smoke_resume, "verify_server", lambda *_args: {"pid": 1})
    monkeypatch.setattr(smoke_resume, "verify_b3_identity", lambda _lock: {"b3": "same"})
    monkeypatch.setattr(smoke_resume, "load_verified_b2", lambda _lock: object())
    monkeypatch.setattr(smoke_resume, "make_execution_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(smoke_resume, "_wait_for_idle", lambda _client: None)

    report = smoke_resume.continue_smoke(private_root=private_root)

    assert client.prompts == ["prompt-STANDARD", "prompt-STRONG"]
    assert report["model_calls"] == 3
    assert report["continued_after_shared_slot_timeout"] is True
    assert [row["action"] for row in report["outputs"]] == ["OFF", "STANDARD", "STRONG"]
    assert off_path.read_bytes() == off_bytes_before


def test_smoke_continuation_skips_valid_intermediate_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "e5b4"
    smoke_dir = private_root / "smoke"
    smoke_dir.mkdir(parents=True)
    plan = [
        {
            "case_id": "case-01",
            "action": action,
            "task_kind": "GUIDANCE_ONLY",
            "guidance_prompt": f"prompt-{action}",
            "guidance_prompt_sha256": f"sha-{action}",
            "evidence_aliases": [],
        }
        for action in ("OFF", "STANDARD", "STRONG")
    ]
    payloads = {
        action: {
            "action_label_for_smoke_only": action,
            "case_id": "case-01",
            "prompt_sha256": f"sha-{action}",
            "alias_map": [],
            "response": {"text": f"{action} smoke", "finish_reason": "stop"},
        }
        for action in ("OFF", "STANDARD")
    }
    for action, payload in payloads.items():
        (smoke_dir / f"{action}.json").write_text(json.dumps(payload), encoding="utf-8")
    client = FakeClient()
    monkeypatch.setattr(smoke_resume, "_read_json", lambda path: json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {"runtime": {"server_url": "http://127.0.0.1:8081"}})
    monkeypatch.setattr(smoke_resume, "verify_protocol_lock", lambda _lock: "lock-sha")
    monkeypatch.setattr(smoke_resume, "verify_frozen_code", lambda _lock: None)
    monkeypatch.setattr(smoke_resume, "LlamaServerClient", lambda _url: client)
    monkeypatch.setattr(smoke_resume, "verify_server", lambda *_args: {"pid": 1})
    monkeypatch.setattr(smoke_resume, "verify_b3_identity", lambda _lock: {"b3": "same"})
    monkeypatch.setattr(smoke_resume, "load_verified_b2", lambda _lock: object())
    monkeypatch.setattr(smoke_resume, "make_execution_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(smoke_resume, "_wait_for_idle", lambda _client: None)

    report = smoke_resume.continue_smoke(private_root=private_root)

    assert client.prompts == ["prompt-STRONG"]
    assert [row["action"] for row in report["outputs"]] == ["OFF", "STANDARD", "STRONG"]
