import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
TOOLS = ROOT / "tools" / "research" / "memory"
sys.path.insert(0, str(TOOLS))
_spec = importlib.util.spec_from_file_location(
    "mem1d4_reader_runner_test", TOOLS / "run_mem1d4_final_reader.py"
)
mem1d4 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(mem1d4)


def test_reader_only_pipeline_reuses_frozen_inputs_and_records_exact_local_calls(tmp_path, monkeypatch):
    artifact_root = ROOT / f".tmp-mem1d4-test-{tmp_path.name}"
    run_dir = artifact_root / "run"
    review_path = artifact_root / "docs" / "mem_1d4_reader_review.json"
    report_path = artifact_root / "docs" / "mem_1d4_final_reader_contract.md"
    assert not artifact_root.exists()

    calls = []

    class FakeResponse:
        status_code = 200

        def __init__(self, request):
            self.request = request

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{
                    "message": {"content": "Local reader diagnostic answer."},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 5,
                },
            }

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json):
            assert url == f"{mem1d4.LOCAL_BASE_URL}/chat/completions"
            calls.append(json)
            return FakeResponse(json)

    monkeypatch.setattr(mem1d4, "D4_RUN", run_dir)
    monkeypatch.setattr(mem1d4, "REVIEW_PATH", review_path)
    monkeypatch.setattr(mem1d4, "REPORT_PATH", report_path)
    monkeypatch.setattr(mem1d4.d3, "_runtime_preflight", lambda manifest: {
        "model": "health-memory-qwen3-8b",
        "model_sha256": "frozen-model-sha",
        "test_runtime": True,
    })
    monkeypatch.setattr(mem1d4.d3, "_render_and_tokenize", lambda client, messages: (17, "render-sha"))
    monkeypatch.setattr(mem1d4.httpx, "Client", FakeClient)

    try:
        result = mem1d4.run()
        manifest = mem1d4.d3.json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert result["gate"] == "MEM1D4_FINAL_READER_CONTRACT_FROZEN=YES"
        assert len(calls) == 50
        assert manifest["reader_calls_successful"] == 50
        assert manifest["memory_system_calls"] == 0
        assert manifest["embedding_calls"] == 0
        assert manifest["judge_calls"] == 0
        assert manifest["hosted_api_calls"] == 0
        assert all(call["model"] == "health-memory-qwen3-8b" for call in calls)
        assert all(
            "Current Date: " in call["messages"][1]["content"]
            for call in calls
        )
        assert (run_dir / "predictions_v3.sha256").is_file()
        assert (run_dir / "call_ledger.sha256").is_file()
        assert report_path.is_file() and review_path.is_file()
    finally:
        shutil.rmtree(artifact_root)
