from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.research.memory import run_mem3b0q_r4_candidate_local_smoke_v1 as smoke
from tools.research.memory import run_mem3b0q_r4_atomwise_candidate_devset_v1 as runner


ROOT = Path(__file__).resolve().parents[1]
PRIOR_RESPONSE = (
    ROOT
    / "runs/memory/mem3/mem3b0q-r4-candidate-local-smoke-v1/response_body.bin"
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _locked_fixture(tmp_path: Path, monkeypatch) -> dict:
    pack, requests = runner._build_requests()
    lock = {
        **runner.EXPECTED_LOCK_FIELDS,
        "runner_sha256": _sha(Path(runner.__file__).resolve().read_bytes()),
        "dependencies": {
            relative: _sha((runner.ROOT / relative).read_bytes())
            for relative in runner.DEPENDENCY_PATHS
        },
        "requests": {
            case_id: {
                "request_sha256": row["request_sha256"],
                "schema_sha256": row["schema_sha256"],
            }
            for case_id, row in requests.items()
        },
        "replayed_case_artifacts": runner._prior_case_artifacts(),
    }
    lock_path = tmp_path / "lock.json"
    lock_bytes = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    lock_path.write_bytes(lock_bytes)
    lock_sha_path = lock_path.with_suffix(".json.sha256")
    lock_sha_path.write_text(f"{_sha(lock_bytes)}  lock.json\n", encoding="ascii")
    monkeypatch.setattr(runner, "LOCK_PATH", lock_path)
    monkeypatch.setattr(runner, "LOCK_SHA_PATH", lock_sha_path)
    monkeypatch.setattr(runner.frozen_gate, "_load_frozen_inputs", lambda: None)
    return lock


def test_devset_requests_are_local_and_exclude_oracle_fields(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    pack, requests = runner._build_requests()

    assert tuple(row["case_id"] for row in pack["cases"]) == runner.CASE_IDS
    assert tuple(requests) == runner.CASE_IDS
    assert runner.POST_CASE_IDS == ("R4C-01", "R4C-02", "R4C-03", "R4C-04", "R4C-06")
    for case_id, record in requests.items():
        request = record["request"]
        user_payload = json.loads(request["messages"][1]["content"])
        assert request["model"] == runner.frozen_gate.MODEL_PATH
        assert set(user_payload) == {"source_id", "proposition_text", "typed_candidates"}
        assert case_id == user_payload["source_id"]
        assert "expected_atoms" not in request["messages"][1]["content"]


def test_r4c05_recorded_response_is_replayed_through_atomwise_scoring() -> None:
    pack, requests = runner._build_requests()
    case = runner._case_by_id(pack, "R4C-05")
    response_bytes = PRIOR_RESPONSE.read_bytes()
    envelope, content = smoke._response_content(response_bytes)
    scored = runner._validate_proposal(
        case,
        content,
        request=requests["R4C-05"]["request"],
        response_sha256=_sha(response_bytes),
        http_status=200,
        attempt_count=0,
        retry_count=0,
        latency_ms=None,
        usage=envelope.get("usage"),
    )

    assert scored["status"] == "QUALITY_PASS"
    assert scored["whole_document_guard_accepts"] is False
    assert scored["expected_atoms"] == scored["admitted_atoms"] == 1
    assert scored["matched_expected_atoms"] == 1
    assert scored["quarantined_atoms"][0]["source_atom_index"] == 1
    assert scored["atom_precision"] == scored["atom_recall"] == 1.0


def test_run_lock_pins_all_six_requests_and_rejects_tampered_body_hash(
    tmp_path, monkeypatch
) -> None:
    lock = _locked_fixture(tmp_path, monkeypatch)
    loaded, lock_sha, pack, requests = runner._load_lock()
    assert loaded == lock
    assert lock_sha == _sha(runner.LOCK_PATH.read_bytes())
    assert len(pack["cases"]) == 6
    assert len(requests) == 6

    lock["requests"]["R4C-01"]["request_sha256"] = "0" * 64
    lock_bytes = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    runner.LOCK_PATH.write_bytes(lock_bytes)
    runner.LOCK_SHA_PATH.write_text(
        f"{_sha(lock_bytes)}  lock.json\n", encoding="ascii"
    )
    with pytest.raises(runner.DevsetRunError, match="request_hash_map_mismatch"):
        runner._load_lock()


def test_aggregate_separates_quality_from_infrastructure() -> None:
    rows = [
        {
            "case_id": "R4C-01",
            "status": "QUALITY_FAILURE",
            "expected_atoms": 2,
            "admitted_atoms": 1,
            "matched_expected_atoms": 1,
            "quarantined_atoms": [{"reason": "binding"}],
            "strict_document_admitted_atom_count": 0,
            "completion_post_attempts": 1,
            "retry_count": 0,
            "hosted_calls": 0,
            "memory_store_mutations": 0,
        },
        {
            "case_id": "R4C-02",
            "status": "INFRA_FAILURE",
            "completion_post_attempts": 1,
            "retry_count": 0,
            "hosted_calls": 0,
            "memory_store_mutations": 0,
        },
    ]
    aggregate = runner._aggregate(rows)

    assert aggregate["completed_quality_cases"] == 1
    assert aggregate["quality_failure_cases"] == 1
    assert aggregate["infra_or_preflight_failures"] == 1
    assert aggregate["expected_atoms"] == 2
    assert aggregate["matched_expected_atoms"] == 1
    assert aggregate["new_post_attempts"] == 2
    assert aggregate["new_retries"] == 0
