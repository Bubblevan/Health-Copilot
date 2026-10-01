from __future__ import annotations

import hashlib
import http.client
import json
import subprocess
from pathlib import Path
import sys

import pytest

from tools.research.memory import preflight_mem3b0q_r4_candidate_local_smoke_grammar_v1 as grammar_preflight
from tools.research.memory import mem3b0q_r4_candidate_builder_v1 as candidate_builder
from tools.research.memory import run_mem3b0q_r4_candidate_local_smoke_v1 as smoke


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _locked_fixture(tmp_path: Path, monkeypatch) -> tuple[dict, bytes]:
    case, request, body = smoke._load_case_and_request()
    schema = request["response_format"]["json_schema"]["schema"]
    grammar_path = (
        smoke.DOCS / "mem3b0q_r4_candidate_local_smoke_grammar_preflight_v1.json"
    )
    grammar_bytes = grammar_path.read_bytes()
    grammar = json.loads(grammar_bytes)
    lock = {
        **smoke.EXPECTED_LOCK_FIELDS,
        "runner_sha256": _sha(Path(smoke.__file__).resolve().read_bytes()),
        "dependencies": {
            relative: _sha((smoke.ROOT / relative).read_bytes())
            for relative in smoke.EXPECTED_DEPENDENCIES
        },
        "request_sha256": _sha(body),
        "schema_sha256": _sha(smoke._canonical_json(schema)),
        "grammar_preflight": {
            "path": "docs/research/memory/mem3b0q_r4_candidate_local_smoke_grammar_preflight_v1.json",
            "sha256": _sha(grammar_bytes),
            "status": "PASS_GRAMMAR_CONVERSION_PRE_MODEL_LOAD",
            "schema_sha256": _sha(smoke._canonical_json(schema)),
            "cli_sha256": "48566cf6e2969464b799dbcac7393b3549f9efd6884688074dc803125dbafa85",
            "cli_version": "10068 (571d0d540)",
            "sampler_initialization": "NOT_VERIFIED",
        },
    }
    lock_path = tmp_path / "lock.json"
    lock_bytes = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    lock_path.write_bytes(lock_bytes)
    lock_path.with_suffix(".json.sha256").write_text(
        f"{_sha(lock_bytes)}  lock.json\n", encoding="ascii"
    )
    monkeypatch.setattr(smoke, "LOCK_PATH", lock_path)
    monkeypatch.setattr(smoke, "LOCK_SHA_PATH", lock_path.with_suffix(".json.sha256"))
    monkeypatch.setattr(smoke.frozen_gate, "_load_frozen_inputs", lambda: None)
    return case, body


def test_local_request_contract_does_not_require_openai_credentials(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    case, request, body = smoke._load_case_and_request()
    user_payload = json.loads(request["messages"][1]["content"])

    assert case["case_id"] == "R4C-05"
    assert request["model"] == smoke.frozen_gate.MODEL_PATH
    assert user_payload["source_id"] == "R4C-05"
    assert "expected_atoms" not in user_payload
    assert request["response_format"]["type"] == "json_schema"
    assert _sha(body)


def test_run_lock_and_sidecar_validate_exact_local_request(tmp_path, monkeypatch) -> None:
    case, body = _locked_fixture(tmp_path, monkeypatch)

    lock, lock_sha, loaded_case, loaded_body = smoke._load_and_validate_lock()

    assert lock["status"] == "FROZEN"
    assert lock["grammar_preflight"]["sampler_initialization"] == "NOT_VERIFIED"
    assert lock_sha == _sha(smoke.LOCK_PATH.read_bytes())
    assert loaded_case == case
    assert loaded_body == body


def test_run_lock_fails_closed_on_mutation_even_with_updated_sidecar(
    tmp_path, monkeypatch
) -> None:
    _locked_fixture(tmp_path, monkeypatch)
    lock = json.loads(smoke.LOCK_PATH.read_text(encoding="utf-8"))
    lock["request_sha256"] = "0" * 64
    lock_bytes = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    smoke.LOCK_PATH.write_bytes(lock_bytes)
    smoke.LOCK_SHA_PATH.write_text(
        f"{_sha(lock_bytes)}  lock.json\n", encoding="ascii"
    )

    with pytest.raises(smoke.CandidateSmokeError, match="run_lock_digest_mismatch"):
        smoke._load_and_validate_lock()


def test_strict_lock_parser_rejects_duplicate_keys(tmp_path, monkeypatch) -> None:
    _locked_fixture(tmp_path, monkeypatch)
    lock_bytes = smoke.LOCK_PATH.read_bytes().rstrip()[:-1] + b',"status":"FROZEN"}'
    smoke.LOCK_PATH.write_bytes(lock_bytes)
    smoke.LOCK_SHA_PATH.write_text(
        f"{_sha(lock_bytes)}  lock.json\n", encoding="ascii"
    )

    with pytest.raises(smoke.CandidateSmokeError, match="duplicate_json_key:status"):
        smoke._load_and_validate_lock()


def _process(pid: int = 100) -> dict:
    return {"pid": pid, "addresses": ["127.0.0.1"]}


def test_loopback_guard_allows_one_exact_post_and_allowlisted_gets(monkeypatch) -> None:
    sent = []

    class FakeConnection:
        def __init__(self, host, port, timeout=None):
            self.host, self.port, self.timeout = host, port, timeout

        def request(self, method, url, body=None, headers=None, *args, **kwargs):
            sent.append((method, url, body, headers))

    monkeypatch.setattr(smoke.http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(smoke.frozen_gate, "_ps_process_snapshot", _process)
    monkeypatch.setattr(smoke.frozen_gate, "_service_snapshot", lambda: {"raw_sha256": {}})
    monkeypatch.setattr(smoke, "validate_runtime_snapshot", lambda process, service: None)

    expected = b'{"request":"frozen"}'
    guard = smoke.LoopbackOneShotGuard(expected)
    with guard:
        connection = FakeConnection(smoke.frozen_gate.HOST, smoke.frozen_gate.PORT)
        with pytest.raises(smoke.CandidateSmokeError, match="post_before_preflight"):
            connection.request(
                "POST",
                smoke.frozen_gate.ENDPOINT_PATH,
                body=expected,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )

        guard.arm_after_preflight(_process(), {"raw_sha256": {}})
        connection.request("GET", "/health", headers={"Accept": "application/json"})
        connection.request(
            "POST",
            smoke.frozen_gate.ENDPOINT_PATH,
            body=expected,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with pytest.raises(smoke.CandidateSmokeError, match="completion_post_limit_exceeded"):
            connection.request(
                "POST",
                smoke.frozen_gate.ENDPOINT_PATH,
                body=expected,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
        guard.verify_postflight()

    assert [(method, path) for method, path, *_ in sent] == [
        ("GET", "/health"),
        ("POST", smoke.frozen_gate.ENDPOINT_PATH),
    ]
    assert guard.post_attempts == 1
    assert [event["checkpoint"] for event in guard.events] == [
        "initial_preflight",
        "immediately_before_post",
        "postflight",
    ]


@pytest.mark.parametrize(
    ("host", "port", "method", "path", "body", "headers", "error"),
    [
        ("example.com", 8081, "GET", "/health", None, None, "connection_not_frozen_loopback"),
        ("127.0.0.1", 8081, "GET", "/admin", None, None, "get_path_not_allowlisted"),
        ("127.0.0.1", 8081, "POST", "/other", b"x", {}, "http_method_or_path_not_allowlisted"),
        ("127.0.0.1", 8081, "POST", "/v1/chat/completions", b"x", {}, "completion_body_not_exact_locked_request"),
    ],
)
def test_loopback_guard_blocks_unpinned_routes(
    monkeypatch, host, port, method, path, body, headers, error
) -> None:
    sent = []

    class FakeConnection:
        def __init__(self, host, port, timeout=None):
            self.host, self.port = host, port

        def request(self, method, url, body=None, headers=None, *args, **kwargs):
            sent.append((method, url))

    monkeypatch.setattr(smoke.http.client, "HTTPConnection", FakeConnection)
    guard = smoke.LoopbackOneShotGuard(b"expected")
    with guard:
        connection = FakeConnection(host, port)
        with pytest.raises(smoke.CandidateSmokeError, match=error):
            connection.request(method, path, body=body, headers=headers)

    assert sent == []
    assert guard.post_attempts == 0


def test_https_is_denied_and_process_change_blocks_post(monkeypatch) -> None:
    sent = []
    process_calls = iter([_process(101)])

    class FakeConnection:
        def __init__(self, host, port, timeout=None):
            self.host, self.port = host, port

        def request(self, *args, **kwargs):
            sent.append(args)

    monkeypatch.setattr(smoke.http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(smoke.frozen_gate, "_ps_process_snapshot", lambda: next(process_calls))
    monkeypatch.setattr(smoke.frozen_gate, "_service_snapshot", lambda: {})
    monkeypatch.setattr(smoke, "validate_runtime_snapshot", lambda process, service: None)
    guard = smoke.LoopbackOneShotGuard(b"expected")

    with guard:
        guard.arm_after_preflight(_process(100), {})
        with pytest.raises(smoke.CandidateSmokeError, match="server_process_changed"):
            FakeConnection("127.0.0.1", 8081).request(
                "POST",
                "/v1/chat/completions",
                body=b"expected",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
        with pytest.raises(smoke.CandidateSmokeError, match="https_outbound_forbidden"):
            smoke.http.client.HTTPSConnection.request(object(), "GET", "https://example.com")

    assert sent == []
    assert guard.post_attempts == 0


def test_existing_run_directory_is_never_reused(tmp_path, monkeypatch) -> None:
    output = tmp_path / "already-consumed"
    output.mkdir()
    case, _, body = smoke._load_case_and_request()
    monkeypatch.setattr(smoke, "OUTPUT_DIR", output)
    monkeypatch.setattr(smoke, "_load_and_validate_lock", lambda: ({"run_id": output.name}, "a" * 64, case, body))
    calls = []
    monkeypatch.setattr(smoke.frozen_gate, "_service_snapshot", lambda: calls.append(True))

    with pytest.raises(FileExistsError):
        smoke._run_once()

    assert calls == []


def _expected_proposal_content(case: dict) -> str:
    expected = case["expected_atoms"][0]
    candidates = candidate_builder.build_candidates(
        case["proposition_text"],
        smoke.frozen_r4.ALIASES,
        source_id=case["case_id"],
    )["candidates"]
    atom = {}
    for field in ("owner", "object", "attribute"):
        match = next(
            row
            for row in candidates[field]
            if row["canonical_id"] == expected[f"{field}_id"]
            and row["source_span"].casefold()
            == expected[f"{field}_span"].casefold()
        )
        atom[f"{field}_candidate_id"] = match["candidate_id"]
    atom["value_span"] = expected["value_span"]
    atom["cardinality_proposal"] = expected["cardinality"]
    return json.dumps(
        {"source_id": case["case_id"], "atoms": [atom], "abstention_reason": "NONE"}
    )


@pytest.mark.parametrize(
    ("content", "expected_status"),
    [(None, "QUALITY_PASS"), ("not-json", "QUALITY_FAILURE")],
)
def test_simulated_completion_separates_exact_quality_from_bad_content(
    tmp_path, monkeypatch, content, expected_status
) -> None:
    case, _body = _locked_fixture(tmp_path, monkeypatch)
    output_dir = tmp_path / smoke.OUTPUT_DIR.name
    monkeypatch.setattr(smoke, "OUTPUT_DIR", output_dir)
    response_content = (
        _expected_proposal_content(case) if content is None else content
    )

    class FakeResponse:
        status = 200

        def getheader(self, name):
            return "application/json" if name.casefold() == "content-type" else None

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": response_content}}]}
            ).encode()

    sent = []

    class FakeConnection:
        def __init__(self, host, port, timeout=None):
            self.host, self.port = host, port

        def request(self, method, url, body=None, headers=None, *args, **kwargs):
            sent.append((method, url, body, headers))

        def getresponse(self):
            return FakeResponse()

        def close(self):
            pass

    monkeypatch.setattr(smoke.http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(smoke.frozen_gate, "_ps_process_snapshot", _process)
    monkeypatch.setattr(smoke.frozen_gate, "_service_snapshot", lambda: {"raw_sha256": {}})
    monkeypatch.setattr(smoke, "validate_runtime_snapshot", lambda process, service: None)

    result = smoke._run_once()

    assert result["status"] == expected_status
    assert result["completion_post_attempts"] == 1
    assert result["retry_count"] == 0
    assert result["hosted_calls"] == 0
    assert len(sent) == 1
    assert (output_dir / "response_body.bin").is_file()
    assert (output_dir / "result.json").is_file()


def test_simulated_transport_failure_is_infra_not_quality(tmp_path, monkeypatch) -> None:
    case, _body = _locked_fixture(tmp_path, monkeypatch)
    output_dir = tmp_path / smoke.OUTPUT_DIR.name
    monkeypatch.setattr(smoke, "OUTPUT_DIR", output_dir)

    class FakeConnection:
        def __init__(self, host, port, timeout=None):
            self.host, self.port = host, port

        def request(self, method, url, body=None, headers=None, *args, **kwargs):
            raise OSError("simulated transport interruption")

        def close(self):
            pass

    monkeypatch.setattr(smoke.http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(smoke.frozen_gate, "_ps_process_snapshot", _process)
    monkeypatch.setattr(smoke.frozen_gate, "_service_snapshot", lambda: {"raw_sha256": {}})
    monkeypatch.setattr(smoke, "validate_runtime_snapshot", lambda process, service: None)

    result = smoke._run_once()

    assert case["case_id"] == "R4C-05"
    assert result["status"] == "INFRA_FAILURE"
    assert result["completion_post_attempts"] == 1
    assert result["retry_count"] == 0
    assert "OSError" in result["failure"]
    assert not (output_dir / "response_body.bin").exists()


@pytest.mark.parametrize("argv", [[], ["--run-r4c-05-onc"]])
def test_cli_needs_exact_explicit_run_flag(monkeypatch, argv) -> None:
    invoked = []
    monkeypatch.setattr(smoke, "_run_once", lambda: invoked.append(True))
    monkeypatch.setattr(sys, "argv", ["candidate-smoke", *argv])

    with pytest.raises(SystemExit):
        smoke.main()

    assert invoked == []


def test_response_envelope_duplicate_keys_are_rejected() -> None:
    with pytest.raises(smoke.CandidateSmokeError, match="duplicate_json_key:choices"):
        smoke._response_content(b'{"choices":[],"choices":[]}')


def test_grammar_preflight_distinguishes_schema_parse_from_missing_model() -> None:
    valid = grammar_preflight._classify_cli_result(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="Loading model... Error: the server exited before becoming ready",
            stderr="llama_model_load: failed to load model from absent.gguf",
        )
    )
    malformed = grammar_preflight._classify_cli_result(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="error while handling argument --json-schema-file",
            stderr="JSON parse_error invalid json",
        )
    )

    assert valid["missing_model_failure_detected"]
    assert not valid["schema_or_grammar_error_detected"]
    assert malformed["schema_or_grammar_error_detected"]
    assert not malformed["missing_model_failure_detected"]
