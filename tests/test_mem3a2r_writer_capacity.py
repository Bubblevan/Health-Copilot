from __future__ import annotations

import inspect
import json

import pytest

from tools.research.memory import flat_proposition_writer_v2 as v2
from tools.research.memory import flat_proposition_writer_v3 as v3
from tools.research.memory import run_mem3a2r_writer_capacity as mem3a2r


def _catalog():
    text = "I prefer swimming."
    return v2.build_source_span_catalog(
        [
            {
                "source_turn_index": 0,
                "source_span_index": 0,
                "char_start": 0,
                "char_end": len(text),
                "role": "user",
                "content": text,
            }
        ]
    )


def _envelope(content: bytes, *, finish_reason: str = "stop", completion_tokens: int = 9) -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "message": {"content": content.decode("utf-8")},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": completion_tokens},
            "model": "local-qwen",
        }
    ).encode("utf-8")


def _packet(text: str = "The user prefers swimming.") -> bytes:
    return json.dumps(
        {"propositions": [{"proposition_text": text, "evidence_refs": ["S0000"]}]}
    ).encode("utf-8")


def _request(cap: int):
    return v3.writer_request(
        session_date="2026-09-29",
        catalog=_catalog(),
        system_prompt="frozen v3 prompt",
        model_alias="local-qwen",
        max_tokens=cap,
    )


def test_16k_request_is_exact_and_execution_position_does_not_change_content():
    request = _request(16384)
    reordered_request = _request(16384)
    assert request["max_tokens"] == 16384
    assert v2.sha256_bytes(v2.canonical_json(request)) == v2.sha256_bytes(
        v2.canonical_json(reordered_request)
    )


def test_historical_truncation_identity_is_first_without_reordering_the_rest():
    sessions = {f"session-{index}": {} for index in range(477)}
    keys = list(sessions)
    sessions[mem3a2r.HISTORICAL_SENTINEL] = sessions.pop("session-54")
    canonical = {key: sessions[key] for key in keys[:54]}
    canonical[mem3a2r.HISTORICAL_SENTINEL] = sessions[mem3a2r.HISTORICAL_SENTINEL]
    canonical.update({key: sessions[key] for key in keys[55:]})
    ordered = mem3a2r._execution_order(canonical)
    assert ordered[0] == mem3a2r.HISTORICAL_SENTINEL
    assert ordered[1:] == [key for key in canonical if key != mem3a2r.HISTORICAL_SENTINEL]


def test_old_4k_cache_cannot_satisfy_new_16k_request(tmp_path):
    catalog = _catalog()
    calls = []

    def provider(request):
        calls.append(request["max_tokens"])
        return 200, _envelope(_packet()), "application/json"

    for cap in (4096, 16384):
        request = _request(cap)
        v2.execute_or_resume(
            request=request,
            catalog=catalog,
            session_identity_sha256="same-session",
            prompt_sha256="same-prompt",
            contract_sha256="same-contract",
            local_cache_root=tmp_path,
            provider=provider,
            stage_identity="MEM-3A.2R-test",
            packet_validator=v3.validate_packet,
        )
    assert calls == [4096, 16384]


def test_16k_length_response_is_fatal_without_retry_or_cap_increase(tmp_path):
    catalog = _catalog()
    request = _request(16384)
    calls = []

    def provider(_request):
        calls.append(1)
        return (
            200,
            _envelope(_packet(), finish_reason="length", completion_tokens=16384),
            "application/json",
        )

    with pytest.raises(v2.WriterQualificationFailure) as error:
        v2.execute_or_resume(
            request=request,
            catalog=catalog,
            session_identity_sha256="sentinel-session",
            prompt_sha256="prompt",
            contract_sha256="contract",
            local_cache_root=tmp_path,
            provider=provider,
            stage_identity="MEM-3A.2R-test",
            packet_validator=v3.validate_packet,
        )
    assert error.value.error["code"] == "COMPLETION_TRUNCATED"
    assert request["max_tokens"] == 16384
    assert calls == [1]


def test_redundant_high_proposition_packet_is_accepted_without_dedup(tmp_path):
    catalog = _catalog()
    redundant = json.dumps(
        {
            "propositions": [
                {"proposition_text": "The user prefers swimming.", "evidence_refs": ["S0000"]},
                {"proposition_text": "The user prefers swimming.", "evidence_refs": ["S0000"]},
            ]
        }
    ).encode("utf-8")
    calls = []

    def provider(_request):
        calls.append(1)
        return 200, _envelope(redundant), "application/json"

    normalized, ledger = v2.execute_or_resume(
        request=_request(16384),
        catalog=catalog,
        session_identity_sha256="redundant-session",
        prompt_sha256="prompt",
        contract_sha256="contract",
        local_cache_root=tmp_path,
        provider=provider,
        stage_identity="MEM-3A.2R-test",
        packet_validator=v3.validate_packet,
    )
    assert len(normalized["propositions"]) == 2
    assert ledger["validation"] == "passed"
    assert calls == [1]


def test_label_barrier_remains_after_prediction_and_reader_freeze():
    source = inspect.getsource(mem3a2r.base._downstream)
    assert source.index("flat.PREDICTIONS_PATH") < source.index("flat._labels_after_freeze()")
    assert source.index("flat.READER_LEDGER_PATH") < source.index("flat._labels_after_freeze()")
