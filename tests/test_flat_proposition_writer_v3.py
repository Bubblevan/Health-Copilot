from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.research.memory import flat_proposition_writer_v2 as v2
from tools.research.memory import flat_proposition_writer_v3 as v3


def _catalog():
    user_text = "I like swimming every weekend."
    assistant_text = "The assistant suggested swimming."
    raw_spans = [
        {
            "source_turn_index": 0,
            "source_span_index": 0,
            "char_start": 0,
            "char_end": len(user_text),
            "role": "user",
            "content": user_text,
        },
        {
            "source_turn_index": 1,
            "source_span_index": 0,
            "char_start": 0,
            "char_end": len(assistant_text),
            "role": "assistant",
            "content": assistant_text,
        },
    ]
    return v2.build_source_span_catalog(raw_spans)


def _packet(propositions):
    return json.dumps({"propositions": propositions}, ensure_ascii=False).encode("utf-8")


def _envelope(content: bytes) -> bytes:
    return json.dumps(
        {
            "choices": [{"message": {"content": content.decode("utf-8")}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 9},
            "model": "local-qwen",
        }
    ).encode("utf-8")


def test_v3_generated_schema_has_only_text_and_evidence_refs():
    schema = v3.dynamic_output_schema(_catalog())
    fields = set(schema["properties"]["propositions"]["items"]["properties"])
    assert fields == {"proposition_text", "evidence_refs"}
    assert schema["properties"]["propositions"]["items"]["required"] == [
        "proposition_text",
        "evidence_refs",
    ]


def test_v3_writer_capacity_is_explicit_and_default_remains_historical_4k():
    catalog = _catalog()
    kwargs = {
        "session_date": "2026-09-29",
        "catalog": catalog,
        "system_prompt": "frozen v3 prompt",
        "model_alias": "local-qwen",
    }
    assert v3.writer_request(**kwargs)["max_tokens"] == 4096
    assert v3.writer_request(**kwargs, max_tokens=16384)["max_tokens"] == 16384
    with pytest.raises(ValueError, match="frozen supported capacity"):
        v3.writer_request(**kwargs, max_tokens=32768)


@pytest.mark.parametrize(
    "legacy_field",
    ["memory_kind", "entity_key_candidate", "attribute_key_candidate", "value_text"],
)
def test_v3_rejects_legacy_semantic_fields(legacy_field):
    proposition = {"proposition_text": "A durable observation.", "evidence_refs": ["S0000"]}
    proposition[legacy_field] = "not part of v3"
    with pytest.raises(v2.ExtractionValidationError) as error:
        v3.validate_packet(_packet([proposition]), _catalog())
    assert error.value.code == "UNEXPECTED_OUTPUT_FIELD"
    assert error.value.field == legacy_field


def test_v3_accepts_assistant_fact_without_semantic_role_gate():
    normalized = v3.validate_packet(
        _packet(
            [
                {
                    "proposition_text": "The assistant suggested swimming.",
                    "evidence_refs": ["S0001"],
                }
            ]
        ),
        _catalog(),
    )
    proposition = normalized["propositions"][0]
    assert proposition["source_authority"] == "assistant"
    assert proposition["evidence"][0]["source_role"] == "assistant"


def test_v3_derives_mixed_authority_from_actual_span_roles():
    normalized = v3.validate_packet(
        _packet(
            [
                {
                    "proposition_text": "A proposition supported across speakers.",
                    "evidence_refs": ["S0000", "S0001"],
                }
            ]
        ),
        _catalog(),
    )
    assert normalized["propositions"][0]["source_authority"] == "mixed"


def test_v3_reconstructs_exact_frozen_quote_turn_span_and_offsets():
    normalized = v3.validate_packet(
        _packet(
            [
                {
                    "proposition_text": "The assistant suggested swimming.",
                    "evidence_refs": ["S0001"],
                }
            ]
        ),
        _catalog(),
    )
    evidence = normalized["propositions"][0]["evidence"][0]
    assert evidence == {
        "evidence_ref": "S0001",
        "source_turn_index": 1,
        "source_span_index": 0,
        "source_role": "assistant",
        "char_start": 0,
        "char_end": len("The assistant suggested swimming."),
        "evidence_quote": "The assistant suggested swimming.",
        "content_sha256": v2.sha256_bytes(b"The assistant suggested swimming."),
    }


def test_semantically_odd_but_structural_packet_is_frozen_without_retry(tmp_path):
    catalog = _catalog()
    request = {
        "model": "local",
        "response_format": {"schema": v3.dynamic_output_schema(catalog)},
    }
    provider_calls = []

    def provider(_request):
        provider_calls.append(1)
        return (
            200,
            _envelope(
                _packet(
                    [
                        {
                            "proposition_text": "The user is an astronaut with 900 followers.",
                            "evidence_refs": ["S0000"],
                        }
                    ]
                )
            ),
            "application/json",
        )

    packet, ledger = v2.execute_or_resume(
        request=request,
        catalog=catalog,
        session_identity_sha256="session-v3",
        prompt_sha256="prompt-v3",
        contract_sha256="contract-v3",
        local_cache_root=tmp_path,
        provider=provider,
        stage_identity="MEM-3A.2-v3-test",
        packet_validator=v3.validate_packet,
        packet_validator_sha256=v2.sha256_bytes(Path(v3.__file__).read_bytes()),
    )
    assert len(provider_calls) == 1
    assert packet["propositions"][0]["proposition_text"].startswith("The user is an astronaut")
    assert ledger["validation"] == "passed"
    assert ledger["provider_calls"] == 1
