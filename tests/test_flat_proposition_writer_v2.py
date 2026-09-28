from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.research.memory import flat_proposition_writer_v2 as writer


def _raw_spans() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pieces = ["A", "BB", "quote, with comma"]
    for index, content in enumerate(["intro", "middle", "example", "text", "other", "turn"]):
        rows.append(
            {
                "source_turn_index": 0,
                "source_span_index": index,
                "char_start": sum(len(value) for value in ["intro", "middle", "example", "text", "other", "turn"][:index]),
                "char_end": sum(len(value) for value in ["intro", "middle", "example", "text", "other", "turn"][: index + 1]),
                "role": "user",
                "content": content,
            }
        )
    offset = 0
    for index, content in enumerate(pieces):
        rows.append(
            {
                "source_turn_index": 8,
                "source_span_index": index,
                "char_start": offset,
                "char_end": offset + len(content),
                "role": "user",
                "content": content,
            }
        )
        offset += len(content)
    rows.append(
        {
            "source_turn_index": 9,
            "source_span_index": 0,
            "char_start": 0,
            "char_end": len("assistant example dialogue"),
            "role": "assistant",
            "content": "assistant example dialogue",
        }
    )
    return rows


@pytest.fixture
def catalog() -> list[dict[str, object]]:
    return writer.build_source_span_catalog(_raw_spans())


def _packet(
    kind: str,
    ref: str,
    *,
    text: str = "The user has an observed fact.",
) -> bytes:
    return json.dumps(
        {
            "propositions": [
                {
                    "proposition_text": text,
                    "memory_kind": kind,
                    "entity_key_candidate": "user",
                    "attribute_key_candidate": "observed_fact",
                    "value_text": "observed",
                    "evidence_refs": [ref],
                }
            ]
        },
        ensure_ascii=False,
    ).encode("utf-8")


def test_punctuation_is_harness_copied_from_exact_frozen_span(catalog):
    normalized = writer.validate_packet(_packet("user_fact", "S0008"), catalog)
    evidence = normalized["propositions"][0]["evidence"][0]
    assert evidence["evidence_quote"] == "quote, with comma"
    assert evidence["source_role"] == "user"
    assert evidence["source_turn_index"] == 8
    assert evidence["source_span_index"] == 2
    assert evidence["char_end"] - evidence["char_start"] == len(evidence["evidence_quote"])


def test_compact_reference_cannot_bind_to_a_different_turn(catalog):
    span = next(row for row in catalog if row["evidence_ref"] == "S0008")
    assert (span["source_turn_index"], span["source_span_index"]) == (8, 2)
    packet = writer.validate_packet(_packet("user_fact", "S0008"), catalog)
    assert packet["propositions"][0]["evidence"][0]["evidence_ref"] == "S0008"


def test_unknown_evidence_reference_is_structured(catalog):
    with pytest.raises(writer.ExtractionValidationError) as error:
        writer.validate_packet(_packet("user_fact", "S9999"), catalog)
    assert error.value.code == "UNKNOWN_EVIDENCE_REF"
    assert error.value.evidence_ref == "S9999"


def test_assistant_hypothetical_cannot_become_user_fact(catalog):
    with pytest.raises(writer.ExtractionValidationError) as error:
        writer.validate_packet(_packet("user_fact", "S0009"), catalog)
    assert error.value.code == "MEMORY_KIND_ROLE_MISMATCH"


def test_genuine_assistant_recommendation_uses_assistant_evidence(catalog):
    packet = writer.validate_packet(
        _packet("assistant_recommendation", "S0009", text="The assistant recommended a sample."),
        catalog,
    )
    assert packet["propositions"][0]["evidence"][0]["source_role"] == "assistant"


def test_transient_request_exclusion_is_explicit_in_frozen_prompt():
    prompt = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "research"
        / "memory"
        / "flat_proposition_extractor_v2.txt"
    ).read_text(encoding="utf-8")
    assert "one-off requests for advice" in prompt
    assert "generic help-me or can-you-suggest intents" in prompt
    assert '"I am vegetarian. Can you help me plan dinner?"' in prompt


def test_dynamic_schema_enumerates_only_the_current_catalog(catalog):
    schema = writer.dynamic_output_schema(catalog)
    enum = schema["properties"]["propositions"]["items"]["properties"]["evidence_refs"]["items"]["enum"]
    assert enum == [row["evidence_ref"] for row in catalog]
    assert "S9999" not in enum


def _cache_identity(request, catalog, session_id, prompt_sha, contract_sha):
    identity = {
        "stage": "MEM-3A.1",
        "contract_sha256": contract_sha,
        "prompt_sha256": prompt_sha,
        "session_identity_sha256": session_id,
        "catalog_sha256": writer.catalog_sha256(catalog),
        "request_sha256": writer.sha256_bytes(writer.canonical_json(request)),
    }
    return writer.sha256_bytes(writer.canonical_json(identity))


def test_response_captured_crash_recovers_without_provider_call(tmp_path, catalog):
    request = {"model": "local", "schema": writer.dynamic_output_schema(catalog)}
    session_id, prompt_sha, contract_sha = "session-1", "prompt-1", "contract-1"
    cache_identity = _cache_identity(request, catalog, session_id, prompt_sha, contract_sha)
    cache_dir = tmp_path / contract_sha / cache_identity
    raw = _packet("user_fact", "S0008")
    raw_sha = hashlib.sha256(raw).hexdigest()
    writer.atomic_write_bytes(cache_dir / "raw_response.txt", raw)
    writer.atomic_write_bytes(
        cache_dir / "response_meta.json",
        writer.canonical_json({"response_sha256": raw_sha, "http_status": 200}) + b"\n",
    )
    journal = writer._Journal(cache_dir / "journal.jsonl")
    journal.append({"state": "STARTED", "cache_identity_sha256": cache_identity})
    journal.append(
        {
            "state": "RESPONSE_CAPTURED",
            "cache_identity_sha256": cache_identity,
            "response_sha256": raw_sha,
        }
    )

    def forbidden_provider(_request):
        pytest.fail("recovery issued a duplicate provider call")

    packet, ledger = writer.execute_or_resume(
        request=request,
        catalog=catalog,
        session_identity_sha256=session_id,
        prompt_sha256=prompt_sha,
        contract_sha256=contract_sha,
        local_cache_root=tmp_path,
        provider=forbidden_provider,
    )
    assert packet["propositions"][0]["evidence"][0]["evidence_quote"] == "quote, with comma"
    assert ledger["provider_calls"] == 1
    assert ledger["provider_calls_this_resume"] == 0
    assert journal.read()[-1]["state"] == "COMPLETE_SUCCESS"


def test_validation_error_subtype_survives_complete_failure_journal(tmp_path, catalog):
    request = {"model": "local", "schema": writer.dynamic_output_schema(catalog)}
    identity = "session-failure"

    with pytest.raises(writer.WriterQualificationFailure) as error:
        writer.execute_or_resume(
            request=request,
            catalog=catalog,
            session_identity_sha256=identity,
            prompt_sha256="prompt",
            contract_sha256="contract",
            local_cache_root=tmp_path,
            provider=lambda _request: (200, _packet("user_fact", "S0009"), "application/json"),
        )
    assert error.value.error["code"] == "MEMORY_KIND_ROLE_MISMATCH"
    cache_dirs = list((tmp_path / "contract").iterdir())
    events = writer._Journal(cache_dirs[0] / "journal.jsonl").read()
    assert events[-1]["state"] == "COMPLETE_FAILURE"
    assert events[-1]["error"]["code"] == "MEMORY_KIND_ROLE_MISMATCH"


def test_frozen_response_is_authoritative_if_provider_would_drift(tmp_path, catalog):
    request = {"model": "local", "schema": writer.dynamic_output_schema(catalog)}
    args = {
        "request": request,
        "catalog": catalog,
        "session_identity_sha256": "session-drift",
        "prompt_sha256": "prompt",
        "contract_sha256": "contract",
        "local_cache_root": tmp_path,
    }
    calls = 0

    def first(_request):
        nonlocal calls
        calls += 1
        return 200, _packet("user_fact", "S0008", text="Frozen first response."), "application/json"

    original, _ = writer.execute_or_resume(**args, provider=first)

    def would_drift(_request):
        nonlocal calls
        calls += 1
        return 200, _packet("user_fact", "S0008", text="Different later output."), "application/json"

    resumed, ledger = writer.execute_or_resume(**args, provider=would_drift)
    assert calls == 1
    assert resumed == original
    assert resumed["propositions"][0]["proposition_text"] == "Frozen first response."
    assert ledger["provider_calls_this_resume"] == 0


def test_writer_input_hides_raw_location_metadata(catalog):
    request = writer.writer_request(
        session_date="2026-01-01",
        catalog=catalog,
        system_prompt="system",
        model_alias="local",
    )
    payload = json.loads(request["messages"][1]["content"])
    assert set(payload) == {"session_date", "source_spans"}
    assert set(payload["source_spans"][0]) == {"evidence_ref", "role", "content"}
    assert "source_turn_index" not in request["messages"][1]["content"]
    schema_refs = request["response_format"]["json_schema"]["schema"]["properties"]["propositions"]["items"]["properties"]["evidence_refs"]["items"]["enum"]
    assert schema_refs == [row["evidence_ref"] for row in catalog]
