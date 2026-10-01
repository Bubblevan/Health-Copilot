from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pytest

from tools.research.memory import mem3b0q_r4


MEMORY_DOCS = Path(__file__).parents[1] / "docs" / "research" / "memory"
REPO_ROOT = Path(__file__).parents[1]
PACK = json.loads((MEMORY_DOCS / "mem3b0q_r4_control_pack_v1.json").read_text("utf-8"))
SCHEMA = json.loads((MEMORY_DOCS / "mem3b0q_r4_response.schema.json").read_text("utf-8"))


def _gold_content(proposition: dict) -> bytes:
    expected = proposition["expected"]
    payload = {
        "source_id": proposition["source_id"],
        "atoms": [
            {
                "owner_span": atom["owner_span"],
                "object_span": atom["object_span"],
                "attribute_span": atom["attribute_span"],
                "value_span": atom["value_span"],
                "cardinality_proposal": atom["cardinality"],
            }
            for atom in expected["atoms"]
        ],
        "abstention_reason": expected["abstention"] or "NONE",
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _gold_observations() -> dict[str, dict]:
    return {
        proposition["source_id"]: mem3b0q_r4.validate_content(
            _gold_content(proposition), proposition, scope_id=PACK["scope_id"]
        )
        for proposition in PACK["propositions"]
    }


def test_all_authored_gold_rows_satisfy_response_contract_and_oracle() -> None:
    observed = _gold_observations()
    for proposition in PACK["propositions"]:
        assert mem3b0q_r4.compare_to_oracle(
            observed[proposition["source_id"]], proposition["expected"]
        ) == []

    result = mem3b0q_r4.evaluate_pack(PACK["propositions"], observed, PACK["relations"])
    assert result["passed"] is True
    assert len(result["proposition_results"]) == 20
    assert len(result["relation_results"]) == 9
    assert result["revision_edges"] == 0
    assert result["memory_store_mutations"] == 0


def test_request_projection_never_serializes_oracle() -> None:
    source = dict(PACK["propositions"][0])
    source["expected"] = {"private_oracle_sentinel": "NEVER_SEND_THIS"}
    request = mem3b0q_r4.build_request_payload(source, SCHEMA, model="Qwen3-8B")
    user_payload = json.loads(request["messages"][1]["content"])
    assert user_payload == {
        "source_id": source["source_id"],
        "proposition_text": source["proposition_text"],
    }
    assert "NEVER_SEND_THIS" not in json.dumps(request)
    assert request["response_format"]["json_schema"]["schema"] == SCHEMA
    assert request["temperature"] == 0
    assert request["seed"] == 42
    assert request["max_tokens"] == 256
    assert request["chat_template_kwargs"] == {"enable_thinking": False}


def test_request_schema_hash_is_stable() -> None:
    assert mem3b0q_r4.canonical_json_sha256(SCHEMA) == (
        "10f976e77594c8e7a31f81ee73d93e964963e2e9e6bdf571ed697d726b246cbc"
    )


def test_overlap_audit_recomputes_against_frozen_source_sets() -> None:
    audit = json.loads((MEMORY_DOCS / "mem3b0q_r4_overlap_audit.json").read_text("utf-8"))
    r3_path = (
        REPO_ROOT
        / "runs/memory/mem3/mem3b0q-span-identity-pilot-r3-20261001/proposal_inputs.jsonl"
    )
    b0q_dir = REPO_ROOT / "runs/memory/mem3/mem3b0q-factorized-admission-20260930"
    eligible_path = b0q_dir / "eligible_records.jsonl"
    r3_rows = [json.loads(line) for line in r3_path.read_text("utf-8").splitlines() if line]
    eligible_rows = [
        json.loads(line) for line in eligible_path.read_text("utf-8").splitlines() if line
    ]
    review_files = list(audit["b0q_review_and_control_ids"]["sources"])
    review_payloads = [json.loads((b0q_dir / name).read_text("utf-8")) for name in review_files]

    def normalized_hash(text: str) -> str:
        normalized = " ".join(unicodedata.normalize("NFC", text).casefold().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def memory_ids(value: object) -> set[str]:
        if isinstance(value, dict):
            return set().union(*(memory_ids(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(memory_ids(item) for item in value))
        if isinstance(value, str):
            return set(re.findall(r"m10flat3-[0-9a-f]{64}", value))
        return set()

    fixture_ids = {row["source_id"] for row in PACK["propositions"]}
    fixture_hashes = {
        normalized_hash(row["proposition_text"]) for row in PACK["propositions"]
    }
    r3_ids = {row["memory_id"] for row in r3_rows}
    eligible_ids = {row["memory_id"] for row in eligible_rows}
    reviewed_ids = set().union(*(memory_ids(item) for item in review_payloads))
    r3_hashes = {normalized_hash(row["proposition_text"]) for row in r3_rows}
    eligible_hashes = {normalized_hash(row["proposition_text"]) for row in eligible_rows}

    assert audit["status"] == "PASS"
    assert len(PACK["propositions"]) == 20
    assert len(fixture_hashes) == 20
    assert len(r3_rows) == 19
    assert len(eligible_rows) == len(eligible_ids) == 1366
    assert len(reviewed_ids) == 219
    assert reviewed_ids <= eligible_ids
    assert not fixture_ids.intersection(r3_ids | eligible_ids | reviewed_ids)
    assert not fixture_hashes.intersection(r3_hashes | eligible_hashes)
    assert audit["r3"]["normalized_proposition_hash_intersection"] == 0
    assert audit["b0q_eligible"]["normalized_proposition_hash_intersection"] == 0
    assert audit["b0q_review_and_control_ids"]["all_ids_are_in_b0q_eligible_records"] is True
    fixture_path = MEMORY_DOCS / "mem3b0q_r4_control_pack_v1.json"
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == audit["fixture_sha256"]
    assert hashlib.sha256(r3_path.read_bytes()).hexdigest() == audit["r3"]["source_sha256"]
    assert hashlib.sha256(eligible_path.read_bytes()).hexdigest() == audit["b0q_eligible"][
        "source_sha256"
    ]
    for source_name, source_hash in audit["b0q_review_and_control_ids"]["sources"].items():
        assert hashlib.sha256((b0q_dir / source_name).read_bytes()).hexdigest() == source_hash


def test_runtime_prompt_matches_reviewed_protocol_prompt() -> None:
    protocol = (MEMORY_DOCS / "mem_3b0q_r4_protocol_v1.md").read_text("utf-8")
    match = re.search(
        (
            r"## Frozen prompt and inference lock\s+Frozen system prompt, kept "
            r"byte-for-byte in the request builder:\s+> (.*?)\s+Frozen local runtime"
        ),
        protocol,
        flags=re.DOTALL,
    )
    assert match is not None
    assert match.group(1) == mem3b0q_r4.SYSTEM_PROMPT


def test_duplicate_json_keys_are_rejected() -> None:
    raw = (
        b'{"source_id":"R4-P01","source_id":"R4-P02",'
        b'"atoms":[],"abstention_reason":"MISSING_OWNER"}'
    )
    with pytest.raises(mem3b0q_r4.R4ContractError, match="duplicate_json_key"):
        mem3b0q_r4.validate_content(raw, PACK["propositions"][0], scope_id=PACK["scope_id"])


def test_abstention_cannot_carry_atoms_or_none_reason() -> None:
    proposition = PACK["propositions"][0]
    payload = json.loads(_gold_content(proposition))
    payload["abstention_reason"] = "MISSING_OWNER"
    with pytest.raises(mem3b0q_r4.R4ContractError, match="atoms_abstention_inconsistent"):
        mem3b0q_r4.validate_content(
            json.dumps(payload).encode(), proposition, scope_id=PACK["scope_id"]
        )


def test_unregistered_alias_is_not_normalized_or_expanded() -> None:
    proposition = PACK["propositions"][0]
    payload = json.loads(_gold_content(proposition))
    payload["atoms"][0]["object_span"] = "plan"
    with pytest.raises(mem3b0q_r4.R4ContractError, match="unsupported_exact_alias"):
        mem3b0q_r4.validate_content(
            json.dumps(payload).encode(), proposition, scope_id=PACK["scope_id"]
        )


def test_repeated_witness_is_rejected_even_if_alias_is_known() -> None:
    proposition = {
        "source_id": "R4-TEST-DUP",
        "proposition_text": "My wallet color is black; my wallet is old.",
    }
    payload = {
        "source_id": proposition["source_id"],
        "atoms": [
            {
                "owner_span": "My",
                "object_span": "wallet",
                "attribute_span": "color",
                "value_span": "black",
                "cardinality_proposal": "SINGLE_VALUE_AT_A_TIME",
            }
        ],
        "abstention_reason": "NONE",
    }
    with pytest.raises(mem3b0q_r4.R4ContractError, match="object_span:not_unique_source_substring"):
        mem3b0q_r4.validate_content(
            json.dumps(payload).encode(), proposition, scope_id=PACK["scope_id"]
        )


def test_span_offsets_are_unicode_codepoint_offsets() -> None:
    proposition = {
        "source_id": "R4-TEST-UNICODE",
        "proposition_text": "My wallet color is café.",
    }
    payload = {
        "source_id": proposition["source_id"],
        "atoms": [
            {
                "owner_span": "My",
                "object_span": "wallet",
                "attribute_span": "color",
                "value_span": "café",
                "cardinality_proposal": "SINGLE_VALUE_AT_A_TIME",
            }
        ],
        "abstention_reason": "NONE",
    }
    row = mem3b0q_r4.validate_content(
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        proposition,
        scope_id=PACK["scope_id"],
    )
    witness = row["atoms"][0]["witnesses"]["value_span"]
    assert witness == {"text": "café", "start": 19, "end": 23}
    assert proposition["proposition_text"][witness["start"] : witness["end"]] == "café"


def test_event_proposal_never_becomes_slot_candidate() -> None:
    proposition = PACK["propositions"][13]
    row = mem3b0q_r4.validate_content(
        _gold_content(proposition), proposition, scope_id=PACK["scope_id"]
    )
    assert row["atoms"][0]["slot_candidate"] is False
    assert row["atoms"][0]["slot_key"] is None


@pytest.mark.parametrize("mutation", ["empty", "partial", "duplicate", "altered_label"])
def test_pack_evaluator_requires_exact_frozen_relation_set(mutation: str) -> None:
    relations = [dict(row) for row in PACK["relations"]]
    if mutation == "empty":
        relations = []
    elif mutation == "partial":
        relations.pop()
    elif mutation == "duplicate":
        relations.append(dict(relations[0]))
    else:
        relations[0]["expected"] = "DISTINCT_SLOT_OWNER"
    with pytest.raises(
        mem3b0q_r4.R4ContractError,
        match="frozen_relation_coverage_mismatch",
    ):
        mem3b0q_r4.evaluate_pack(PACK["propositions"], _gold_observations(), relations)


@pytest.mark.parametrize("bad_index", [-1, True, 99])
def test_invalid_relation_indices_are_rejected(bad_index: int | bool) -> None:
    relation = {
        "left": ["R4-P01", bad_index],
        "right": ["R4-P02", 0],
        "expected": "SAME_SLOT_SAME_VALUE",
    }
    with pytest.raises(mem3b0q_r4.R4ContractError, match="relation_reference_invalid"):
        mem3b0q_r4.evaluate_relations([relation], _gold_observations())


def test_pack_evaluator_rebinds_observation_source_and_scope() -> None:
    observations = _gold_observations()
    observations["R4-P01"]["source_id"] = "R4-P02"
    with pytest.raises(mem3b0q_r4.R4ContractError, match="observation_source_id_mismatch"):
        mem3b0q_r4.evaluate_pack(
            PACK["propositions"], observations, PACK["relations"]
        )

    observations = _gold_observations()
    observations["R4-P01"]["scope_id"] = "other-scope"
    with pytest.raises(mem3b0q_r4.R4ContractError, match="observation_scope_mismatch"):
        mem3b0q_r4.evaluate_pack(
            PACK["propositions"], observations, PACK["relations"]
        )

    observations = _gold_observations()
    observations["R4-P01"]["atoms"][0]["scope_id"] = "other-scope"
    with pytest.raises(mem3b0q_r4.R4ContractError, match="observation_atom_scope_mismatch"):
        mem3b0q_r4.evaluate_pack(
            PACK["propositions"], observations, PACK["relations"]
        )


def test_i_is_not_a_registered_owner_alias() -> None:
    proposition = {
        "source_id": "R4-TEST-ACTOR",
        "proposition_text": "I own one wallet with color black.",
    }
    payload = {
        "source_id": proposition["source_id"],
        "atoms": [
            {
                "owner_span": "I",
                "object_span": "wallet",
                "attribute_span": "color",
                "value_span": "black",
                "cardinality_proposal": "SINGLE_VALUE_AT_A_TIME",
            }
        ],
        "abstention_reason": "NONE",
    }
    with pytest.raises(
        mem3b0q_r4.R4ContractError,
        match="owner_span:unsupported_exact_alias",
    ):
        mem3b0q_r4.validate_content(
            json.dumps(payload).encode(),
            proposition,
            scope_id=PACK["scope_id"],
        )


def test_overlapping_substring_occurrences_are_not_unique_witnesses() -> None:
    proposition = {
        "source_id": "R4-TEST-OVERLAP",
        "proposition_text": "My wallet color is banana.",
    }
    payload = {
        "source_id": proposition["source_id"],
        "atoms": [
            {
                "owner_span": "My",
                "object_span": "wallet",
                "attribute_span": "color",
                "value_span": "ana",
                "cardinality_proposal": "SINGLE_VALUE_AT_A_TIME",
            }
        ],
        "abstention_reason": "NONE",
    }
    with pytest.raises(
        mem3b0q_r4.R4ContractError,
        match="value_span:not_unique_source_substring",
    ):
        mem3b0q_r4.validate_content(
            json.dumps(payload).encode(),
            proposition,
            scope_id=PACK["scope_id"],
        )


def test_oracle_comparison_detects_owner_misbinding() -> None:
    proposition = PACK["propositions"][3]
    payload = json.loads(_gold_content(proposition))
    payload["atoms"][0]["owner_span"] = "My"
    observed = mem3b0q_r4.validate_content(
        json.dumps(payload).encode(), proposition, scope_id=PACK["scope_id"]
    )
    assert mem3b0q_r4.compare_to_oracle(observed, proposition["expected"]) == [
        "atom_0:oracle_mismatch"
    ]
