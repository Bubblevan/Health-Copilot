from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import mem3b0q_r4_candidate_binding_guard_v1 as locality_guard
from tools.research.memory import mem3b0q_r4_candidate_builder_v1 as candidate_builder
from tools.research.memory import mem3b0q_r4_candidate_request_v1 as request_builder
from tools.research.memory import mem3b0q_r4_joint_binding_guard_v2 as joint_guard_v2
from tools.research.memory import mem3b0q_r4_joint_binding_guard_v3 as joint_guard_v3
from tools.research.memory.mem3b0q_r4_atomwise_admission_v1 import (
    AtomwiseAdmissionError,
    admit_atoms_independently,
)
from tools.research.memory import qualify_mem3b0q_r4_atomwise_admission_v1 as qualification
from tools.research.memory import run_mem3b0q_r4_candidate_local_smoke_v1 as smoke


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PACK = ROOT / "docs/research/memory/mem3b0q_r4_candidate_control_pack_v1.json"
FROZEN_PACK = ROOT / "docs/research/memory/mem3b0q_r4_control_pack_v1.json"
SMOKE_RESPONSE = (
    ROOT
    / "runs/memory/mem3/mem3b0q-r4-candidate-local-smoke-v1/response_body.bin"
)


def _admitted_atoms(admission: dict) -> list[dict]:
    return [row["atom"] for row in admission["admitted"]]


def _schema_validate(case: dict, content: bytes) -> None:
    request = smoke.request_builder.build_candidate_request(
        case, model=smoke.frozen_gate.MODEL_PATH
    )
    proposal = json.loads(content)
    Draft202012Validator(
        request["response_format"]["json_schema"]["schema"]
    ).validate(proposal)


def test_recorded_r4c05_model_response_keeps_linux_and_quarantines_windows() -> None:
    pack = json.loads(CANDIDATE_PACK.read_text(encoding="utf-8"))
    case = next(row for row in pack["cases"] if row["case_id"] == "R4C-05")
    content = smoke._response_content(SMOKE_RESPONSE.read_bytes())[1].encode("utf-8")
    _schema_validate(case, content)
    proposition = {
        "source_id": case["case_id"],
        "proposition_text": case["proposition_text"],
    }

    with pytest.raises(joint_guard_v2.JointBindingError, match="value_crosses_typed_anchor"):
        joint_guard_v3.validate_joint_bound_candidate_proposal(
            content, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )

    admission = admit_atoms_independently(
        content, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
    )
    exact = request_builder.exact_expected_atom_multiset_match(
        qualification._candidate_case(case),
        {
            "source_id": admission["source_id"],
            "scope_id": admission["scope_id"],
            "atoms": _admitted_atoms(admission),
        },
    )

    assert admission["status"] == "PARTIAL"
    assert len(admission["admitted"]) == 1
    assert admission["admitted"][0]["atom"]["value_text"] == "Linux"
    assert len(admission["quarantined"]) == 1
    assert admission["quarantined"][0]["source_atom_index"] == 1
    assert "value_crosses_typed_anchor" in admission["quarantined"][0]["reason"]
    assert exact


@pytest.mark.parametrize("case_id", ["R4C-05", "R4C-06"])
def test_known_cross_binding_negative_controls_preserve_valid_sibling(case_id: str) -> None:
    pack = json.loads(CANDIDATE_PACK.read_text(encoding="utf-8"))
    case = next(row for row in pack["cases"] if row["case_id"] == case_id)
    good_atoms = [
        qualification._raw_atom(case_id, case["proposition_text"], expected)
        for expected in case["expected_atoms"]
    ]
    wrong_atom = qualification._injected_invalid_atom(
        case, case["invalid_cross_bindings"][0]
    )
    content = qualification._content(case_id, good_atoms + [wrong_atom], "NONE")
    proposition = {
        "source_id": case_id,
        "proposition_text": case["proposition_text"],
    }
    _schema_validate(case, content)

    with pytest.raises(joint_guard_v2.JointBindingError):
        joint_guard_v3.validate_joint_bound_candidate_proposal(
            content, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )

    admission = admit_atoms_independently(
        content, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
    )
    exact = request_builder.exact_expected_atom_multiset_match(
        qualification._candidate_case(case),
        {
            "source_id": admission["source_id"],
            "scope_id": admission["scope_id"],
            "atoms": _admitted_atoms(admission),
        },
    )
    assert admission["status"] == "PARTIAL"
    assert len(admission["admitted"]) == len(case["expected_atoms"])
    assert len(admission["quarantined"]) == 1
    assert exact


def test_all_frozen_r4_oracle_atoms_and_abstentions_survive() -> None:
    pack = json.loads(FROZEN_PACK.read_text(encoding="utf-8"))
    for proposition in pack["propositions"]:
        content = qualification._oracle_frozen_proposal(proposition)
        admission = admit_atoms_independently(
            content,
            {
                "source_id": proposition["source_id"],
                "proposition_text": proposition["proposition_text"],
            },
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )
        expected = proposition["expected"]
        if expected["atoms"]:
            assert admission["status"] == "ACCEPTED"
            assert not admission["quarantined"]
            assert request_builder.exact_expected_atom_multiset_match(
                {
                    "case_id": proposition["source_id"],
                    "proposition_text": proposition["proposition_text"],
                    "expected_atoms": expected["atoms"],
                },
                {
                    "source_id": admission["source_id"],
                    "scope_id": admission["scope_id"],
                    "atoms": _admitted_atoms(admission),
                },
            )
        else:
            assert admission["status"] == "ABSTAINED"
            assert admission["source_abstention_reason"] == expected["abstention"]


def test_exact_duplicate_atoms_are_deduplicated() -> None:
    proposition = {
        "source_id": "R4-P05",
        "proposition_text": "My work laptop's operating system is Linux.",
    }
    expected = {
        "owner_id": "SELF",
        "owner_span": "My",
        "object_id": "WORK_LAPTOP",
        "object_span": "work laptop",
        "attribute_id": "OPERATING_SYSTEM",
        "attribute_span": "operating system",
        "value_span": "Linux",
        "cardinality": "SINGLE_VALUE_AT_A_TIME",
    }
    atom = qualification._raw_atom("R4-P05", proposition["proposition_text"], expected)
    content = qualification._content("R4-P05", [atom, atom], "NONE")

    admission = admit_atoms_independently(
        content, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
    )

    assert admission["status"] == "PARTIAL"
    assert len(admission["admitted"]) == 1
    assert admission["quarantined"] == [
        {
            "source_atom_index": 1,
            "reason": "duplicate_admitted_atom",
            "duplicate_of_source_atom_index": 0,
        }
    ]


def test_invalid_atom_does_not_suppress_multiple_valid_siblings() -> None:
    pack = json.loads(CANDIDATE_PACK.read_text(encoding="utf-8"))
    case = next(row for row in pack["cases"] if row["case_id"] == "R4C-01")
    good_atoms = [
        qualification._raw_atom("R4C-01", case["proposition_text"], expected)
        for expected in case["expected_atoms"]
    ]
    negative = {
        "object_id": "EXERCISE_PLAN",
        "attribute_id": "ACTIVITY",
        "attribute_occurrence": 0,
        "wrong_value_span": "Fedora",
    }
    wrong_atom = qualification._injected_invalid_atom(case, negative)
    content = qualification._content("R4C-01", good_atoms + [wrong_atom], "NONE")
    proposition = {
        "source_id": "R4C-01",
        "proposition_text": case["proposition_text"],
    }

    with pytest.raises(locality_guard.CandidateBindingError):
        joint_guard_v3.validate_joint_bound_candidate_proposal(
            content, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )

    admission = admit_atoms_independently(
        content, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
    )

    assert admission["status"] == "PARTIAL"
    assert len(admission["admitted"]) == 2
    assert {row["atom"]["value_text"] for row in admission["admitted"]} == {
        "rowing",
        "Fedora",
    }
    assert admission["quarantined"][0]["source_atom_index"] == 2
    assert request_builder.exact_expected_atom_multiset_match(
        qualification._candidate_case(case),
        {
            "source_id": admission["source_id"],
            "scope_id": admission["scope_id"],
            "atoms": _admitted_atoms(admission),
        },
    )


def test_invalid_candidate_in_one_atom_does_not_drop_valid_sibling() -> None:
    proposition = {
        "source_id": "R4-P05",
        "proposition_text": "My work laptop's operating system is Linux.",
    }
    expected = {
        "owner_id": "SELF",
        "owner_span": "My",
        "object_id": "WORK_LAPTOP",
        "object_span": "work laptop",
        "attribute_id": "OPERATING_SYSTEM",
        "attribute_span": "operating system",
        "value_span": "Linux",
        "cardinality": "SINGLE_VALUE_AT_A_TIME",
    }
    good = qualification._raw_atom("R4-P05", proposition["proposition_text"], expected)
    bad = {**good, "object_candidate_id": "object:TABLET:0:6"}
    content = qualification._content("R4-P05", [good, bad], "NONE")

    admission = admit_atoms_independently(
        content, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
    )

    assert admission["status"] == "PARTIAL"
    assert [row["atom"]["value_text"] for row in admission["admitted"]] == ["Linux"]
    assert admission["quarantined"][0]["source_atom_index"] == 1
    assert "CandidateBuildError" in admission["quarantined"][0]["reason"]


@pytest.mark.parametrize(
    "content",
    [
        b'{"source_id":"x","source_id":"x","atoms":[],"abstention_reason":"UNSUPPORTED_ANCHOR"}',
        b"not-json",
        b'{"source_id":"R4-P05","atoms":[],"abstention_reason":"NONE"}',
    ],
)
def test_malformed_envelope_fails_closed(content: bytes) -> None:
    proposition = {
        "source_id": "R4-P05",
        "proposition_text": "My work laptop's operating system is Linux.",
    }
    with pytest.raises(AtomwiseAdmissionError):
        admit_atoms_independently(
            content, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )


def test_qualification_is_offline_and_reproduces_document_level_loss() -> None:
    result = qualification.qualify()

    assert result["status"] == "PASS"
    assert result["new_model_calls"] == 0
    assert result["hosted_calls"] == 0
    assert result["actual_model_response_replay"]["strict_document_accepts"] is False
    assert result["actual_model_response_replay"]["exact_expected_atom_multiset_match"]
    assert result["mixed_valid_invalid_proposal_summary"]["valid_atom_retention"] == 1.0
    assert result["mixed_valid_invalid_proposal_summary"]["strict_document_level_retention"] == 0
