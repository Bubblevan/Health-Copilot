from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory.mem3b0q_r4_candidate_builder_v1 import build_candidates
from tools.research.memory.mem3b0q_r4_candidate_request_v1 import (
    SYSTEM_PROMPT,
    build_candidate_request,
    exact_expected_atom_multiset_match,
)
from tools.research.memory.mem3b0q_r4_joint_binding_guard_v3 import (
    validate_joint_bound_candidate_proposal,
)


ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "docs/research/memory/mem3b0q_r4_candidate_control_pack_v1.json"
MODEL_ID = r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf"
DEV_MANIFEST_PATH = ROOT / "docs/research/memory/mem3b0q_r4_development_control_manifest_v1.json"
EXPECTED_DEV_CONTROL_IDS = {
    "V2-X1", "V2-X2", "V2-X3", "V2-X4-A", "V2-X4-B", "V2-X5", "V2-X6", "V2-X7", "V2-X8",
    "V3-X9-A", "V3-X9-B", "V3-X10", "V3-X11", "BIND-D1",
    "BUILDER-01", "BUILDER-02", "BUILDER-03", "BUILDER-04", "BUILDER-05",
    "BUILDER-06", "BUILDER-07", "BUILDER-08", "BUILDER-09", "BUILDER-10",
}


def _pack() -> dict:
    return json.loads(PACK_PATH.read_text(encoding="utf-8"))


def _candidate_ids(source_id: str, source: str, expected: dict) -> dict[str, str]:
    candidates = build_candidates(source, frozen_r4.ALIASES, source_id=source_id)[
        "candidates"
    ]
    result = {}
    for field in ("owner", "object", "attribute"):
        matches = [
            row
            for row in candidates[field]
            if row["canonical_id"] == expected[f"{field}_id"]
            and row["source_span"].casefold()
            == expected[f"{field}_span"].casefold()
        ]
        match = matches[expected.get(f"{field}_occurrence", 0)]
        result[f"{field}_candidate_id"] = match["candidate_id"]
    return result


def _proposal(case: dict, expected_atoms: list[dict]) -> str:
    atoms = []
    for expected in expected_atoms:
        atom = _candidate_ids(
            case["case_id"], case["proposition_text"], expected
        )
        atom["value_span"] = expected["value_span"]
        atom["cardinality_proposal"] = expected["cardinality"]
        atoms.append(atom)
    return json.dumps(
        {
            "source_id": case["case_id"],
            "atoms": atoms,
            "abstention_reason": "NONE" if atoms else "UNSUPPORTED_ANCHOR",
        }
    )


def test_request_uses_candidate_ids_and_omits_expected_labels() -> None:
    case = _pack()["cases"][0]
    request = build_candidate_request(case, model=MODEL_ID)
    user_payload = json.loads(request["messages"][1]["content"])

    assert user_payload["source_id"] == case["case_id"]
    assert user_payload["proposition_text"] == case["proposition_text"]
    assert set(user_payload) == {"source_id", "proposition_text", "typed_candidates"}
    assert "expected_atoms" not in json.dumps(user_payload)
    assert request["temperature"] == 0
    assert request["seed"] == 42
    assert request["max_tokens"] == 256
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "untrusted data" in SYSTEM_PROMPT


def test_candidate_schema_bounds_ids_per_source() -> None:
    case = _pack()["cases"][4]
    request = build_candidate_request(case, model=MODEL_ID)
    user_payload = json.loads(request["messages"][1]["content"])
    properties = request["response_format"]["json_schema"]["schema"]["properties"]
    atom_properties = properties["atoms"]["items"]["properties"]

    for field in ("owner", "object", "attribute"):
        enum = atom_properties[f"{field}_candidate_id"]["enum"]
        supplied = [row["candidate_id"] for row in user_payload["typed_candidates"][field]]
        assert enum == supplied
    assert properties["atoms"]["maxItems"] == 4


def test_all_dynamic_candidate_schemas_pass_draft_2020_12() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator
    for case in _pack()["cases"]:
        request = build_candidate_request(case, model=MODEL_ID)
        schema = request["response_format"]["json_schema"]["schema"]
        validator.check_schema(schema)


def test_control_pack_is_disjoint_and_covers_every_typed_slot() -> None:
    pack = _pack()
    old_pack = json.loads(
        (ROOT / "docs/research/memory/mem3b0q_r4_control_pack_v1.json").read_text(
            encoding="utf-8"
        )
    )
    dev_manifest = json.loads(DEV_MANIFEST_PATH.read_text(encoding="utf-8"))
    normalize = lambda text: " ".join(text.split()).casefold()
    old_sources = {
        normalize(row["proposition_text"]) for row in old_pack["propositions"]
    }
    dev_controls = dev_manifest["controls"]
    dev_sources = {normalize(row["text"]) for row in dev_controls}
    new_sources = {normalize(case["proposition_text"]) for case in pack["cases"]}

    assert pack["status"] == "FROZEN_PROTOCOL_NO_INFERENCE_AUTHORIZATION"
    assert dev_manifest["status"] == "MAINTAINED_OFFLINE_INVENTORY"
    assert {row["control_id"] for row in dev_controls} == EXPECTED_DEV_CONTROL_IDS
    assert len({row["control_id"] for row in dev_controls}) == len(dev_controls)
    assert not old_sources.intersection(new_sources)
    assert not dev_sources.intersection(new_sources)
    assert len({case["case_id"] for case in pack["cases"]}) == len(pack["cases"])

    expected_pairs = Counter(
        (atom["object_id"], atom["attribute_id"])
        for case in pack["cases"]
        for atom in case["expected_atoms"]
    )
    required_pairs = {tuple(pair) for pair in pack["required_slot_pairs"]}
    assert required_pairs.issubset(expected_pairs)

    negative_pairs = {
        (row["object_id"], row["attribute_id"])
        for case in pack["cases"]
        for row in case.get("negative_value_bindings", [])
    }
    assert required_pairs.issubset(negative_pairs)
    assert sum(len(case.get("negative_value_bindings", [])) for case in pack["cases"]) == 9


def test_all_expected_atoms_pass_v3_and_all_value_swaps_fail_closed() -> None:
    pack = _pack()
    for case in pack["cases"]:
        proposition = {
            "source_id": case["case_id"],
            "proposition_text": case["proposition_text"],
        }
        valid = validate_joint_bound_candidate_proposal(
            _proposal(case, case["expected_atoms"]),
            proposition,
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )
        assert exact_expected_atom_multiset_match(case, valid)

        for negative in case.get("negative_value_bindings", []):
            expected_atom = next(
                atom
                for atom in case["expected_atoms"]
                if atom["object_id"] == negative["object_id"]
                and atom["attribute_id"] == negative["attribute_id"]
            )
            invalid = dict(expected_atom)
            invalid["value_span"] = negative["wrong_value_span"]
            with pytest.raises(ValueError):
                validate_joint_bound_candidate_proposal(
                    _proposal(case, [invalid]),
                    proposition,
                    scope_id=frozen_r4.FROZEN_SCOPE_ID,
                )


def test_exact_atom_gate_rejects_duplicate_extra_and_witness_drift() -> None:
    case = _pack()["cases"][4]
    proposition = {
        "source_id": case["case_id"],
        "proposition_text": case["proposition_text"],
    }
    valid = validate_joint_bound_candidate_proposal(
        _proposal(case, case["expected_atoms"]),
        proposition,
        scope_id=frozen_r4.FROZEN_SCOPE_ID,
    )
    assert exact_expected_atom_multiset_match(case, valid)

    duplicate = {**valid, "atoms": [*valid["atoms"], valid["atoms"][0]]}
    assert not exact_expected_atom_multiset_match(case, duplicate)

    extra = {
        **valid,
        "atoms": [
            *valid["atoms"],
            {**valid["atoms"][0], "value_text": "Windows"},
        ],
    }
    assert not exact_expected_atom_multiset_match(case, extra)

    drifted = json.loads(json.dumps(valid))
    drifted["atoms"][0]["witnesses"]["value_span"]["start"] += 1
    drifted["atoms"][0]["witnesses"]["value_span"]["end"] += 1
    assert not exact_expected_atom_multiset_match(case, drifted)


def test_unknown_object_cross_bindings_are_rejected() -> None:
    pack = _pack()
    for case in pack["cases"]:
        proposition = {
            "source_id": case["case_id"],
            "proposition_text": case["proposition_text"],
        }
        for invalid_binding in case.get("invalid_cross_bindings", []):
            target = next(
                atom
                for atom in case["expected_atoms"]
                if atom["object_id"] == invalid_binding["object_id"]
                and atom["attribute_id"] == invalid_binding["attribute_id"]
            )
            invalid = dict(target)
            invalid["attribute_span"] = invalid_binding["attribute_span"]
            invalid["attribute_occurrence"] = invalid_binding[
                "attribute_occurrence"
            ]
            invalid["value_span"] = invalid_binding["wrong_value_span"]
            with pytest.raises(ValueError):
                validate_joint_bound_candidate_proposal(
                    _proposal(case, [invalid]),
                    proposition,
                    scope_id=frozen_r4.FROZEN_SCOPE_ID,
                )
