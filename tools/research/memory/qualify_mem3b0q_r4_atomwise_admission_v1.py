"""Reproduce offline atomwise-admission qualification counts for R4 controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import mem3b0q_r4_candidate_binding_guard_v1 as locality_guard
from tools.research.memory import mem3b0q_r4_candidate_builder_v1 as candidate_builder
from tools.research.memory import mem3b0q_r4_candidate_request_v1 as request_builder
from tools.research.memory import mem3b0q_r4_joint_binding_guard_v2 as joint_guard_v2
from tools.research.memory import mem3b0q_r4_joint_binding_guard_v3 as joint_guard_v3
from tools.research.memory.mem3b0q_r4_atomwise_admission_v1 import (
    admit_atoms_independently,
)
from tools.research.memory import run_mem3b0q_r4_candidate_local_smoke_v1 as smoke


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs" / "research" / "memory"
CANDIDATE_PACK_PATH = DOCS / "mem3b0q_r4_candidate_control_pack_v1.json"
FROZEN_PACK_PATH = DOCS / "mem3b0q_r4_control_pack_v1.json"
SMOKE_DIR = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-candidate-local-smoke-v1"
DEFAULT_OUTPUT = (
    ROOT
    / "runs"
    / "memory"
    / "mem3"
    / "mem3b0q-r4-atomwise-admission-qualification-v1"
    / "result.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_not_object:{path.name}")
    return value


def _candidate(
    candidates: dict[str, list[dict[str, Any]]],
    field: str,
    canonical_id: str,
    span: str | None = None,
    occurrence: int = 0,
) -> dict[str, Any]:
    rows = [
        row
        for row in candidates[field]
        if row["canonical_id"] == canonical_id
        and (span is None or row["source_span"].casefold() == span.casefold())
    ]
    if occurrence < 0 or occurrence >= len(rows):
        raise ValueError(f"candidate_missing:{field}:{canonical_id}:{occurrence}")
    return rows[occurrence]


def _raw_atom(
    source_id: str,
    source: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    candidates = candidate_builder.build_candidates(
        source, frozen_r4.ALIASES, source_id=source_id
    )["candidates"]
    atom = {}
    for field in ("owner", "object", "attribute"):
        row = _candidate(
            candidates,
            field,
            expected[f"{field}_id"],
            expected[f"{field}_span"],
            expected.get(f"{field}_occurrence", 0),
        )
        atom[f"{field}_candidate_id"] = row["candidate_id"]
    atom["value_span"] = expected["value_span"]
    atom["cardinality_proposal"] = expected["cardinality"]
    return atom


def _content(source_id: str, atoms: list[dict[str, Any]], reason: str) -> bytes:
    return json.dumps(
        {"source_id": source_id, "atoms": atoms, "abstention_reason": reason},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _candidate_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "proposition_text": case["proposition_text"],
        "expected_atoms": case["expected_atoms"],
    }


def _normalized_for_eval(admission: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": admission["source_id"],
        "scope_id": admission["scope_id"],
        "atoms": [row["atom"] for row in admission["admitted"]],
    }


def _strict_document_accepts(content: bytes, proposition: dict[str, str]) -> bool:
    try:
        joint_guard_v3.validate_joint_bound_candidate_proposal(
            content, proposition, scope_id=frozen_r4.FROZEN_SCOPE_ID
        )
    except (
        candidate_builder.CandidateBuildError,
        locality_guard.CandidateBindingError,
        joint_guard_v2.JointBindingError,
        joint_guard_v3.JointBindingError,
    ):
        return False
    return True


def _injected_invalid_atom(case: dict[str, Any], negative: dict[str, Any]) -> dict[str, Any]:
    source_id = case["case_id"]
    source = case["proposition_text"]
    candidates = candidate_builder.build_candidates(
        source, frozen_r4.ALIASES, source_id=source_id
    )["candidates"]
    expected = next(
        atom
        for atom in case["expected_atoms"]
        if atom["object_id"] == negative["object_id"]
        and atom["attribute_id"] == negative["attribute_id"]
    )
    owner = _candidate(
        candidates,
        "owner",
        expected["owner_id"],
        expected["owner_span"],
        expected.get("owner_occurrence", 0),
    )
    obj = _candidate(candidates, "object", negative["object_id"])
    attribute = _candidate(
        candidates,
        "attribute",
        negative["attribute_id"],
        occurrence=negative["attribute_occurrence"],
    )
    return {
        "owner_candidate_id": owner["candidate_id"],
        "object_candidate_id": obj["candidate_id"],
        "attribute_candidate_id": attribute["candidate_id"],
        "value_span": negative["wrong_value_span"],
        "cardinality_proposal": expected["cardinality"],
    }


def _oracle_frozen_proposal(proposition: dict[str, Any]) -> bytes:
    atoms = [
        _raw_atom(
            proposition["source_id"], proposition["proposition_text"], expected
        )
        for expected in proposition["expected"]["atoms"]
    ]
    abstention = proposition["expected"].get("abstention") or "NONE"
    return _content(proposition["source_id"], atoms, abstention)


def qualify() -> dict[str, Any]:
    candidate_pack = _read_json(CANDIDATE_PACK_PATH)
    frozen_pack = _read_json(FROZEN_PACK_PATH)
    smoke_result = _read_json(SMOKE_DIR / "result.json")
    response_bytes = (SMOKE_DIR / "response_body.bin").read_bytes()
    if smoke_result.get("status") != "QUALITY_FAILURE":
        raise ValueError("expected_recorded_r4c05_quality_failure")
    envelope, model_content = smoke._response_content(response_bytes)
    del envelope

    candidate_cases = {row["case_id"]: row for row in candidate_pack["cases"]}
    actual_case = candidate_cases["R4C-05"]
    actual_proposition = {
        "source_id": actual_case["case_id"],
        "proposition_text": actual_case["proposition_text"],
    }
    actual_request = smoke.request_builder.build_candidate_request(
        actual_case, model=smoke.frozen_gate.MODEL_PATH
    )
    proposal_object = json.loads(model_content)
    Draft202012Validator(
        actual_request["response_format"]["json_schema"]["schema"]
    ).validate(proposal_object)
    actual_content = model_content.encode("utf-8")
    actual_strict_accepts = _strict_document_accepts(
        actual_content, actual_proposition
    )
    actual_admission = admit_atoms_independently(
        actual_content,
        actual_proposition,
        scope_id=frozen_r4.FROZEN_SCOPE_ID,
    )
    actual_exact = request_builder.exact_expected_atom_multiset_match(
        _candidate_case(actual_case),
        _normalized_for_eval(actual_admission),
    )

    injected_results = []
    for case_id in ("R4C-05", "R4C-06"):
        case = candidate_cases[case_id]
        source = case["proposition_text"]
        expected_raw = [
            _raw_atom(case_id, source, atom) for atom in case["expected_atoms"]
        ]
        invalid_raw = _injected_invalid_atom(
            case, case["invalid_cross_bindings"][0]
        )
        content = _content(case_id, expected_raw + [invalid_raw], "NONE")
        strict_accepts = _strict_document_accepts(
            content, {"source_id": case_id, "proposition_text": source}
        )
        admission = admit_atoms_independently(
            content,
            {"source_id": case_id, "proposition_text": source},
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )
        exact = request_builder.exact_expected_atom_multiset_match(
            _candidate_case(case), _normalized_for_eval(admission)
        )
        injected_results.append(
            {
                "case_id": case_id,
                "is_synthetic_injection_not_model_output": True,
                "expected_atoms": len(expected_raw),
                "injected_invalid_atoms": 1,
                "strict_document_accepts": strict_accepts,
                "atomwise_status": admission["status"],
                "admitted_atoms": len(admission["admitted"]),
                "quarantined_atoms": len(admission["quarantined"]),
                "exact_expected_atom_multiset_match": exact,
                "quarantined": admission["quarantined"],
            }
        )

    frozen_gold_results = []
    total_gold_atoms = 0
    for proposition in frozen_pack["propositions"]:
        expected = proposition["expected"]
        total_gold_atoms += len(expected["atoms"])
        content = _oracle_frozen_proposal(proposition)
        admission = admit_atoms_independently(
            content,
            {
                "source_id": proposition["source_id"],
                "proposition_text": proposition["proposition_text"],
            },
            scope_id=frozen_r4.FROZEN_SCOPE_ID,
        )
        if expected["atoms"]:
            exact = request_builder.exact_expected_atom_multiset_match(
                {
                    "case_id": proposition["source_id"],
                    "proposition_text": proposition["proposition_text"],
                    "expected_atoms": expected["atoms"],
                },
                _normalized_for_eval(admission),
            )
        else:
            exact = (
                admission["status"] == "ABSTAINED"
                and admission["source_abstention_reason"] == expected["abstention"]
            )
        frozen_gold_results.append(
            {
                "source_id": proposition["source_id"],
                "expected_atoms": len(expected["atoms"]),
                "admitted_atoms": len(admission["admitted"]),
                "quarantined_atoms": len(admission["quarantined"]),
                "exact_expected_result": exact,
            }
        )

    mixed_expected_atoms = 1 + sum(row["expected_atoms"] for row in injected_results)
    mixed_retained_atoms = len(actual_admission["admitted"]) + sum(
        row["admitted_atoms"] for row in injected_results
    )
    mixed_quarantined_atoms = len(actual_admission["quarantined"]) + sum(
        row["quarantined_atoms"] for row in injected_results
    )
    input_paths = {
        "candidate_control_pack": CANDIDATE_PACK_PATH,
        "frozen_r4_control_pack": FROZEN_PACK_PATH,
        "prior_smoke_request": SMOKE_DIR / "request.json",
        "prior_smoke_response": SMOKE_DIR / "response_body.bin",
        "prior_smoke_result": SMOKE_DIR / "result.json",
        "atomwise_adapter": ROOT
        / "tools/research/memory/mem3b0q_r4_atomwise_admission_v1.py",
        "candidate_builder": ROOT
        / "tools/research/memory/mem3b0q_r4_candidate_builder_v1.py",
        "joint_guard_v2": ROOT
        / "tools/research/memory/mem3b0q_r4_joint_binding_guard_v2.py",
        "joint_guard_v3": ROOT
        / "tools/research/memory/mem3b0q_r4_joint_binding_guard_v3.py",
    }
    passed = (
        actual_admission["status"] == "PARTIAL"
        and not actual_strict_accepts
        and actual_exact
        and len(actual_admission["quarantined"]) == 1
        and all(not row["strict_document_accepts"] for row in injected_results)
        and all(row["atomwise_status"] == "PARTIAL" for row in injected_results)
        and all(row["exact_expected_atom_multiset_match"] for row in injected_results)
        and all(row["quarantined_atoms"] == 1 for row in injected_results)
        and all(row["exact_expected_result"] for row in frozen_gold_results)
        and mixed_retained_atoms == mixed_expected_atoms
        and mixed_quarantined_atoms == 3
    )
    return {
        "schema_version": 1,
        "qualification_id": "mem3b0q-r4-atomwise-admission-qualification-v1",
        "status": "PASS" if passed else "FAIL",
        "evaluation_scope": "FROZEN_SYNTHETIC_CONTROLS_ONLY",
        "new_model_calls": 0,
        "hosted_calls": 0,
        "memory_store_mutations": 0,
        "clinical_content": False,
        "method": {
            "version": "mem3b0q-r4-atomwise-admission-v1-development",
            "per_atom_validators_changed": False,
            "failure_granularity_changed": True,
            "invalid_atom_policy": "quarantine_atom_keep_valid_siblings",
            "duplicate_atom_policy": "keep_first_quarantine_later_exact_duplicates",
        },
        "actual_model_response_replay": {
            "case_id": "R4C-05",
            "prior_response_sha256": _sha(SMOKE_DIR / "response_body.bin"),
            "strict_document_accepts": actual_strict_accepts,
            "atomwise_status": actual_admission["status"],
            "expected_atoms": len(actual_case["expected_atoms"]),
            "admitted_atoms": len(actual_admission["admitted"]),
            "quarantined_atoms": len(actual_admission["quarantined"]),
            "exact_expected_atom_multiset_match": actual_exact,
            "quarantined": actual_admission["quarantined"],
        },
        "synthetic_injected_invalid_binding_controls": injected_results,
        "frozen_r4_oracle_control_pack": {
            "case_count": len(frozen_gold_results),
            "expected_atom_count": total_gold_atoms,
            "exact_expected_results": sum(
                row["exact_expected_result"] for row in frozen_gold_results
            ),
            "quarantined_atoms": sum(
                row["quarantined_atoms"] for row in frozen_gold_results
            ),
        },
        "mixed_valid_invalid_proposal_summary": {
            "scenarios": 3,
            "expected_valid_atoms": mixed_expected_atoms,
            "valid_atoms_retained": mixed_retained_atoms,
            "invalid_atoms_quarantined": mixed_quarantined_atoms,
            "valid_atom_retention": mixed_retained_atoms / mixed_expected_atoms,
            "strict_document_level_retention": 0,
        },
        "input_sha256": {
            label: _sha(path) for label, path in input_paths.items()
        },
        "limitations": [
            "The two injected cases are deterministic negative-control corruptions, not model outputs.",
            "The one recorded model response is development evidence and is not an unbiased generalization estimate.",
            "No public LongMemEval or Memora benchmark was run.",
            "Admission remains proposal validation only; no revision materializer or MemoryStore write is tested.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = qualify()
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered + "\n")
    print(rendered)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
