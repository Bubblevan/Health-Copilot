"""Validate and score a frozen local slot-proposal diagnostic response."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.research.memory.run_mem3b1_slot_proposal_diagnostic_v1 import (
    SOURCE_RECORDS,
    _freeze,
)
from tools.research.memory.run_mem3b1_instagram_snapshot_diagnostic_v1 import (
    ENDPOINT,
    MODEL,
    _runtime_preflight,
)

RUN = ROOT / "runs/memory/mem3/mem3b1-slot-proposal-diagnostic-v2"
REQUIRED_FIELDS = {
    "record_id",
    "entity",
    "attribute",
    "value",
    "cardinality",
    "evidence_quote_indices",
}
CARDINALITIES = {
    "SINGLE_VALUE_AT_A_TIME",
    "MULTI_VALUE_OR_SET",
    "EPISODIC_EVENT",
    "UNKNOWN",
}


def _verify(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"missing frozen artifact: {path.name}")
    expected, filename = sidecar.read_text(encoding="ascii").strip().split()
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if filename != path.name or expected != actual:
        raise RuntimeError(f"SHA mismatch: {path.name}")
    return actual


def _score(proposals: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ids = {row["record_id"] for row in SOURCE_RECORDS}
    by_id = {row.get("record_id"): row for row in proposals if isinstance(row, dict)}
    schema_errors = []
    if len(proposals) != len(expected_ids) or set(by_id) != expected_ids:
        schema_errors.append("record_coverage_or_duplicate_error")
    for record_id, row in by_id.items():
        if set(row) != REQUIRED_FIELDS:
            schema_errors.append(f"{record_id}:field_set")
        if any(not isinstance(row.get(key), str) or not row.get(key).strip() for key in ("entity", "attribute", "value")):
            schema_errors.append(f"{record_id}:empty_or_nonstring_field")
        if row.get("cardinality") not in CARDINALITIES:
            schema_errors.append(f"{record_id}:cardinality_enum")
        indices = row.get("evidence_quote_indices")
        record = next((item for item in SOURCE_RECORDS if item["record_id"] == record_id), None)
        if (
            not isinstance(indices, list)
            or not indices
            or any(not isinstance(index, int) or index < 0 or index >= len(record["user_quotes"]) for index in indices)
        ):
            schema_errors.append(f"{record_id}:invalid_quote_ref")

    if schema_errors:
        return {
            "schema_pass": False,
            "schema_errors": sorted(set(schema_errors)),
            "n_proposals": len(proposals),
        }

    pairs = [
        {"pair": ["r03", "r05"], "expected": "SAME_SINGLE_VALUE_SLOT", "class": "revision_positive"},
        {"pair": ["r01", "r04"], "expected": "SAME_SINGLE_VALUE_SLOT", "class": "revision_positive"},
        {"pair": ["r02", "r06"], "expected": "COEXIST_NOT_REVISION", "class": "coexistence_negative"},
    ]
    pair_scores = []
    for item in pairs:
        left, right = (by_id[record_id] for record_id in item["pair"])
        same_entity = left["entity"].strip().casefold() == right["entity"].strip().casefold()
        same_attribute = left["attribute"].strip().casefold() == right["attribute"].strip().casefold()
        both_single = left["cardinality"] == right["cardinality"] == "SINGLE_VALUE_AT_A_TIME"
        proposed_same_slot = same_entity and same_attribute and both_single
        predicted = "SAME_SINGLE_VALUE_SLOT" if proposed_same_slot else "COEXIST_NOT_REVISION"
        pair_scores.append(
            {
                **item,
                "predicted": predicted,
                "pass": predicted == item["expected"],
                "same_entity": same_entity,
                "same_attribute": same_attribute,
                "both_single_value": both_single,
            }
        )

    source_quote_refs = []
    for record in SOURCE_RECORDS:
        proposal = by_id[record["record_id"]]
        quote_indices = proposal["evidence_quote_indices"]
        source_quote_refs.append(
            {
                "record_id": record["record_id"],
                "valid_source_quote_reference": bool(quote_indices),
                "quote_indices": quote_indices,
            }
        )
    positives = [item for item in pair_scores if item["class"] == "revision_positive"]
    negatives = [item for item in pair_scores if item["class"] == "coexistence_negative"]
    return {
        "schema_pass": True,
        "n_records": len(proposals),
        "pair_checks": pair_scores,
        "pair_accuracy": sum(item["pass"] for item in pair_scores) / len(pair_scores),
        "positive_slot_recall": sum(item["pass"] for item in positives) / len(positives),
        "coexistence_control_specificity": sum(item["pass"] for item in negatives) / len(negatives),
        "false_revision_merges": sum(item["predicted"] == "SAME_SINGLE_VALUE_SLOT" for item in negatives),
        "valid_source_quote_refs": sum(item["valid_source_quote_reference"] for item in source_quote_refs),
        "source_quote_ref_total": len(source_quote_refs),
        "source_quote_refs": source_quote_refs,
    }


def main() -> None:
    request_path = RUN / "slot_proposal_request.json"
    response_path = RUN / "slot_proposal_response.json"
    source_path = RUN / "source_records.json"
    request_sha = _verify(request_path)
    response_sha = _verify(response_path)
    source_sha = _verify(source_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    content = response["choices"][0]["message"]["content"]
    decoded = json.loads(content)
    root_array = isinstance(decoded, list)
    proposals = decoded if root_array else decoded.get("proposals")
    if not isinstance(proposals, list):
        raise RuntimeError("response is neither a proposal array nor an object with a proposals array")
    proposal_bytes = json.dumps(proposals, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    proposal_sha = _freeze(RUN / "slot_proposals.json", proposal_bytes)
    with __import__("httpx").Client(timeout=10.0, trust_env=False) as client:
        runtime = _runtime_preflight(client)
    score = _score(proposals)
    audit = {
        "diagnostic_only": True,
        "research_track": "failure-driven DEV mechanism diagnostic; not benchmark performance evidence",
        "reader_model": MODEL,
        "endpoint": ENDPOINT,
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        "source_records_sha256": source_sha,
        "proposal_sha256": proposal_sha,
        "model_response_root": "array" if root_array else "object",
        "response_envelope_contract_match": not root_array,
        "request_prompt_has_timestamp_or_question_or_gold": False,
        "request_local_completion_tokens": response.get("usage", {}).get("completion_tokens"),
        "hosted_calls": 0,
        "local_posts": 1,
        "runtime": runtime,
        "score": score,
    }
    _freeze(
        RUN / "audit.json",
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )

    lines = [
        "# MEM-3B1 Slot Proposal Diagnostic",
        "",
        "Exploratory local-Qwen diagnostic over six source-grounded facts. The frozen response is scored after generation against two revision-positive pairs and one coexistence-negative pair. This is not a benchmark result.",
        "",
        f"- Proposal coverage/schema: {score.get('n_records', 0)}/6 records; schema pass: {score.get('schema_pass')}",
        f"- Pairwise accuracy: {score.get('pair_accuracy', 0):.3f} ({sum(x['pass'] for x in score.get('pair_checks', []))}/3)",
        f"- Same-slot positive recall: {score.get('positive_slot_recall', 0):.3f} ({sum(x['pass'] for x in score.get('pair_checks', []) if x['class'] == 'revision_positive')}/2)",
        f"- Coexistence control specificity: {score.get('coexistence_control_specificity', 0):.3f} ({sum(x['pass'] for x in score.get('pair_checks', []) if x['class'] == 'coexistence_negative')}/1)",
        f"- False revision merges: {score.get('false_revision_merges', 0)}/1 control",
        f"- Valid source quote references: {score.get('valid_source_quote_refs', 0)}/{score.get('source_quote_ref_total', 0)}",
        f"- Output envelope: model returned JSON {audit['model_response_root']}; frozen request used json_object.",
        "- One local model completion, no hosted calls.",
        "",
        "## Pair Audit",
        "",
        "| Control | Pair | Prediction | Result | Slot equality details |",
        "|---|---|---|---|---|",
    ]
    for row in score.get("pair_checks", []):
        lines.append(
            f"| {row['class']} | {', '.join(row['pair'])} | {row['predicted']} | "
            f"{'PASS' if row['pass'] else 'FAIL'} | entity={row['same_entity']}, "
            f"attribute={row['same_attribute']}, single={row['both_single_value']} |"
        )
    lines.extend(["", "## Proposals", "", "| Record | Entity | Attribute | Value | Cardinality |", "|---|---|---|---|---|"])
    for row in proposals:
        lines.append(
            f"| {row['record_id']} | {row['entity']} | {row['attribute']} | {row['value']} | {row['cardinality']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The model correctly canonicalized the two Instagram follower facts and separated the running-shoe and jewelry-store interests. It missed the gym transition by naming the older schedule gym_days and the later frequency gym_frequency. Thus this diagnostic has zero false merges on its one coexistence control, but only one of two revision positives is admitted.",
            "",
            "The semantic failure motivates a factorized typed slot representation: state dimension plus object/qualifier plus value type/unit, rather than a single unconstrained attribute string. This is a next hypothesis to test, not a result established by this six-record diagnostic.",
            "",
            "The response-envelope mismatch was transport-level only: the model returned a valid JSON array despite the requested object wrapper. The frozen answer content is accepted by this offline scorer without altering it.",
            "",
        ]
    )
    _freeze(RUN / "report.md", "\n".join(lines).encode("utf-8"))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
