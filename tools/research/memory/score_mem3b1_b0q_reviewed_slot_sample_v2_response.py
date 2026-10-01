"""Re-score a frozen v2 response under the documented nullable-unit semantics."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.research.memory.run_mem3b1_b0q_reviewed_slot_sample_v2 import (
    OUTPUT,
    _freeze,
    _score,
)


def _verify(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected, filename = sidecar.read_text(encoding="ascii").strip().split()
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if expected != actual or filename != path.name:
        raise RuntimeError(f"frozen artifact SHA mismatch: {path.name}")
    return actual


def main() -> None:
    original_audit_path = OUTPUT / "audit.json"
    original_audit_sha = _verify(original_audit_path)
    response_sha = _verify(OUTPUT / "slot_proposal_response.json")
    proposal_sha = _verify(OUTPUT / "slot_proposals.json")
    evaluation_sha = _verify(OUTPUT / "evaluation_key.json")
    raw_audit = json.loads(original_audit_path.read_text(encoding="utf-8"))
    proposals = json.loads((OUTPUT / "slot_proposals.json").read_text(encoding="utf-8"))
    evaluation = json.loads((OUTPUT / "evaluation_key.json").read_text(encoding="utf-8"))
    score = _score(proposals, evaluation)
    amendment = {
        "amendment_type": "parser_semantics_correction; raw model response unchanged",
        "original_audit_sha256": original_audit_sha,
        "raw_response_sha256": response_sha,
        "proposal_sha256": proposal_sha,
        "evaluation_key_sha256": evaluation_sha,
        "original_schema_pass": raw_audit["score"].get("schema_pass"),
        "nullable_unit_semantics": "JSON null is equivalent to unit=none because prompt allowed no unit",
        "model_response_modified": False,
        "hosted_calls": 0,
        "score_after_reparse": score,
    }
    _freeze(
        OUTPUT / "validation_amendment.json",
        json.dumps(amendment, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )
    score_view = score
    report = [
        "# V2 Review-Sample Scoring Amendment",
        "",
        "The raw response and original first-pass audit remain unchanged. The initial validator rejected JSON null for unit, although the prompt allowed no unit. This offline amendment treats null as none and re-evaluates the same frozen response.",
        "",
        f"- Raw response/schema check: unit-format mismatch on {sum('unit' in x for x in raw_audit['score'].get('schema_errors', []))}/{len(proposals)} records.",
        f"- Semantic reparse schema pass: {score_view.get('schema_pass')}",
        f"- Groups: {score_view.get('n_groups', 0)}/15",
        f"- Group accuracy: {score_view.get('group_accuracy', 0):.3f}",
        f"- Admission precision: {score_view.get('admission_precision')}",
        f"- Admission recall: {score_view.get('admission_recall')}",
        f"- False admissions: {score_view.get('false_admissions')}; missed true slots: {score_view.get('missed_true_slots')}",
        "",
        "## Strata",
        "",
        "| Review label | Correct | N | Accuracy | Predicted admissions |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in score_view.get("strata", {}).items():
        report.append(
            f"| {label} | {item['correct']} | {item['n']} | {item['accuracy']:.3f} | "
            f"{item['predicted_admissions']} |"
        )
    report.extend(["", "## Group Decisions", "", "| Review label | Admit | Correct |", "|---|---:|---:|"])
    for group in score_view.get("group_results", []):
        report.append(
            f"| {group['review_label']} | {group['predicted_admission']} | {group['pass']} |"
        )
    report.extend(
        [
            "",
            "## Protocol Note",
            "",
            "This is a parser-semantics correction on an already frozen response, not a model rerun or method update. If the corrected score is usable, the code-level validator must be fixed before any next run; this response remains the only observation for this sample.",
            "",
        ]
    )
    _freeze(OUTPUT / "validation_amendment.md", "\n".join(report).encode("utf-8"))
    print(json.dumps(amendment, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
