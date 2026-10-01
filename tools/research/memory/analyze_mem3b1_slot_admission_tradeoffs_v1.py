"""Compare conservative post-hoc admission gates on the frozen second sample."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/memory/mem3/mem3b1-b0q-reviewed-slot-sample-v2"


def _verify(path: Path) -> str:
    expected, filename = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip().split()
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if expected != actual or filename != path.name:
        raise RuntimeError(f"SHA mismatch: {path.name}")
    return actual


def _variants() -> dict[str, dict[str, float | int]]:
    evaluation_path = RUN / "evaluation_key.json"
    proposals_path = RUN / "slot_proposals.json"
    _verify(evaluation_path)
    _verify(proposals_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    proposals = {
        row["record_id"]: row
        for row in json.loads(proposals_path.read_text(encoding="utf-8"))
    }
    prior = {
        row["revision_slot_id"]: row
        for row in json.loads((RUN / "validation_amendment.json").read_text(encoding="utf-8"))[
            "score_after_reparse"
        ]["group_results"]
    }
    modes = {
        "typed_slot_key_only": [],
        "typed_slot_plus_qualifier_equality": [],
        "typed_slot_plus_distinct_values": [],
        "typed_slot_plus_both": [],
    }
    for group in evaluation["groups"]:
        rows = [proposals[record_id] for record_id in group["member_record_ids"]]
        base = prior[group["revision_slot_id"]]["predicted_admission"]
        qualifier_keys = {
            tuple(sorted(str(value).casefold().strip() for value in row["value_qualifiers"]))
            for row in rows
        }
        value_keys = {str(row["value"]).casefold().strip() for row in rows}
        qualifier_equal = len(qualifier_keys) == 1
        values_distinct = len(value_keys) == len(rows)
        predictions = {
            "typed_slot_key_only": base,
            "typed_slot_plus_qualifier_equality": base and qualifier_equal,
            "typed_slot_plus_distinct_values": base and values_distinct,
            "typed_slot_plus_both": base and qualifier_equal and values_distinct,
        }
        for name, prediction in predictions.items():
            expected = group["review_label"] == "true_singleton_slot"
            modes[name].append(
                {
                    "label": group["review_label"],
                    "prediction": prediction,
                    "correct": prediction == expected,
                }
            )
    summary: dict[str, dict[str, float | int]] = {}
    for name, rows in modes.items():
        true_positive = sum(
            row["label"] == "true_singleton_slot" and row["prediction"] for row in rows
        )
        false_positive = sum(
            row["label"] != "true_singleton_slot" and row["prediction"] for row in rows
        )
        false_negative = sum(
            row["label"] == "true_singleton_slot" and not row["prediction"] for row in rows
        )
        summary[name] = {
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "correct": sum(row["correct"] for row in rows),
            "n": len(rows),
            "admission_precision": (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            ),
            "admission_recall": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            ),
            "false_admissions": false_positive,
            "true_admissions": true_positive,
            "missed_true_slots": false_negative,
        }
    return summary


def main() -> None:
    summary = _variants()
    report = [
        "# Slot Admission Gate Trade-Offs",
        "",
        "Offline analysis of the frozen second 15-group diagnostic. No model call was made and no raw proposal was changed. A reviewed true-singleton group is the positive class; false-merge and uncertain groups are conservatively negative.",
        "",
        "| Gate | Correct | N | Accuracy | Precision | Recall | False admissions | Missed true |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary.items():
        report.append(
            f"| {name} | {row['correct']} | {row['n']} | {row['accuracy']:.3f} | "
            f"{row['admission_precision']:.3f} | {row['admission_recall']:.3f} | "
            f"{row['false_admissions']} | {row['missed_true_slots']} |"
        )
    report.extend(
        [
            "",
            "## Reading",
            "",
            "The broad slot key alone has high recall but merges some complementary or uncertain task dimensions. Requiring exact equality of value qualifiers tests one conservative alternative; requiring all values to differ tests another. These are post-hoc development diagnostics on the same frozen 15 groups and are not independent performance estimates.",
            "",
            "The next method decision must separate slot identity from revision compatibility: a shared topical slot may accumulate evidence without superseding an earlier fact. Materialization should require a separately established, mutually exclusive value transition.",
            "",
        ]
    )
    payload = json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    output = RUN / "admission_gate_tradeoffs.json"
    with output.open("xb") as handle:
        handle.write(payload)
    digest = hashlib.sha256(payload).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    report_path = RUN / "admission_gate_tradeoffs.md"
    report_bytes = "\n".join(report).encode("utf-8")
    with report_path.open("xb") as handle:
        handle.write(report_bytes)
    report_digest = hashlib.sha256(report_bytes).hexdigest()
    report_path.with_suffix(report_path.suffix + ".sha256").write_text(
        f"{report_digest}  {report_path.name}\n", encoding="ascii"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
