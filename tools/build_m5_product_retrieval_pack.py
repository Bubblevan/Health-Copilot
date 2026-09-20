"""Build the M5 retrieval suite from already frozen/reviewed component rows.

This is a mechanical provenance-preserving assembly, not annotation generation:
it never creates a new question, changes expected source IDs, or asserts a new
human review.  The manifest records the exact reviewed components it reuses.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0", default="evals/m0.jsonl")
    parser.add_argument("--expansion", default="evals/expansion/m0_hypertension_candidates_v1.jsonl")
    parser.add_argument(
        "--expansion-review", default="evals/expansion/m0_hypertension_expansion_v1.review.json"
    )
    parser.add_argument("--output", default="evals/retrieval/m5_product_retrieval_v1.jsonl")
    args = parser.parse_args(argv)
    m0_path, expansion_path, review_path, output_path = map(
        Path, (args.m0, args.expansion, args.expansion_review, args.output)
    )
    selected_m0 = _select_m0(_read_jsonl(m0_path))
    expansion = _read_jsonl(expansion_path)
    if len(selected_m0) != 40 or len(expansion) != 40:
        raise ValueError("M5 v1 requires 40 frozen M0 rows and 40 reviewed expansion rows")
    components = [(row, m0_path, "frozen_m0_baseline") for row in selected_m0] + [
        (row, expansion_path, "approved_m0_expansion_component") for row in expansion
    ]
    rows = [
        _pack_row(row, index, source_path=source_path, component_state=component_state)
        for index, (row, source_path, component_state) in enumerate(components, 1)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, rows)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    manifest = {
        "release_id": "m5-product-retrieval-v1",
        "dataset_path": str(output_path).replace("\\", "/"),
        "dataset_sha256": _sha256(output_path),
        "case_count": len(rows),
        "source_anchored_case_count": sum(bool(row["expected_source_ids"]) for row in rows),
        "corpus_uncovered_control_count": sum(not row["expected_source_ids"] for row in rows),
        "assembly": "mechanical reuse of frozen/reviewed component rows; no new annotations",
        "component_datasets": [
            {"path": str(m0_path).replace("\\", "/"), "sha256": _sha256(m0_path), "state": "frozen"},
            {
                "path": str(expansion_path).replace("\\", "/"),
                "sha256": _sha256(expansion_path),
                "state": "review-manifest-approved",
                "review_manifest_path": str(review_path).replace("\\", "/"),
                "review_manifest_sha256": _sha256(review_path),
                "component_release_id": review["release_id"],
            },
        ],
        "new_annotation_or_user_approval_asserted": False,
    }
    (output_path.parent / "m5_product_retrieval_v1.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(output_path)
    return 0


def _select_m0(rows):
    """Select 40 source-labelled frozen rows, first covering each source ID."""

    source_rows = [row for row in rows if row.get("expected_source_ids")]
    selected, seen_ids = [], set()
    for row in source_rows:
        primary = row["expected_source_ids"][0]
        if primary not in seen_ids:
            selected.append(row)
            seen_ids.add(primary)
    for row in source_rows:
        if len(selected) >= 40:
            break
        if row not in selected and row.get("challenge_type"):
            selected.append(row)
    for row in source_rows:
        if len(selected) >= 40:
            break
        if row not in selected:
            selected.append(row)
    return selected[:40]


def _pack_row(row, index, *, source_path, component_state):
    return {
        "id": f"m5r-{index:03d}",
        "question": row["question"],
        "expected_source_ids": row.get("expected_source_ids", []),
        "category": row.get("category", "unclassified"),
        "challenge_type": row.get("challenge_type") or "legacy_untyped",
        "status": "derived_from_reviewed_component",
        "review_metadata": {
            "source_dataset": str(source_path).replace("\\", "/"),
            "source_case_id": row["id"],
            "component_state": component_state,
            "new_annotation": False,
        },
    }


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
