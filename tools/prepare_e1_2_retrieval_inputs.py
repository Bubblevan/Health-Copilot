"""Materialize a split-specific MIRAGE retrieval input without labels/options leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.prepare_e1_2_mirage_split import DEFAULT_CASES, DEFAULT_OUTPUT, load_cases

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_MANIFEST = DEFAULT_OUTPUT
CLEAN_SUBDATASETS = frozenset({"medqa", "medmcqa", "mmlu"})


def build_retrieval_benchmark(
    cases: list[dict[str, Any]], split_manifest: dict[str, Any], split: str
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, int]]:
    split_name = split.upper()
    if split_name not in {"DEV", "TEST"}:
        raise ValueError("split must be DEV or TEST")
    assignments = {
        row["case_id"]: row
        for row in split_manifest.get("cases", [])
        if row.get("split") == split_name
    }
    benchmark: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in CLEAN_SUBDATASETS}
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        split_row = assignments.get(case_id)
        if split_row is None:
            continue
        subdataset = str(split_row.get("subdataset", "")).lower()
        if subdataset not in CLEAN_SUBDATASETS:
            raise ValueError(f"unexpected subdataset in clean split: {subdataset}")
        if case_id in seen:
            raise ValueError(f"duplicate case_id in normalized data: {case_id}")
        seen.add(case_id)
        prefix = f"{subdataset}:"
        if not isinstance(case_id, str) or not case_id.startswith(prefix):
            raise ValueError(f"case_id does not use the frozen subdataset prefix: {case_id}")
        payload = case.get("payload")
        if not isinstance(payload, dict):
            raise TypeError(f"case {case_id} has no payload object")
        question = payload.get("question")
        options = payload.get("options")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"case {case_id} has no payload.question")
        if not isinstance(options, dict) or not options:
            raise ValueError(f"case {case_id} has no non-empty payload.options")
        raw_id = case_id[len(prefix) :]
        if raw_id in benchmark[subdataset]:
            raise ValueError(f"duplicate upstream case id in {subdataset}: {raw_id}")
        # Retrieval helpers receive no gold label, metadata, or answer field.
        benchmark[subdataset][raw_id] = {"question": question, "options": options}
        counts[subdataset] += 1

    expected = split_manifest.get("counts", {})
    for subdataset in CLEAN_SUBDATASETS:
        expected_count = int(expected.get(subdataset, {}).get(split_name, 0))
        if counts[subdataset] != expected_count:
            raise ValueError(
                f"materialized {subdataset} count {counts[subdataset]} != frozen count {expected_count}"
            )
    return benchmark, dict(sorted(counts.items()))


def materialize(
    *, cases_path: Path, split_manifest_path: Path, split: str, output_path: Path
) -> dict[str, Any]:
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    cases_hash = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    expected_cases_hash = split_manifest.get("source_identity", {}).get(
        "normalized_cases_sha256"
    )
    if cases_hash != expected_cases_hash:
        raise ValueError("normalized MIRAGE data hash does not match the frozen split manifest")
    benchmark, counts = build_retrieval_benchmark(load_cases(cases_path), split_manifest, split)
    content = json.dumps(benchmark, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    encoded = content.encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = output_path.read_bytes()
        if existing != encoded:
            raise FileExistsError(f"existing retrieval input differs; refusing overwrite: {output_path}")
        reused = True
    else:
        temporary = output_path.with_name(output_path.name + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"temporary retrieval input exists: {temporary}")
        temporary.write_bytes(encoded)
        temporary.replace(output_path)
        reused = False
    result = {
        "status": "ready",
        "split": split.upper(),
        "counts": counts,
        "case_count": sum(counts.values()),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "normalized_cases_sha256": cases_hash,
        "split_manifest_sha256": hashlib.sha256(split_manifest_path.read_bytes()).hexdigest(),
        "contains_gold_or_answer_fields": False,
        "output": str(output_path),
        "reused": reused,
    }
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    manifest_value = {key: value for key, value in result.items() if key != "reused"}
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest_value:
            raise FileExistsError(
                f"existing retrieval input manifest differs; refusing overwrite: {manifest_path}"
            )
    else:
        temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        if temporary_manifest.exists():
            raise FileExistsError(f"temporary retrieval input manifest exists: {temporary_manifest}")
        temporary_manifest.write_text(
            json.dumps(manifest_value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_manifest.replace(manifest_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--split", choices=("DEV", "TEST"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(
        cases_path=args.cases,
        split_manifest_path=args.split_manifest,
        split=args.split,
        output_path=args.output,
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
