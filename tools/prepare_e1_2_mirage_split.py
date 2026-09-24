"""Freeze a leakage-aware, question-only split for the E1.2 MIRAGE experiment.

The manifest contains case IDs and question fingerprints only. It deliberately
does not copy questions, answer choices, or gold labels into the split artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / ".health-bench-data/normalized/medical-mirage-v1/cases.jsonl"
DEFAULT_IDENTITY = ROOT / ".health-bench-data/normalized/medical-mirage-v1/identity.json"
DEFAULT_RAW = ROOT / ".health-bench-data/raw/medical-mirage-v1/benchmark.json"
DEFAULT_OUTPUT = ROOT / "runs/e1_2/mirage_split_manifest.json"
EXPECTED_RAW_SHA256 = "6f7f08c64cd2efe02a5d0c247229813c90db345d9dd6e3a451b5d24146d0f8fa"
EXPECTED_CASES_SHA256 = "3a31d3e7fbe6b1184afdbd46925db28747e2ec60d3e53948d8b7ce8ecd614923"
EXPECTED_CASE_COUNT = 7663
EXPOSED_SUBDATASETS = frozenset({"bioasq", "pubmedqa"})
CLEAN_SUBDATASETS = frozenset({"medqa", "medmcqa", "mmlu"})
SPLIT_SEED = "health-copilot-e1.2-mirage-question-group-v1"
DEV_FRACTION = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _question_fingerprint(question: str) -> str:
    canonical = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _order_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def build_split_manifest(
    cases: list[dict[str, Any]],
    *,
    raw_sha256: str,
    normalized_sha256: str,
    seed: str = SPLIT_SEED,
    dev_fraction: float = DEV_FRACTION,
) -> dict[str, Any]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between zero and one")
    if not seed.strip():
        raise ValueError("seed must be non-empty")
    if not cases:
        raise ValueError("MIRAGE cases are empty")

    seen_ids: set[str] = set()
    normalized: list[dict[str, str]] = []
    group_members: dict[str, list[int]] = defaultdict(list)
    totals: Counter[str] = Counter()

    for index, case in enumerate(cases):
        case_id = case.get("case_id")
        metadata = case.get("metadata")
        payload = case.get("payload")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"case at index {index} has no case_id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        if not isinstance(metadata, dict) or not isinstance(payload, dict):
            raise TypeError(f"case {case_id} is missing metadata or payload")
        subdataset = str(metadata.get("subdataset", "")).strip().lower()
        question = payload.get("question")
        if not subdataset or not isinstance(question, str) or not question.strip():
            raise ValueError(f"case {case_id} is missing its subdataset or payload.question")
        if subdataset not in EXPOSED_SUBDATASETS | CLEAN_SUBDATASETS:
            raise ValueError(f"unrecognized MIRAGE subdataset: {subdataset}")

        question_hash = _question_fingerprint(question)
        normalized.append(
            {"case_id": case_id, "subdataset": subdataset, "question_sha256": question_hash}
        )
        if subdataset in CLEAN_SUBDATASETS:
            group_members[question_hash].append(len(normalized) - 1)
            totals[subdataset] += 1

    if set(totals) != CLEAN_SUBDATASETS:
        raise ValueError("all three untouched MIRAGE subdatasets must be present")

    targets = {name: max(1, math.floor(count * dev_fraction + 0.5)) for name, count in totals.items()}
    dev_counts: Counter[str] = Counter()
    group_splits: dict[str, str] = {}
    ordered_groups = sorted(group_members, key=lambda value: _order_key(seed, value))

    # Assign exact-question duplicate groups together. Greedily fill the
    # per-subdataset DEV targets without consulting answer labels.
    for question_hash in ordered_groups:
        sizes = Counter(normalized[index]["subdataset"] for index in group_members[question_hash])
        dev_cost = sum(
            ((dev_counts[name] + size - targets[name]) ** 2 - (dev_counts[name] - targets[name]) ** 2)
            / (targets[name] ** 2)
            for name, size in sizes.items()
        )
        if dev_cost <= 0:
            group_splits[question_hash] = "DEV"
            dev_counts.update(sizes)
        else:
            group_splits[question_hash] = "TEST"

    entries: list[dict[str, str]] = []
    for case in normalized:
        subdataset = case["subdataset"]
        assignment = (
            "EXPOSED_HISTORY"
            if subdataset in EXPOSED_SUBDATASETS
            else group_splits[case["question_sha256"]]
        )
        entries.append({**case, "split": assignment})
    entries.sort(key=lambda row: row["case_id"])

    counts: dict[str, dict[str, int]] = {}
    for name in sorted(EXPOSED_SUBDATASETS | CLEAN_SUBDATASETS):
        counts[name] = dict(
            sorted(
                Counter(row["split"] for row in entries if row["subdataset"] == name).items()
            )
        )
    for name in CLEAN_SUBDATASETS:
        if not counts[name].get("DEV") or not counts[name].get("TEST"):
            raise ValueError(f"split produced an empty DEV or TEST partition for {name}")

    return {
        "schema_version": "e1-2-mirage-split-v1",
        "benchmark": "Medical MIRAGE / MedRAG",
        "source_identity": {
            "raw_benchmark_sha256": raw_sha256,
            "normalized_cases_sha256": normalized_sha256,
            "normalized_case_count": len(entries),
        },
        "protocol": {
            "seed": seed,
            "dev_fraction_target": dev_fraction,
            "assignment": "question-only SHA-256 order with duplicate normalized questions grouped across untouched subdatasets; greedy per-subdataset DEV target fill",
            "question_normalization": "Unicode NFKC, casefold, collapse whitespace",
            "exposed_history_subdatasets": sorted(EXPOSED_SUBDATASETS),
            "clean_subdatasets": sorted(CLEAN_SUBDATASETS),
            "label_or_option_fields_read": False,
        },
        "counts": counts,
        "cases": entries,
    }


def load_cases(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            result.append(row)
    return result


def prepare_manifest(
    *,
    cases_path: Path = DEFAULT_CASES,
    identity_path: Path = DEFAULT_IDENTITY,
    raw_path: Path = DEFAULT_RAW,
) -> dict[str, Any]:
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    raw_hash = sha256_file(raw_path)
    normalized_hash = sha256_file(cases_path)
    cases = load_cases(cases_path)
    checks = {
        "raw hash": (raw_hash, EXPECTED_RAW_SHA256),
        "identity raw hash": (identity.get("raw_artifact_sha256", [None])[0], EXPECTED_RAW_SHA256),
        "normalized hash": (normalized_hash, EXPECTED_CASES_SHA256),
        "identity normalized hash": (identity.get("normalized_sha256"), EXPECTED_CASES_SHA256),
        "identity count": (identity.get("case_count"), EXPECTED_CASE_COUNT),
        "actual count": (len(cases), EXPECTED_CASE_COUNT),
    }
    failures = [name for name, (actual, expected) in checks.items() if actual != expected]
    if failures:
        raise ValueError(f"frozen E0 MIRAGE identity check failed: {', '.join(failures)}")
    return build_split_manifest(
        cases,
        raw_sha256=raw_hash,
        normalized_sha256=normalized_hash,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--raw-benchmark", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen split manifest: {args.output}")
    manifest = prepare_manifest(
        cases_path=args.cases,
        identity_path=args.identity,
        raw_path=args.raw_benchmark,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary split artifact already exists: {temporary}")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": "frozen", "counts": manifest["counts"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
