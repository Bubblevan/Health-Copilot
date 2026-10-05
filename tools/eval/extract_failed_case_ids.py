"""Freeze case IDs whose prior checkpoint failed at the local model transport layer."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

RETRYABLE_REASONS = {
    "reasoning_failure:APIConnectionError",
    "reasoning_failure:APITimeoutError",
    "reasoning_failure:TimeoutError",
}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_case_ids(run_dir: Path) -> dict[str, object]:
    checkpoint = run_dir / "cases.jsonl"
    invalidation_path = run_dir / "invalidation.json"
    if not checkpoint.is_file() or not invalidation_path.is_file():
        raise ValueError("source run must contain cases.jsonl and invalidation.json")
    invalidation = json.loads(invalidation_path.read_text(encoding="utf-8"))
    if invalidation.get("status") != "INVALID_INFRASTRUCTURE_FAILURE":
        raise ValueError("source run is not marked INVALID_INFRASTRUCTURE_FAILURE")
    if invalidation.get("profile_id") != "B2":
        raise ValueError("selective transport recovery is restricted to the B2 arm")

    counts: Counter[str] = Counter()
    selected: list[str] = []
    seen: set[str] = set()
    checkpoint_dataset_ids: set[str] = set()
    with checkpoint.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            checkpoint_dataset_ids.add(str(row.get("dataset_id")))
            if row.get("profile_id") != "B2":
                raise ValueError(f"profile mismatch in checkpoint line {line_no}")
            reasons = row.get("response", {}).get("safety_flags", ())
            reason = next((item for item in reasons if item in RETRYABLE_REASONS), None)
            if reason is None:
                continue
            case_id = str(row["case_id"])
            if case_id in seen:
                raise ValueError(f"duplicate case ID in checkpoint: {case_id}")
            seen.add(case_id)
            selected.append(case_id)
            counts[reason.removeprefix("reasoning_failure:")] += 1

    expected = {
        str(reason): int(count)
        for reason, count in invalidation.get("connection_failures", {}).items()
    }
    observed = dict(counts)
    if observed != expected:
        raise ValueError(f"transport failure counts disagree with invalidation: {observed} != {expected}")
    if len(checkpoint_dataset_ids) != 1:
        raise ValueError(f"checkpoint has inconsistent dataset IDs: {sorted(checkpoint_dataset_ids)}")
    checkpoint_dataset_id = next(iter(checkpoint_dataset_ids))
    declared_dataset_id = str(invalidation.get("dataset_id"))
    compatible_dataset_ids = {
        "diagnosisarena-915": "diagnosisarena",
    }
    if compatible_dataset_ids.get(declared_dataset_id, declared_dataset_id) != checkpoint_dataset_id:
        raise ValueError(
            f"dataset mismatch: invalidation={declared_dataset_id}, checkpoint={checkpoint_dataset_id}"
        )
    selected.sort()
    id_hash = sha256(("\n".join(selected) + "\n").encode("utf-8")).hexdigest()
    return {
        "schema_version": "harness-v1-targeted-retry-case-ids-v1",
        "dataset_id": checkpoint_dataset_id,
        "invalidation_dataset_id": declared_dataset_id,
        "profile_id": invalidation["profile_id"],
        "selection_reason": "retry transport failures only; retain all non-transport rows unchanged",
        "retryable_failure_counts": dict(sorted(observed.items())),
        "source_run_dir": str(run_dir.resolve()),
        "source_cases_sha256": sha256_file(checkpoint),
        "source_invalidation_sha256": sha256_file(invalidation_path),
        "case_count": len(selected),
        "case_ids_sha256": id_hash,
        "case_ids": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = extract_case_ids(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({key: manifest[key] for key in (
        "dataset_id", "case_count", "retryable_failure_counts", "case_ids_sha256",
    )}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
