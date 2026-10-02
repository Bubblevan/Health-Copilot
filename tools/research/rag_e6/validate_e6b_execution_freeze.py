"""Validate paired E6B outputs before committing the execution freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.rag_e6b.protocol import ARMS, POOL_SEEDS, RUN_ROOT_RELATIVE, read_json, sha256_file
from eval.rag_e6b.runner import _paired_identity_pass


def validate(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve() / RUN_ROOT_RELATIVE
    manifest_path = root / "execution" / "reserved_execution_manifest.json"
    manifest = read_json(manifest_path)
    outputs_path = root / "execution" / "reserved_reader_outputs.jsonl"
    journal_path = root / "execution" / "reserved_generation_calls.jsonl"
    with outputs_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    materialization = read_json(root / "reserved" / "reserved_materialization_manifest.json")
    expected_total = sum(int(value) for value in materialization["actual_episodes_per_pool"].values())
    if (
        len(rows) != expected_total
        or manifest.get("total_episode_count") != expected_total
        or manifest.get("all_reserved_episodes_executed") is not True
        or manifest.get("all_required_arm_executions_complete") is not True
        or manifest.get("reader_output_sha256") != sha256_file(outputs_path)
        or manifest.get("call_journal_sha256") != sha256_file(journal_path)
        or manifest.get("evaluator_truth_opened") is not False
        or manifest.get("extra_rsel_model_calls") != 0
        or manifest.get("rsel_no_match_parity_pass") is not True
    ):
        raise ValueError("execution freeze cardinality or hash validation failed")
    by_pool = {pool: [] for pool in POOL_SEEDS}
    seen: set[str] = set()
    for row in rows:
        if row.get("episode_id") in seen:
            raise ValueError("duplicate episode in frozen output")
        seen.add(row["episode_id"])
        if row.get("pool") not in by_pool or set(row.get("arms", {})) != set(ARMS):
            raise ValueError("execution row is missing a required arm")
        if not _paired_identity_pass(row):
            raise ValueError("paired evidence identity or no-match parity failed")
        vanilla, rsel = row["arms"][ARMS[0]], row["arms"][ARMS[1]]
        for field in ("ranked_evidence_ids", "candidate_union_ids", "channel_ids",
                      "evidence_identity_sha256", "evidence_aliases"):
            if vanilla.get(field) != rsel.get(field):
                raise ValueError("paired arm evidence identity mismatch")
        by_pool[row["pool"]].append(row)
    expected_by_pool = materialization["actual_episodes_per_pool"]
    if (
        sum(len(items) for items in by_pool.values()) != expected_total
        or any(len(by_pool[pool]) != int(expected_by_pool[pool]) for pool in POOL_SEEDS)
    ):
        raise ValueError("execution pools are incomplete")
    return {
        "status": "PASS_BEFORE_TRUTH_ACCESS",
        "episodes": len(rows),
        "episodes_by_pool": {pool: len(by_pool[pool]) for pool in POOL_SEEDS},
        "paired_evidence_identity": "PASS",
        "rsel_no_match_parity": "PASS",
        "evaluator_truth_opened": False,
        "reader_output_sha256": sha256_file(outputs_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(validate(args.repository_root), sort_keys=True))


if __name__ == "__main__":
    main()
