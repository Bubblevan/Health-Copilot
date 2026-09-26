from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SPLIT_MANIFEST = ROOT / "docs" / "research" / "memory" / "split_manifest.json"
MODEL_PROTOCOL = ROOT / "docs" / "research" / "memory" / "model_protocol.json"


def _sorted_digest(question_ids: list[str]) -> str:
    payload = "".join(f"{question_id}\n" for question_id in sorted(question_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest() -> dict:
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    model = json.loads(MODEL_PROTOCOL.read_text(encoding="utf-8"))
    dev_ids = set(split["dev"]["question_ids"])
    sanity = model["upstream_parity_sanity_track"]["question_ids"]
    sanity_ids = list(sanity.values())
    if len(set(sanity_ids)) != 6 or not set(sanity_ids) <= dev_ids:
        raise ValueError("Frozen upstream sanity IDs must be six unique DEV cases")

    remaining = dev_ids - set(sanity_ids)
    extra_ids = sorted(
        remaining,
        key=lambda question_id: hashlib.sha256(
            f"memeval_main_smoke_v1:{question_id}".encode("utf-8")
        ).hexdigest(),
    )[:4]
    if len(extra_ids) != 4:
        raise ValueError("The frozen DEV split must have at least four remaining IDs")

    question_ids = sanity_ids + extra_ids
    return {
        "manifest_version": "memeval-main-smoke-10-v1",
        "status": "FROZEN_BEFORE_ANY_10_CASE_RESULTS",
        "dataset_id": split["dataset_id"],
        "dataset_revision": split["dataset_revision"],
        "dataset_sha256": split["dataset_sha256"],
        "dev_manifest": "split_manifest.json",
        "dev_ids_sha256_sorted_lf": split["dev"]["question_ids_sha256_sorted_lf"],
        "selection_algorithm": (
            "Start with the six frozen upstream sanity IDs (one per question_type); "
            "then sort remaining DEV IDs by SHA256('memeval_main_smoke_v1:' + question_id) "
            "ascending and take four."
        ),
        "upstream_sanity_ids_by_question_type": sanity,
        "additional_dev_ids": extra_ids,
        "question_count": len(question_ids),
        "question_ids": question_ids,
        "question_ids_sha256_sorted_lf": _sorted_digest(question_ids),
        "diagnostic_only": True,
        "test_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
