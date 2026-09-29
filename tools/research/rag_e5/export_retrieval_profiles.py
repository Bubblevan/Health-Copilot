"""Write the machine-readable frozen OFF/STANDARD/STRONG profile registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from health_ai_copilot.retrieval_capabilities import retrieval_action_specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": "rag-e5-retrieval-action-profiles-v1",
        "external_corpus_identity": None,
        "profiles": [item.to_dict() for item in retrieval_action_specs()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
