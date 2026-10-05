"""Run a source-title retrieval wiring smoke without opening benchmark rows."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.providers.common_medical_kb_v1 import common_medical_kb_v1_provider_factory


async def run(output: Path) -> dict:
    config_path = ROOT / "configs/eval/common_medical_kb_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cards_by_id = {
        card.id: card for card in load_knowledge_cards(ROOT / "data/knowledge_cards")
    }
    provider = common_medical_kb_v1_provider_factory(config)
    cases = []
    for source_id in config["source_list"]:
        card = cards_by_id[source_id]
        result = await provider.retrieve(
            query=card.title,
            context=SimpleNamespace(
                request_id=f"common-kb-title-smoke:{source_id}",
                subject_id=None,
                as_of_time=None,
            ),
        )
        ranked_ids = [item.source_id for item in result.evidence]
        rank = ranked_ids.index(source_id) + 1 if source_id in ranked_ids else None
        cases.append(
            {
                "query_source_id": source_id,
                "top_k": len(ranked_ids),
                "retrieved_source_ids": ranked_ids,
                "target_rank_at_k5": rank,
                "evidence_sha256": result.evidence_sha256,
            }
        )
    hits = sum(case["target_rank_at_k5"] is not None for case in cases)
    report = {
        "schema_version": "common-medical-kb-v1-title-smoke",
        "classification": "DIAGNOSTIC_NOT_ACCURACY",
        "benchmark_rows_opened": False,
        "provider_model_calls": 0,
        "corpus_id": config["corpus_id"],
        "corpus_sha256": config["corpus_sha256"],
        "index_hash": config["index_hash"],
        "query_count": len(cases),
        "target_source_recall_at_5": hits / len(cases),
        "target_source_hits_at_5": hits,
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/common_eval/harness-v1-base-20261005/common-kb-v1-qualification/title_smoke.json",
    )
    args = parser.parse_args()
    report = asyncio.run(run(args.output))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "query_count": report["query_count"],
                "hits_at_5": report["target_source_hits_at_5"],
                "index_hash": report["index_hash"],
                "classification": report["classification"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
