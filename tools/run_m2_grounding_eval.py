"""Standalone live grounding evaluation over reviewed evidence-relation fixtures."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from health_ai_copilot.contracts import Evidence, GenerationDraft
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.verification.citations import verify_citations
from health_ai_copilot.verification.grounding import (
    GroundedClaim,
    OpenAICompatibleGroundingVerifier,
    validate_grounding_result,
)

KNOWLEDGE_PACK_VERSION = "m0.2-2026-09-15"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run standalone M2 grounding evaluation")
    parser.add_argument("--knowledge-dir", default="data/knowledge_cards")
    parser.add_argument("--dataset", default="evals/m2_grounding.jsonl")
    parser.add_argument("--output-root", default="runs/m2")
    args = parser.parse_args(argv)
    cases = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    cards = {card.id: card for card in load_knowledge_cards(args.knowledge_dir)}
    verifier = OpenAICompatibleGroundingVerifier()
    run_dir = Path(args.output_root) / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for case in cases:
        evidence = [_evidence(cards[source_id]) for source_id in case["evidence_source_ids"]]
        claims = tuple(GroundedClaim(item["text"], tuple(item["citation_ids"])) for item in case["claims"])
        citation_ids = [source_id for claim in claims for source_id in claim.citation_ids]
        integrity = verify_citations(GenerationDraft(case["answer"], citation_ids), evidence)
        result = None
        stage = "deterministic_citation" if not integrity.valid else "semantic_verifier"
        error = None
        if integrity.valid:
            try:
                result = validate_grounding_result(verifier.verify(case["answer"], claims, evidence), claims, evidence)
            except Exception as exc:  # noqa: BLE001
                stage = "verifier_error"
                error = type(exc).__name__
        accepted = bool(result and result.coverage_ok and all(item.verdict.value == "supported" for item in result.claim_results))
        rows.append({"case_id": case["id"], "category": case["category"], "expected_coverage_ok": case["expected_coverage_ok"], "expected_verdicts": case["expected_verdicts"], "accepted": accepted, "rejection_stage": None if accepted else stage, "citation_integrity": integrity.valid, "coverage_ok": result.coverage_ok if result else None, "claim_results": [asdict(item) for item in result.claim_results] if result else [], "error": error})
    metrics = grounding_metrics(rows)
    _write_jsonl(run_dir / "grounding_results.jsonl", rows)
    config = {
        "commit_sha": _git_sha(),
        "model": verifier.model_name,
        "base_url": verifier.base_url,
        "temperature": verifier.temperature,
        "dataset_path": str(Path(args.dataset)),
        "dataset_sha256": _sha256(Path(args.dataset)),
        "knowledge_pack_version": KNOWLEDGE_PACK_VERSION,
        "case_count": len(cases),
        "timeout": verifier.timeout_seconds,
        "retries": verifier.max_retries,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(_report(metrics, rows), encoding="utf-8")
    print(run_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def grounding_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(category: str, predicate) -> float | None:
        selected = [row for row in rows if row["category"] == category]
        return sum(predicate(row) for row in selected) / len(selected) if selected else None
    return {"case_count": len(rows), "supported_accept_rate": rate("supported", lambda row: row["accepted"]), "unsupported_reject_rate": rate("unsupported", lambda row: not row["accepted"]), "contradicted_reject_rate": rate("contradicted", lambda row: not row["accepted"]), "coverage_failure_reject_rate": rate("coverage_missing", lambda row: not row["accepted"]), "fabricated_citation_reject_rate": rate("fabricated_citation", lambda row: not row["accepted"]), "deterministic_citation_rejections": sum(row["rejection_stage"] == "deterministic_citation" for row in rows), "semantic_verifier_rejections": sum(row["rejection_stage"] == "semantic_verifier" for row in rows), "verifier_errors": sum(row["rejection_stage"] == "verifier_error" for row in rows)}


def _evidence(card: Any) -> Evidence:
    return Evidence(card.id, card.title, card.content, card.source_url, 0.0)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _report(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# M2 standalone grounding evaluation",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Fixture verdicts",
        "",
    ]
    for row in rows:
        verdicts = [item["verdict"] for item in row["claim_results"]]
        lines.append(
            f"- `{row['case_id']}` ({row['category']}): expected "
            f"`{row['expected_verdicts']}`, observed `{verdicts}`, accepted="
            f"`{row['accepted']}`, rejection_stage=`{row['rejection_stage']}`."
        )
    return "\n".join(lines) + "\n"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
