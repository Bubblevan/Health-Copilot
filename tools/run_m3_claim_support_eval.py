"""Standalone live M3 claim-support evaluation without semantic answer coverage."""

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
from health_ai_copilot.knowledge.scope import load_knowledge_scope
from health_ai_copilot.verification.citations import verify_citations
from health_ai_copilot.verification.grounding import (
    ClaimVerdict,
    GroundedClaim,
    OpenAICompatibleClaimSupportVerifier,
    materialize_cited_evidence,
    validate_claim_support_result,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run standalone M3 claim-support evaluation")
    parser.add_argument("--knowledge-dir", default="data/knowledge_cards")
    parser.add_argument("--knowledge-scope", default="data/knowledge_scope.json")
    parser.add_argument("--dataset", default="evals/m3_claim_support.jsonl")
    parser.add_argument("--output-root", default="runs/m3")
    args = parser.parse_args(argv)
    cards = {card.id: card for card in load_knowledge_cards(args.knowledge_dir)}
    scope = load_knowledge_scope(args.knowledge_scope, list(cards.values()))
    cases = _load_cases(args.dataset)
    verifier = OpenAICompatibleClaimSupportVerifier()
    run_dir = Path(args.output_root) / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for case in cases:
        evidence = [_evidence(cards[source_id]) for source_id in case["evidence_source_ids"]]
        claims = tuple(
            GroundedClaim(item["text"], tuple(item["citation_ids"]))
            for item in case["claims"]
        )
        claim_ids = [source_id for claim in claims for source_id in claim.citation_ids]
        integrity = verify_citations(GenerationDraft("", claim_ids), evidence)
        result = None
        stage = "deterministic_citation" if not integrity.valid else "semantic_verifier"
        error = None
        if integrity.valid:
            try:
                cited_evidence = materialize_cited_evidence(claims, evidence)
                result = validate_claim_support_result(
                    verifier.verify(claims, cited_evidence), claims, evidence
                )
            except Exception as exc:  # noqa: BLE001 - evaluator records controlled failures
                stage = "verifier_error"
                error = type(exc).__name__
        accepted = bool(result and all(item.verdict == ClaimVerdict.SUPPORTED for item in result.claim_results))
        rows.append(
            {
                "case_id": case["id"],
                "category": case["category"],
                "expected_verdicts": case["expected_verdicts"],
                "accepted": accepted,
                "rejection_stage": None if accepted else stage,
                "citation_integrity": integrity.valid,
                "claim_results": [asdict(item) for item in result.claim_results] if result else [],
                "error": error,
            }
        )
    metrics = claim_support_metrics(rows)
    _write_jsonl(run_dir / "results.jsonl", rows)
    config = {
        "commit_sha": _git_sha(),
        "model": verifier.model_name,
        "policy_model": None,
        "verifier_model": verifier.model_name,
        "same_model_verification": None,
        "base_url": verifier.base_url,
        "temperature": verifier.temperature,
        "policy_temperature": None,
        "verifier_temperature": verifier.temperature,
        "dataset_path": str(Path(args.dataset)),
        "dataset_sha256": _sha256(Path(args.dataset)),
        "knowledge_pack_version": scope.knowledge_pack_version,
        "knowledge_scope_id": scope.scope_id,
        "knowledge_scope_version": scope.version,
        "knowledge_scope_sha256": _sha256(Path(args.knowledge_scope)),
        "case_count": len(cases),
        "trial_count": None,
        "timeout": verifier.timeout_seconds,
        "retries": verifier.max_retries,
        "max_model_turns": 2,
        "max_tool_calls": 1,
        "initial_top_k": 5,
        "recovery_top_k": 3,
    }
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(_report(metrics, rows), encoding="utf-8")
    print(run_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def claim_support_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def reject_rate(category: str) -> float | None:
        selected = [row for row in rows if row["category"] == category]
        return sum(not row["accepted"] for row in selected) / len(selected) if selected else None

    def accept_rate(category: str) -> float | None:
        selected = [row for row in rows if row["category"] == category]
        return sum(row["accepted"] for row in selected) / len(selected) if selected else None

    fine_grained_total = 0
    fine_grained_correct = 0
    for row in rows:
        # Fabricated-citation rows intentionally have no semantic-verdict gold:
        # their expected behavior is deterministic citation rejection.
        expected = row["expected_verdicts"]
        if not expected:
            continue
        observed_by_index = {
            item["claim_index"]: (
                item["verdict"].value
                if isinstance(item["verdict"], ClaimVerdict)
                else item["verdict"]
            )
            for item in row["claim_results"]
        }
        for claim_index, expected_verdict in enumerate(expected):
            fine_grained_total += 1
            fine_grained_correct += observed_by_index.get(claim_index) == expected_verdict

    return {
        "case_count": len(rows),
        "supported_accept_rate": accept_rate("supported"),
        "unsupported_reject_rate": reject_rate("unsupported"),
        "contradicted_reject_rate": reject_rate("contradicted"),
        "multi_claim_all_supported_accept_rate": accept_rate("multi_claim_all_supported"),
        "multi_claim_one_unsupported_reject_rate": reject_rate("multi_claim_one_unsupported"),
        "fabricated_citation_reject_rate": reject_rate("fabricated_citation"),
        "wrong_citation_binding_reject_rate": reject_rate("wrong_citation_binding"),
        "fine_grained_claim_verdict_accuracy": (
            fine_grained_correct / fine_grained_total if fine_grained_total else None
        ),
        "fine_grained_claim_verdict_correct": fine_grained_correct,
        "fine_grained_claim_verdict_count": fine_grained_total,
        "deterministic_citation_rejections": sum(row["rejection_stage"] == "deterministic_citation" for row in rows),
        "semantic_verifier_rejections": sum(row["rejection_stage"] == "semantic_verifier" for row in rows),
        "verifier_errors": sum(row["rejection_stage"] == "verifier_error" for row in rows),
    }


def _load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"id", "category", "claims", "evidence_source_ids", "expected_verdicts", "status"}
    for case in cases:
        if required - case.keys():
            raise ValueError(f"M3 claim-support case missing fields: {sorted(required - case.keys())}")
    return cases


def _evidence(card: Any) -> Evidence:
    return Evidence(card.id, card.title, card.content, card.source_url, 0.0)


def _report(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# M3 standalone claim-support evaluation",
        "",
        "M3 verifies claims only. It does not call a semantic answer-coverage judge.",
        (
            "Disposition metrics report whether a fixture was accepted or rejected as intended. "
            "`fine_grained_claim_verdict_accuracy` separately compares each semantic "
            "SUPPORTED/UNSUPPORTED/CONTRADICTED verdict; fabricated-citation fixtures are excluded "
            "because they are rejected before semantic verification."
        ),
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Fixture results",
        "",
    ]
    for row in rows:
        verdicts = [item["verdict"] for item in row["claim_results"]]
        lines.append(
            f"- `{row['case_id']}` ({row['category']}): expected={row['expected_verdicts']}, "
            f"observed={verdicts}, accepted={row['accepted']}, stage={row['rejection_stage']}."
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
