"""E0-only ``health-bench`` command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import adapter_for
from .audit import (
    audit_manifest,
    audit_normalized_dataset,
    audit_research_pack,
    data_root,
    load_research_pack,
    write_closeout,
)
from .contracts import canonical_json
from .fetch import FetchError, collect_source_evidence, fetch_benchmark
from .registry import BenchmarkRegistryError, default_benchmark_registry, repository_root
from .review import ReviewDecisionError, apply_review, freeze_research_pack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="health-bench",
        description="Prepare and audit E0 benchmark identity; model scoring is intentionally absent",
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list registered benchmarks without network calls")
    inspect = sub.add_parser("inspect", help="inspect one manifest without raw content")
    inspect.add_argument("benchmark")
    prepare = sub.add_parser("prepare", help="create explicit local cache directories; never downloads")
    prepare.add_argument("benchmark")
    fetch = sub.add_parser("fetch", help="explicitly fetch the pinned raw artifact")
    fetch.add_argument("benchmark")
    source_review = sub.add_parser("review-source", help="collect canonical source evidence and a proposed pin")
    source_review.add_argument("benchmark")
    for name, help_text in (
        ("verify", "verify local raw and normalized identity"),
        ("normalize", "normalize an already-present local raw artifact"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("benchmark")
    audit = sub.add_parser("audit", help="run deterministic fail-closed audits")
    audit.add_argument("benchmark", nargs="?", default=None)
    profile = sub.add_parser("profile", help="show research task profiles only")
    profile.add_argument("benchmark", nargs="?", default="research-architecture-v1")
    profile.add_argument("--case-id")
    export = sub.add_parser("review-export", help="export a deterministic human-review packet")
    export.add_argument("benchmark", nargs="?", default="research-architecture-v1")
    export.add_argument("--output", type=Path)
    freeze = sub.add_parser("freeze", help="deliberately freeze reviewed internal research data")
    freeze.add_argument("benchmark", nargs="?", default="research-architecture-v1")
    freeze.add_argument("--pack-root", type=Path)
    apply = sub.add_parser("apply-review", help="apply explicit per-case human review decisions")
    apply.add_argument("benchmark")
    apply.add_argument("decisions", type=Path)
    apply.add_argument("--output-root", type=Path, required=True)
    closeout = sub.add_parser("closeout", help="write the metadata-only E0 stage-gate report")
    closeout.add_argument("--functional-code-sha", required=True)
    closeout.add_argument("--final-documentation-sha")
    return parser


def _paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = (args.repo_root or repository_root()).resolve()
    benchmark_data = (args.data_root or data_root(root)).resolve()
    return root, benchmark_data


def _registry(args: argparse.Namespace):
    root, _ = _paths(args)
    return root, default_benchmark_registry()


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            return _list(args)
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "fetch":
            return _fetch(args)
        if args.command == "review-source":
            return _review_source(args)
        if args.command == "verify":
            return _verify(args)
        if args.command == "normalize":
            return _normalize(args)
        if args.command == "audit":
            return _audit(args)
        if args.command == "profile":
            return _profile(args)
        if args.command == "review-export":
            return _review_export(args)
        if args.command == "freeze":
            return _freeze(args)
        if args.command == "apply-review":
            return _apply_review(args)
        if args.command == "closeout":
            return _closeout(args)
    except (BenchmarkRegistryError, FetchError, ReviewDecisionError, OSError, ValueError, KeyError) as exc:
        print(f"health-bench: {exc}", file=sys.stderr)
        return 2
    return 2


def _list(args: argparse.Namespace) -> int:
    _, benchmark_registry = _registry(args)
    _, benchmark_data = _paths(args)
    rows = []
    for manifest in benchmark_registry.list():
        raw_root = benchmark_data / "raw" / manifest.benchmark_id
        normalized_root = benchmark_data / "normalized" / manifest.benchmark_id
        rows.append(
            {
                "benchmark_id": manifest.benchmark_id,
                "family": manifest.benchmark_family,
                "source": manifest.source_name,
                "version": manifest.version,
                "upstream_revision": manifest.upstream_revision,
                "admissibility_status": manifest.status.value,
                "raw_data_available": raw_root.exists() and any(raw_root.rglob("*")),
                "normalized_data_available": (normalized_root / "cases.jsonl").exists(),
                "license_review_status": manifest.license.license_review_status.value,
                "judge_required": manifest.judge_protocol is not None,
            }
        )
    _print(rows)
    return 0


def _inspect(args: argparse.Namespace) -> int:
    _, benchmark_registry = _registry(args)
    _print(benchmark_registry.inspect(args.benchmark))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    _, benchmark_registry = _registry(args)
    _, benchmark_data = _paths(args)
    manifest = benchmark_registry.get(args.benchmark)
    raw_root = benchmark_data / "raw" / manifest.benchmark_id
    normalized_root = benchmark_data / "normalized" / manifest.benchmark_id
    cache_root = benchmark_data / "cache" / manifest.benchmark_id
    for path in (raw_root, normalized_root, cache_root):
        path.mkdir(parents=True, exist_ok=True)
    _print(
        {
            "benchmark_id": manifest.benchmark_id,
            "raw_root": str(raw_root),
            "normalized_root": str(normalized_root),
            "cache_root": str(cache_root),
            "network_called": False,
            "next_step": "place a reviewed raw artifact under raw_root, then run verify and normalize",
        }
    )
    return 0


def _fetch(args: argparse.Namespace) -> int:
    _, benchmark_registry = _registry(args)
    _, benchmark_data = _paths(args)
    manifest = benchmark_registry.get(args.benchmark)
    raw_root = fetch_benchmark(manifest, benchmark_data)
    _print(
        {
            "benchmark_id": manifest.benchmark_id,
            "raw_root": str(raw_root),
            "manifest_changed": False,
            "network_called": True,
            "next_step": "run review-source, explicitly pin the reviewed manifest SHA, then normalize and verify",
        }
    )
    return 0


def _review_source(args: argparse.Namespace) -> int:
    root, benchmark_registry = _registry(args)
    manifest = benchmark_registry.get(args.benchmark)
    output = collect_source_evidence(manifest, root)
    print(output)
    return 0


def _verify(args: argparse.Namespace) -> int:
    _, benchmark_registry = _registry(args)
    _, benchmark_data = _paths(args)
    manifest = benchmark_registry.get(args.benchmark)
    raw_root = benchmark_data / "raw" / manifest.benchmark_id
    normalized_root = benchmark_data / "normalized" / manifest.benchmark_id
    errors: list[str] = []
    manifest_audit = audit_manifest(manifest)
    pin_errors = [error for error in manifest_audit.errors if error != "unknown raw hash"]
    errors.extend(pin_errors)
    if not raw_root.exists() or not any(raw_root.rglob("*")):
        errors.append("raw data unavailable")
    else:
        raw_identity = json.loads((raw_root / "raw_identity.json").read_text(encoding="utf-8"))
        by_name = {item.get("name"): item for item in raw_identity.get("artifacts", [])}
        mismatches = [
            item.name
            for item in manifest.raw_artifacts
            if item.sha256 is not None and by_name.get(item.name, {}).get("sha256") != item.sha256
        ]
        if mismatches:
            errors.append(f"raw SHA mismatch: {', '.join(sorted(mismatches))}")
        inspection = adapter_for(manifest.benchmark_id).inspect(raw_root)
        _print(
            {
                "benchmark_id": manifest.benchmark_id,
                "inspection": inspection,
                "manifest_sha_pinned": all(item.sha256 is not None for item in manifest.raw_artifacts),
                "errors": errors,
            }
        )
    if normalized_root.exists():
        errors.extend(
            error
            for error in audit_normalized_dataset(manifest, normalized_root).errors
            if error != "unknown raw hash"
        )
    if errors:
        if raw_root.exists() and any(raw_root.rglob("*")):
            _print({"benchmark_id": manifest.benchmark_id, "errors": sorted(set(errors))})
        return 2
    return 0


def _normalize(args: argparse.Namespace) -> int:
    _, benchmark_registry = _registry(args)
    _, benchmark_data = _paths(args)
    manifest = benchmark_registry.get(args.benchmark)
    raw_root = benchmark_data / "raw" / manifest.benchmark_id
    if not raw_root.exists() or not any(raw_root.rglob("*")):
        raise FileNotFoundError(f"raw data unavailable: {raw_root}")
    identity = adapter_for(manifest.benchmark_id).normalize(
        raw_root, benchmark_data / "normalized" / manifest.benchmark_id
    )
    _print(identity.to_dict())
    return 0


def _audit(args: argparse.Namespace) -> int:
    root, benchmark_registry = _registry(args)
    _, benchmark_data = _paths(args)
    reports = []
    if args.benchmark == "research-architecture-v1":
        selected = []
    elif args.benchmark:
        selected = [args.benchmark]
    else:
        selected = [
            item.benchmark_id
            for item in benchmark_registry.list()
            if item.benchmark_id != "research-architecture-v1"
        ]
    for benchmark_id in selected:
        manifest = benchmark_registry.get(benchmark_id)
        report = audit_manifest(manifest)
        normalized_root = benchmark_data / "normalized" / manifest.benchmark_id
        if normalized_root.exists():
            report = audit_normalized_dataset(manifest, normalized_root)
        reports.append(report.to_dict())
    research_report = audit_research_pack(root / "benchmarks" / "research_architecture_v1")
    if args.benchmark in {None, "research-architecture-v1"}:
        reports.append(research_report.to_dict())
    _print(reports)
    return 0 if all(item["ok"] for item in reports) else 2


def _profile(args: argparse.Namespace) -> int:
    root, _ = _paths(args)
    cases, profiles, _ = load_research_pack(root / "benchmarks" / "research_architecture_v1")
    case_by_id = {row["case_id"]: row for row in cases}
    rows = []
    for profile in profiles:
        if args.case_id and profile.case_id != args.case_id:
            continue
        rows.append(
            {
                "case_id": profile.case_id,
                "question": case_by_id[profile.case_id].get("payload", {}).get("question"),
                "task_profile": profile.to_dict(),
            }
        )
    _print(rows)
    return 0


def _review_export(args: argparse.Namespace) -> int:
    root, _ = _paths(args)
    cases, profiles, annotation = load_research_pack(root / "benchmarks" / "research_architecture_v1")
    case_by_id = {row["case_id"]: row for row in cases}
    flags_by_case = {
        item.get("case_id"): item for item in annotation.get("review_flags", []) if item.get("case_id")
    }
    cards = {}
    cards_root = root / "data" / "knowledge_cards"
    for card_path in sorted(cards_root.glob("*.json")):
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if isinstance(card, dict) and card.get("id"):
            cards[card["id"]] = card
    lines = [
        "# research-architecture-v1 review export",
        "",
        f"Status: `{annotation.get('status')}`",
        "",
        "Human reviewer must verify wording, evidence groups, source families, dependency annotations, OOD rationale, safety boundary and absence of private information.",
        "The source blocks below are copied from the reviewed local KnowledgeCard records; a missing source ID is an explicit review blocker, not an inferred correction.",
        "",
    ]
    for profile in profiles:
        case = case_by_id[profile.case_id]
        lines.extend(
            [
                f"## {profile.case_id}",
                "",
                f"Question: {case.get('payload', {}).get('question', '')}",
                "",
                "```json",
                canonical_json(
                    {
                        "gold": case.get("gold", {}),
                        "required_evidence_groups": profile.required_evidence_groups,
                        "source_families": profile.source_families,
                        "task_profile": profile.to_dict(),
                        "source_review_flags": [
                            *case.get("metadata", {}).get("review_flags", []),
                            flags_by_case.get(profile.case_id),
                        ]
                        if flags_by_case.get(profile.case_id)
                        else case.get("metadata", {}).get("review_flags", []),
                        "sources": [
                            {
                                "source_id": source_id,
                                "title": cards[source_id].get("title"),
                                "publisher": cards[source_id].get("publisher"),
                                "content_excerpt": cards[source_id].get("content"),
                                "source_url": cards[source_id].get("source_url"),
                                "published_at": cards[source_id].get("published_at"),
                                "reviewed_at": cards[source_id].get("reviewed_at"),
                            }
                            if source_id in cards
                            else {"source_id": source_id, "status": "UNRESOLVED_SOURCE_REFERENCE"}
                            for group in profile.required_evidence_groups
                            for source_id in group
                        ],
                        "review_checklist": {
                            "wording_matches_evidence": None,
                            "substitutes_not_missing": None,
                            "architecture_neutral": None,
                            "no_diagnosis_or_prescription_drift": None,
                            "no_private_patient_information": None,
                            "all_referenced_sources_resolved": None,
                        },
                    }
                ),
                "```",
                "",
            ]
        )
    output = args.output or root / "runs" / "e0" / "research_architecture_v1_review.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)
    return 0


def _freeze(args: argparse.Namespace) -> int:
    root, _ = _paths(args)
    pack_root = args.pack_root or root / "benchmarks" / "research_architecture_v1"
    result = freeze_research_pack(pack_root)
    _print({"benchmark_id": args.benchmark, **result})
    return 0


def _apply_review(args: argparse.Namespace) -> int:
    root, _ = _paths(args)
    if args.benchmark != "research-architecture-v1":
        raise ReviewDecisionError("apply-review is currently defined only for research-architecture-v1")
    result = apply_review(
        root / "benchmarks" / "research_architecture_v1",
        args.decisions,
        args.output_root,
    )
    _print(result)
    return 0


def _closeout(args: argparse.Namespace) -> int:
    root, benchmark_registry = _registry(args)
    output = write_closeout(
        benchmark_registry,
        root,
        args.functional_code_sha,
        args.final_documentation_sha,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
