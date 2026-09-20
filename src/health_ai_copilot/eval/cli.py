"""Command line interface for the explicit M7 evaluation system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .registry import default_eval_suite_registry
from .schema import EvalConfigurationError, EvalExecutionMode
from .system import EvaluationRunner


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            return _list()
        if args.command == "run":
            return _run(args)
        if args.command == "compare":
            return _compare(args)
    except (EvalConfigurationError, OSError, ValueError, KeyError) as exc:
        print(f"health-eval: {exc}", file=sys.stderr)
        return 2
    parser.error("unknown command")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="health-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list source-registered historical suites")
    run = sub.add_parser("run", help="run one registered evaluation suite")
    run.add_argument("--suite", required=True)
    run.add_argument("--execution", choices=[item.value for item in EvalExecutionMode], required=True)
    run.add_argument("--profile")
    run.add_argument("--trials", type=int, default=1)
    run.add_argument("--output-root", default="runs/m7")
    run.add_argument("--run-dir", help="recorded M4-compatible directory for replay")
    run.add_argument("--allow-live-provider", action="store_true")
    run.add_argument("--public-eval-content", action="store_true")
    run.add_argument("--knowledge-dir", default="data/knowledge_cards")
    run.add_argument("--knowledge-scope", default="data/knowledge_scope.json")
    compare = sub.add_parser("compare", help="compare compatible completed bundles")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    return parser


def _list() -> int:
    rows = []
    for suite in default_eval_suite_registry().list():
        rows.append(
            {
                "suite_id": suite.suite_id,
                "version": suite.version,
                "target_kind": suite.target_kind.value,
                "dataset_path": suite.dataset_path,
                "execution_modes": [item.value for item in suite.supported_execution_modes],
                "default_profile_id": suite.default_profile_id,
                "grader_ids": list(suite.grader_ids),
                "metric_definition_version": suite.metric_definition_version,
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run(args) -> int:
    runner = EvaluationRunner()
    spec = runner.prepare_run_spec(
        args.suite,
        execution_mode=args.execution,
        profile_id=args.profile,
        trials=args.trials,
        output_root=args.output_root,
        replay_run_dir=args.run_dir,
    )
    path = runner.run(
        spec,
        allow_live_provider=args.allow_live_provider,
        public_eval_content=args.public_eval_content,
        knowledge_dir=args.knowledge_dir,
        knowledge_scope=args.knowledge_scope,
    )
    print(path)
    return 0


def _compare(args) -> int:
    left = _read_manifest(args.left)
    right = _read_manifest(args.right)
    for key in ("suite_id", "suite_version", "dataset_sha256", "metric_definition_version"):
        if left.get(key) != right.get(key):
            raise EvalConfigurationError(f"incompatible bundles: {key} differs")
    if left.get("suite_id") == "m8-agent-team-focused-v1":
        allowed = set(
            default_eval_suite_registry()
            .get("m8-agent-team-focused-v1")
            .provenance.get("allowed_profiles", ())
        )
        if left.get("profile_id") not in allowed or right.get("profile_id") not in allowed:
            raise EvalConfigurationError(
                "M8 comparison only permits the workflow, single-agent, and team profiles"
            )
    left_metrics = json.loads((args.left / "metrics.json").read_text(encoding="utf-8"))
    right_metrics = json.loads((args.right / "metrics.json").read_text(encoding="utf-8"))
    delta = {}
    for metric_id in sorted(set(left_metrics) | set(right_metrics)):
        left_value = left_metrics.get(metric_id, {}).get("value")
        right_value = right_metrics.get(metric_id, {}).get("value")
        delta[metric_id] = {
            "left": left_value,
            "right": right_value,
            "delta": right_value - left_value
            if left_value is not None and right_value is not None
            else None,
        }
    print(json.dumps({"left": left.get("profile_id"), "right": right.get("profile_id"), "metrics": delta}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _read_manifest(path: Path) -> dict:
    return json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
