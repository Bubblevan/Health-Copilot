"""Gold-blind replay of the frozen Adaptive advanced-team recruitment failures."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from health_ai_copilot.harness.budget import BudgetLedger, BudgetLimits
from health_ai_copilot.harness.runtime import _BudgetedModelProvider
from health_ai_copilot.harness.trace import ExecutionTrace, query_fingerprint
from health_ai_copilot.multi_agent.mdagents_style import (
    MDAgentsStyleConfig,
    _format_question,
    _parse_teams,
    _team_recruitment_schema,
)
from health_ai_copilot.providers.model import ModelRequest, VllmModelProvider

DEFAULT_ROOT = ROOT / "runs/common_eval/harness-v1-base-20261005/adaptive-failure-audit-20261005"
RECRUITMENT_SYSTEM = (
    "Create two independent medical expert teams with different complementary perspectives. "
    "Each team has 2 or 3 distinct specialists, each with a narrow focus. "
    'Return JSON only: {"teams":[{"name":"...",'
    '"specialists":[{"name":"...","focus":"..."}]}]}'
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _digest_ids(case_ids: list[str]) -> str:
    return sha256(("\n".join(case_ids) + "\n").encode()).hexdigest()


def _load_cases(candidate_view: Path, id_manifest: Path) -> tuple[list[str], dict[str, str]]:
    selection = json.loads(id_manifest.read_text(encoding="utf-8"))
    case_ids = selection["case_ids"]
    if selection.get("case_count") != 219 or len(case_ids) != 219:
        raise ValueError("the frozen recruitment replay must contain exactly 219 IDs")
    if _digest_ids(case_ids) != selection.get("case_ids_sha256"):
        raise ValueError("frozen recruitment replay ID hash mismatch")
    candidate_by_id = {}
    for row in _read_jsonl(candidate_view):
        case_id = row.get("id") or row.get("case_id")
        prompt = row.get("prompt") or row.get("query")
        if isinstance(case_id, str) and isinstance(prompt, str):
            candidate_by_id[case_id] = prompt
    missing = [case_id for case_id in case_ids if case_id not in candidate_by_id]
    if missing:
        raise ValueError(f"candidate view is missing {len(missing)} frozen case IDs")
    return case_ids, candidate_by_id


async def _recruit(
    *, case_id: str, query: str, provider: VllmModelProvider,
    config: MDAgentsStyleConfig,
) -> dict[str, Any]:
    question = _format_question(
        query,
        (),
        ((
            "Harness answer schema: single_choice. Follow this output shape; "
            "do not add options outside that schema."
        ),),
    )
    request = ModelRequest(
        model=config.model_name,
        messages=(
            {"role": "system", "content": RECRUITMENT_SYSTEM},
            {"role": "user", "content": f"Question:\n{question}"},
        ),
        temperature=0.0,
        max_output_tokens=320,
        timeout_seconds=config.provider_timeout_seconds,
        json_mode=True,
        json_schema=_team_recruitment_schema(
            max_teams=config.max_advanced_teams,
            max_specialists=config.max_specialists_per_team,
        ),
    )
    budgeted = _BudgetedModelProvider(
        provider,
        BudgetLedger(BudgetLimits(deadline_ms=120_000)),
        ExecutionTrace(query_sha256=query_fingerprint(query), profile_id="B2-recruitment-debug"),
    )
    reply = await budgeted.complete(request)
    output: dict[str, Any] = {
        "case_id": case_id,
        "status": "complete",
        "raw_recruitment_output": reply.content,
        "raw_output_sha256": sha256(reply.content.encode()).hexdigest(),
        "finish_reason": reply.finish_reason,
        "input_tokens": reply.input_tokens,
        "output_tokens": reply.output_tokens,
        "latency_ms": round(reply.latency_ms, 3),
    }
    try:
        teams = _parse_teams(
            reply.content,
            max_teams=config.max_advanced_teams,
            max_specialists=config.max_specialists_per_team,
        )
    except (TypeError, ValueError) as exc:
        output.update({
            "validation_status": "invalid",
            "validation_error": f"{type(exc).__name__}:{exc}",
        })
    else:
        output.update({
            "validation_status": "valid",
            "team_count": len(teams),
            "specialist_count": sum(len(members) for _, members in teams),
        })
    return output


async def run(args: argparse.Namespace) -> dict[str, Any]:
    case_ids, candidate_by_id = _load_cases(args.candidate_view, args.id_manifest)
    config_data = json.loads(args.model_config.read_text(encoding="utf-8"))
    serving = config_data["serving"]
    runtime_data = json.loads(args.vllm_runtime_config.read_text(encoding="utf-8"))
    configured_structured = serving.get("structured_outputs_config")
    if (
        configured_structured is not None
        and runtime_data.get("structured_outputs_config") != configured_structured
    ):
        raise ValueError("vLLM structured_outputs_config differs from the frozen model config")
    model = os.environ.get(serving["served_model_env"], serving["served_model"])
    base_url = os.environ.get(serving["base_url_env"], serving["base_url"])
    provider = VllmModelProvider(
        base_url=base_url,
        model=model,
        api_key=os.environ.get("VLLM_API_KEY", "local-vllm"),
        default_temperature=float(serving["temperature"]),
        default_top_p=float(serving["top_p"]),
        max_output_tokens=int(serving["max_output_tokens"]),
        chat_template_kwargs=serving.get("chat_template_kwargs"),
    )
    config = MDAgentsStyleConfig(
        model_name=model,
        max_output_tokens=int(serving["max_output_tokens"]),
        capture_validation_outputs=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "recruitment.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    expected_identity = {
        "schema_version": "adaptive-recruitment-replay-v1",
        "purpose": "diagnose only the previously failing advanced team-recruitment stage",
        "model_config_sha256": _digest_file(args.model_config),
        "candidate_view_sha256": _digest_file(args.candidate_view),
        "failure_id_manifest_sha256": _digest_file(args.id_manifest),
        "case_count": len(case_ids),
        "case_ids_sha256": _digest_ids(case_ids),
        "served_model": model,
        "base_url": base_url,
        "temperature": serving["temperature"],
        "top_p": serving["top_p"],
        "max_output_tokens": 320,
        "thinking": serving.get("chat_template_kwargs", {}).get("enable_thinking", False),
        "vllm_runtime_config_sha256": _digest_file(args.vllm_runtime_config),
        "structured_outputs_config": runtime_data.get("structured_outputs_config", {}),
        "seed": serving.get("seed"),
        "concurrency": 1,
        "gold_visible": False,
        "full_reasoning_replayed": False,
        "replay_scope": "same advanced recruiter prompt/schema; classifier and downstream MDT stages skipped",
    }
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"output exists; pass --resume to continue: {args.output_dir}")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected_identity.items():
            if previous.get(key) != value:
                raise ValueError(f"resume refused: replay identity changed at {key}")
    else:
        manifest_path.write_text(json.dumps({
            **expected_identity,
            "status": "RUNNING",
            "started_at_utc": datetime.now(UTC).isoformat(),
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    existing = _read_jsonl(rows_path) if rows_path.exists() else []
    by_id = {str(row["case_id"]): row for row in existing}
    unknown = set(by_id) - set(case_ids)
    if unknown:
        raise ValueError("resume output contains case IDs outside the frozen selection")
    with rows_path.open("a", encoding="utf-8") as sink:
        for case_id in case_ids:
            if by_id.get(case_id, {}).get("status") == "complete":
                continue
            try:
                row = await _recruit(
                    case_id=case_id,
                    query=candidate_by_id[case_id],
                    provider=provider,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001 - persist per-case infrastructure errors
                row = {
                    "case_id": case_id,
                    "status": "provider_error",
                    "error": f"{type(exc).__name__}:{str(exc)[:240]}",
                }
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            by_id[case_id] = row
            if len(by_id) % 10 == 0:
                print(f"completed {len(by_id)}/{len(case_ids)} recruiter replays", flush=True)

    counts = Counter(
        f"{row.get('status')}:{row.get('validation_status', 'n/a')}"
        for row in by_id.values()
    )
    summary = {
        **expected_identity,
        "status": "COMPLETE" if len(by_id) == len(case_ids) else "INCOMPLETE",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status_counts": dict(sorted(counts.items())),
        "valid_team_schemas": sum(row.get("validation_status") == "valid" for row in by_id.values()),
        "invalid_team_schemas": sum(row.get("validation_status") == "invalid" for row in by_id.values()),
        "provider_errors": sum(row.get("status") == "provider_error" for row in by_id.values()),
        "recruitment_jsonl_sha256": _digest_file(rows_path),
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-view", type=Path, required=True)
    parser.add_argument("--id-manifest", type=Path, default=DEFAULT_ROOT / "team-recruitment-failures.json")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/models/qwen3_8b_base_system_eval.json")
    parser.add_argument("--vllm-runtime-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
