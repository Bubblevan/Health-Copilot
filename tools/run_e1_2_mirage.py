"""Execute the preregistered, question-only E1.2 Medical MIRAGE experiment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.e1_2_answer_provider import (
    E1_2AnswerProvider,
    settings_for_candidate,
    verify_served_model,
)
from eval.e1_2_capability_router import (
    JEV_QUESTIONS,
    RetrievalAction,
    route_cheap,
    route_from_jev_probabilities,
)
from eval.e1_2_protocol import assert_committed_artifact
from eval.e1_2_runner import (
    CLEAN_SUBDATASETS,
    case_for_answer,
    file_sha256,
    gold_label,
    is_correct,
    load_partition,
    load_retrieval_results,
    per_case_retrieval_latency,
    read_jsonl,
    result_identity,
    select_generator_pilot,
    summarize_results,
)
from tools.prepare_e1_2_mirage_split import DEFAULT_CASES, DEFAULT_OUTPUT
from tools.prepare_e1_2_retrieval_inputs import materialize
from tools.run_e1_2_retrieval import (
    CORPUS_DEFAULT,
    MODEL_DEFAULT,
    SCRATCH_DEFAULT,
    ensure_storage,
    run_retrieval,
)

CONFIG_PATH = ROOT / "runs/e1_2/frozen_test_config.json"
SELECTION_PATH = ROOT / "runs/e1_2/generator_selection.json"
DEV_SELECTION_LOCK_PATH = ROOT / "runs/e1_2/dev_selection_lock.json"
JEV_MODEL = "jev-1.13.0"
ARM_NAMES = ("closed_book", "rag_bm25", "rag_medcpt", "random_context", "cheap_router", "jev_router")
LOCAL_API_KEY_DEFAULT = Path(r"F:\Health-Copilot-E1.2\llama-api-key.txt")


def load_frozen_config(path: Path = CONFIG_PATH) -> tuple[dict[str, Any], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_ANY_E1_2_ANSWER_CALLS":
        raise ValueError("E1.2 configuration is not frozen")
    return config, file_sha256(path)


def verify_frozen_source_identity(
    config: dict[str, Any], *, cases_path: Path, split_manifest_path: Path
) -> None:
    if file_sha256(cases_path) != config["benchmark"]["normalized_cases_sha256"]:
        raise ValueError("normalized MIRAGE case file differs from the frozen E1.2 identity")
    if file_sha256(split_manifest_path) != config["benchmark"]["split_manifest_sha256"]:
        raise ValueError("MIRAGE split manifest differs from the frozen E1.2 identity")


def write_json_atomic(path: Path, value: Any, *, refuse_overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"refusing to overwrite frozen E1.2 artifact: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary artifact already exists: {temporary}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_selection(path: Path = SELECTION_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("run the matched DEV closed-book model pilot before answer arms")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("selected_candidate") != "QWEN3_LOCAL":
        raise ValueError("generator selection artifact has no valid selected_candidate")
    return value


def create_provider(
    candidate: str, local_api_key_path: Path
) -> E1_2AnswerProvider:
    settings = settings_for_candidate(
        candidate,
        dotenv_path=ROOT / ".env",
        local_api_key_path=local_api_key_path,
    )
    return E1_2AnswerProvider(settings)


def pilot_paths(scratch_root: Path, candidate: str) -> tuple[Path, Path]:
    directory = scratch_root / "runs" / "e1_2" / "generator_pilot"
    return directory / f"{candidate}.jsonl", directory / f"{candidate}.manifest.json"


def run_generator_pilot(
    candidate: str,
    *,
    cases_path: Path,
    split_manifest_path: Path,
    scratch_root: Path,
    local_api_key_path: Path,
    config_sha256: str,
) -> None:
    if candidate != "QWEN3_LOCAL":
        raise ValueError("only the frozen local Qwen candidate may run the DEV pilot")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    dev_cases = load_partition(cases_path, split_manifest, "DEV")
    pilot_cases = select_generator_pilot(dev_cases)
    result_path, manifest_path = pilot_paths(scratch_root, candidate)
    identity = result_identity(
        frozen_config_sha256=config_sha256,
        partition="DEV_PILOT",
        arm="closed_book",
        model=candidate,
    )
    pilot_ids = [str(case["case_id"]) for case in pilot_cases]
    pilot_id_hash = hashlib.sha256("\n".join(pilot_ids).encode()).hexdigest()
    expected_manifest = {
        "status": "RUNNING",
        "candidate": candidate,
        "config_sha256": config_sha256,
        "result_identity": identity,
        "case_count": len(pilot_cases),
        "case_id_list_sha256": pilot_id_hash,
        "prompt_mode": "simple_evidence_injection with empty evidence",
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in expected_manifest.items() if key != "status"):
            raise RuntimeError("existing pilot output uses a different frozen identity")
    else:
        write_json_atomic(manifest_path, expected_manifest)
    existing_rows: dict[str, dict[str, Any]] = {}
    if result_path.exists():
        for row in read_jsonl(result_path):
            if row.get("result_identity") != identity:
                raise RuntimeError("existing pilot rows use a different frozen identity")
            existing_rows[str(row["case_id"])] = row

    provider = create_provider(candidate, local_api_key_path)
    config, _ = load_frozen_config()
    candidate_config = next(
        row for row in config["generator_selection"]["candidates"] if row["id"] == candidate
    )
    if provider.settings.model != candidate_config["requested_model"]:
        raise ValueError(f"configured model for {candidate} differs from the frozen pilot")
    expected_served_model = str(
        candidate_config.get("expected_served_model", candidate_config["requested_model"])
    )
    with result_path.open("a", encoding="utf-8") as handle:
        for index, case in enumerate(pilot_cases, start=1):
            case_id = str(case["case_id"])
            if case_id in existing_rows:
                continue
            answer_case = case_for_answer(case)
            started = time.perf_counter()
            try:
                output = provider.answer(case=answer_case, evidence=[])
                verify_served_model(output, expected_model_id=expected_served_model)
                status = "completed"
                prediction = output["prediction"]
                failure_type = None
            except Exception as exc:  # noqa: BLE001 -- classify without persisting secrets or provider body
                output = {}
                status = "failed"
                prediction = None
                failure_type = type(exc).__name__
            record = {
                "case_id": case_id,
                "subdataset": answer_case["subdataset"],
                "candidate": candidate,
                "result_identity": identity,
                "status": status,
                "prediction": prediction,
                "invalid_answer": bool(output.get("invalid_answer")) if status == "completed" else None,
                "abstained": prediction == "ABSTAIN",
                "is_correct": status == "completed" and is_correct(prediction, gold_label(case)),
                "answer_input_tokens": output.get("answer_input_tokens"),
                "answer_output_tokens": output.get("answer_output_tokens"),
                "answer_latency_ms": output.get("answer_latency_ms", round((time.perf_counter() - started) * 1000)),
                "requested_model": output.get("requested_model"),
                "served_model": output.get("served_model"),
                "failure_type": failure_type,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(pilot_cases):
                print(f"DEV generator pilot {candidate}: {index}/{len(pilot_cases)} cases", flush=True)
    expected_manifest["status"] = "COMPLETED"
    expected_manifest["served_models"] = sorted(
        {row.get("served_model") for row in read_jsonl(result_path) if row.get("served_model")}
    )
    write_json_atomic(manifest_path, expected_manifest)


def select_generator(*, scratch_root: Path, config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    if SELECTION_PATH.exists():
        existing = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        if existing.get("config_sha256") == config_sha256:
            return existing
        raise RuntimeError("generator selection already exists for a different frozen config")

    candidate = "QWEN3_LOCAL"
    result_path, manifest_path = pilot_paths(scratch_root, candidate)
    if not manifest_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"missing completed DEV pilot for {candidate}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED" or manifest.get("config_sha256") != config_sha256:
        raise RuntimeError(f"DEV pilot for {candidate} is incomplete or incompatible")
    rows = read_jsonl(result_path)
    if len(rows) != 90:
        raise ValueError(f"DEV pilot for {candidate} has {len(rows)} rows; expected 90")
    by_case = {str(row["case_id"]): row for row in rows}
    if len(by_case) != len(rows):
        raise ValueError(f"DEV pilot for {candidate} contains duplicate cases")
    candidate_config = next(
        row for row in config["generator_selection"]["candidates"] if row["id"] == candidate
    )
    requested_model = str(candidate_config["requested_model"])
    expected_served_model = str(candidate_config.get("expected_served_model", requested_model))
    served_models = sorted({str(row.get("served_model") or "") for row in rows})
    if served_models != [expected_served_model]:
        raise RuntimeError("local pilot served-model identity is missing or differs from the frozen model")
    if any(row.get("requested_model") != requested_model for row in rows):
        raise RuntimeError("local pilot requested-model identity differs from the frozen model")
    completion = sum(row.get("status") == "completed" for row in rows) / len(rows)
    valid_rate = sum(
        row.get("status") == "completed" and row.get("invalid_answer") is False for row in rows
    ) / max(sum(row.get("status") == "completed" for row in rows), 1)
    if completion < float(config["generator_selection"]["minimum_completion_coverage"]):
        raise RuntimeError("local Qwen DEV pilot completion coverage is below the preregistered minimum")
    if valid_rate < float(config["generator_selection"]["minimum_format_valid_rate"]):
        raise RuntimeError("local Qwen DEV pilot output-format validity is below the preregistered minimum")
    subset_accuracy = {
        subset: sum(
            row.get("is_correct") is True and row.get("subdataset") == subset for row in rows
        ) / sum(row.get("subdataset") == subset for row in rows)
        for subset in CLEAN_SUBDATASETS
    }
    result = {
        "schema_version": "e1-2-generator-selection-v1",
        "config_sha256": config_sha256,
        "selection_uses": "DEV closed-book 90-case pilot for stability/identity only; no model comparison",
        "selected_candidate": candidate,
        "rule": config["generator_selection"]["selection_rule"],
        "pilot": {
            "case_count": len(rows),
            "completed": sum(row.get("status") == "completed" for row in rows),
            "completion_coverage": completion,
            "format_valid_rate_among_completed": valid_rate,
            "subset_accuracy": subset_accuracy,
            "served_models": served_models,
        },
        "case_id_set_sha256": hashlib.sha256("\n".join(sorted(by_case)).encode()).hexdigest(),
    }
    write_json_atomic(SELECTION_PATH, result, refuse_overwrite=True)
    return result


def run_jev_preflight() -> dict[str, Any]:
    from dataclasses import replace

    from health_ai_copilot.routing.jev import JevClient, JevConfig

    base: dict[str, Any] = {
        "requested_model": JEV_MODEL,
        "endpoint": None,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "secret_values_recorded": False,
    }
    try:
        settings = JevConfig.from_env(dotenv_path=ROOT / ".env")
        client = JevClient(replace(settings, model=JEV_MODEL))
        base["endpoint"] = (
            "POST /api/v1/decide" if settings.api_mode == "hosted" else "POST /v1/systemone"
        )
        result = asyncio.run(
            client.evaluate(
                state=(
                    "Synthetic adapter smoke only: would an HTTP 404 troubleshooting question "
                    "benefit from consulting an external technical reference?"
                ),
                questions={
                    "retrieval_likely_to_help": JEV_QUESTIONS["retrieval_likely_to_help"],
                    "requires_specialized_detail": JEV_QUESTIONS["requires_specialized_detail"],
                },
            )
        )
        answer_schema_valid = all(
            name in result.answers
            and isinstance(result.answers[name].get("noul"), (int, float))
            and not isinstance(result.answers[name].get("noul"), bool)
            and 0 <= float(result.answers[name]["noul"]) <= 1
            for name in JEV_QUESTIONS
        )
        base.update(
            {
                "status": "AUTHENTICATED"
                if result.model == JEV_MODEL and answer_schema_valid
                else "BLOCKED_MODEL_OR_SCHEMA_MISMATCH",
                "http_status": 200,
                "api_mode": settings.api_mode,
                "served_model": result.model,
                "answer_schema_valid": answer_schema_valid,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "actual_cost_usd": result.cost_usd,
            }
        )
    except Exception as exc:  # noqa: BLE001 -- never persist request details or secrets
        base.update(
            {
                "status": "BLOCKED_PREFLIGHT",
                "api_mode": settings.api_mode if "settings" in locals() else None,
                "error_type": type(exc).__name__,
                "error": str(exc)[:160] if type(exc).__name__ == "JevAPIError" else None,
            }
        )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / "runs" / "e1_2" / f"jev_preflight_{timestamp}.json"
    write_json_atomic(path, base, refuse_overwrite=True)
    print(json.dumps({"status": base["status"], "artifact": str(path)}))
    return base


def evidence_context_characters(evidence: list[dict[str, Any]]) -> int:
    if not evidence:
        return len("No retrieved evidence was supplied.")
    blocks = []
    for index, row in enumerate(evidence, start=1):
        source_id = str(row.get("id") or f"retrieved-{index}")
        title = str(row.get("title") or "Retrieved excerpt")
        content = str(row.get("content") or row.get("contents") or "")
        blocks.append(f"[source_id={source_id}] {title}\n{content}")
    return sum(map(len, blocks)) + 2 * (len(blocks) - 1)


def random_context_evidence(
    connection: sqlite3.Connection,
    case_id: str,
    target_evidence: list[dict[str, Any]],
    *,
    seed: str,
) -> tuple[list[dict[str, Any]], bool, bool]:
    """Sample without query/qrel inputs and trim authentic text to the BM25 budget."""
    target_count = len(target_evidence)
    target_characters = evidence_context_characters(target_evidence)
    if target_count == 0:
        return [], True, target_characters == evidence_context_characters([])
    count = int(connection.execute("SELECT count(*) FROM chunks").fetchone()[0])
    if count < target_count:
        raise ValueError("random-context corpus has fewer chunks than the BM25 context")
    seed_bytes = hashlib.sha256(f"{seed}\0{case_id}".encode()).digest()
    generator = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    for _ in range(64):
        rowids = generator.sample(range(1, count + 1), target_count)
        placeholders = ",".join("?" for _ in rowids)
        rows = connection.execute(
            f"SELECT rowid,id,title,content FROM chunks WHERE rowid IN ({placeholders})",
            rowids,
        )
        by_id = {
            int(row[0]): {"id": row[1], "title": row[2], "content": row[3]}
            for row in rows
        }
        if len(by_id) != target_count:
            raise RuntimeError("random-context index lookup returned an incomplete sample")
        selected = [by_id[rowid] for rowid in rowids]
        overhead = sum(
            len(
                f"[source_id={row['id']}] "
                f"{row.get('title') or 'Retrieved excerpt'}\n"
            )
            for index, row in enumerate(selected)
        ) + 2 * (target_count - 1)
        target_content_chars = target_characters - overhead
        if target_content_chars < target_count or sum(len(row["content"]) for row in selected) < target_content_chars:
            continue
        allocations = [min(len(row["content"]), max(1, target_content_chars // target_count)) for row in selected]
        remaining = target_content_chars - sum(allocations)
        for index, row in enumerate(selected):
            extra = min(remaining, len(row["content"]) - allocations[index])
            allocations[index] += extra
            remaining -= extra
        if remaining:
            continue
        evidence = [
            {**row, "content": row["content"][:allocations[index]]}
            for index, row in enumerate(selected)
        ]
        return evidence, len(evidence) == target_count, evidence_context_characters(evidence) == target_characters
    selected = [by_id[rowid] for rowid in rowids]
    return selected, len(selected) == target_count, evidence_context_characters(selected) == target_characters


def _load_arm_retrievals(
    *, arm: str, partition: str, scratch_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    base = scratch_root / "retrieval" / partition.lower()
    bm25: dict[str, dict[str, Any]] = {}
    medcpt: dict[str, dict[str, Any]] = {}
    if arm in {"rag_bm25", "random_context", "cheap_router", "jev_router"}:
        bm25 = load_retrieval_results(base / "bm25")
    if arm in {"rag_medcpt", "cheap_router", "jev_router"}:
        medcpt = load_retrieval_results(base / "medcpt")
    return bm25, medcpt


def _jev_client():
    from dataclasses import replace

    from health_ai_copilot.routing.jev import JevClient, JevConfig

    loaded = JevConfig.from_env(dotenv_path=ROOT / ".env")
    return JevClient(replace(loaded, model=JEV_MODEL))


def _jev_preflight_is_authenticated() -> bool:
    files = sorted((ROOT / "runs" / "e1_2").glob("jev_preflight_*.json"), reverse=True)
    if not files:
        return False
    return json.loads(files[0].read_text(encoding="utf-8")).get("status") == "AUTHENTICATED"


def require_committed_dev_selection_lock(
    *, config_sha256: str, selected_candidate: str
) -> dict[str, Any]:
    if not DEV_SELECTION_LOCK_PATH.is_file():
        raise FileNotFoundError("QA TEST requires a committed DEV selection lock")
    lock = json.loads(DEV_SELECTION_LOCK_PATH.read_text(encoding="utf-8"))
    if (
        lock.get("status") != "LOCKED_BEFORE_TEST"
        or lock.get("config_sha256") != config_sha256
        or lock.get("generator_candidate") != selected_candidate
        or lock.get("generator_selection_sha256") != file_sha256(SELECTION_PATH)
        or lock.get("selected_retrieval_cost_reference") not in {"rag_bm25", "rag_medcpt"}
        or lock.get("test_opened") is not False
    ):
        raise ValueError("QA DEV selection lock is incomplete or incompatible")
    assert_committed_artifact(DEV_SELECTION_LOCK_PATH, repo_root=ROOT)
    return lock


def run_arm(
    *,
    partition: str,
    arm: str,
    cases_path: Path,
    split_manifest_path: Path,
    scratch_root: Path,
    local_api_key_path: Path = LOCAL_API_KEY_DEFAULT,
    limit: int | None = None,
) -> None:
    partition_name = partition.upper()
    arm_name = arm.lower()
    if partition_name not in {"DEV", "TEST"}:
        raise ValueError("partition must be DEV or TEST")
    if arm_name not in ARM_NAMES:
        raise ValueError(f"unsupported arm: {arm_name}")
    if partition_name == "TEST" and limit is not None:
        raise ValueError("TEST must run every frozen case; --limit is allowed only for DEV smoke checks")
    if arm_name == "jev_router" and not _jev_preflight_is_authenticated():
        raise RuntimeError("Jev arm is blocked until authenticated decision-API preflight succeeds")

    config, config_sha256 = load_frozen_config()
    selection = load_selection()
    if selection.get("config_sha256") != config_sha256:
        raise ValueError("generator selection was made against another frozen config")
    selected_candidate = str(selection["selected_candidate"])
    if partition_name == "TEST":
        require_committed_dev_selection_lock(
            config_sha256=config_sha256,
            selected_candidate=selected_candidate,
        )
    provider = create_provider(selected_candidate, local_api_key_path)
    candidate_config = next(
        row for row in config["generator_selection"]["candidates"] if row["id"] == selected_candidate
    )
    if provider.settings.model != candidate_config["requested_model"]:
        raise ValueError("selected generator model differs from the frozen config")
    expected_served_model = str(
        candidate_config.get("expected_served_model", candidate_config["requested_model"])
    )
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    cases = load_partition(cases_path, split_manifest, partition_name)
    if partition_name == "DEV" and limit is not None:
        cases = select_generator_pilot(cases)[:limit]
    expected_test = int(config["benchmark"]["clean_test_case_count"])
    if partition_name == "TEST" and len(cases) != expected_test:
        raise ValueError(f"frozen TEST case count changed: {len(cases)} != {expected_test}")

    bm25_rows, medcpt_rows = _load_arm_retrievals(
        arm=arm_name, partition=partition_name, scratch_root=scratch_root
    )
    random_connection: sqlite3.Connection | None = None
    if arm_name == "random_context":
        index_path = scratch_root / "index" / "medrag_textbooks_fts5.sqlite3"
        if not index_path.is_file():
            raise FileNotFoundError(f"random-context sampling requires the BM25 index: {index_path}")
        random_connection = sqlite3.connect(f"{index_path.resolve().as_uri()}?mode=ro", uri=True)
    if arm_name == "jev_router":
        jev = _jev_client()
    else:
        jev = None
    arm_identity = result_identity(
        frozen_config_sha256=config_sha256,
        partition=partition_name,
        arm=arm_name,
        model=selected_candidate,
    )
    output_dir = scratch_root / "runs" / "e1_2" / partition_name.lower() / arm_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "case_results.jsonl"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("result_identity") != arm_identity:
            raise RuntimeError(f"existing {arm_name} output does not match the frozen config")
    elif output_path.exists() and output_path.stat().st_size:
        raise RuntimeError("results file exists without a matching run manifest")
    else:
        write_json_atomic(
            manifest_path,
            {
                "status": "RUNNING",
                "partition": partition_name,
                "arm": arm_name,
                "result_identity": arm_identity,
                "config_sha256": config_sha256,
                "selected_candidate": selected_candidate,
                "expected_cases": len(cases),
            },
        )
    prior_rows = read_jsonl(output_path) if output_path.exists() else []
    completed = set()
    for row in prior_rows:
        if row.get("result_identity") != arm_identity:
            raise RuntimeError("existing case rows have an incompatible identity")
        completed.add(str(row.get("case_id")))

    with output_path.open("a", encoding="utf-8") as handle:
        for index, case in enumerate(cases, start=1):
            case_id = str(case["case_id"])
            if case_id in completed:
                continue
            answer_case = case_for_answer(case)
            question = answer_case["question"]
            decision: dict[str, Any] | None = None
            route_latency_ms = 0
            router_tokens = {"input": 0, "output": 0}
            router_cost_usd = None
            jev_served_model = None
            if arm_name == "closed_book":
                action = RetrievalAction.CLOSED_BOOK
                retrieved = None
                retrieval_latency_ms = 0.0
                router_calls = 0
            elif arm_name == "rag_bm25":
                action = RetrievalAction.RAG_BM25
                retrieved = bm25_rows.get(case_id)
                retrieval_latency_ms = per_case_retrieval_latency(retrieved or {}, "bm25")
                router_calls = 0
            elif arm_name == "rag_medcpt":
                action = RetrievalAction.RAG_MEDCPT
                retrieved = medcpt_rows.get(case_id)
                retrieval_latency_ms = per_case_retrieval_latency(retrieved or {}, "medcpt")
                router_calls = 0
            elif arm_name == "random_context":
                action = RetrievalAction.RANDOM_CONTEXT
                retrieved = bm25_rows.get(case_id)
                router_calls = 0
                random_doc_count_matched = False
                random_char_budget_matched = False
                if retrieved is None:
                    retrieval_latency_ms = 0.0
                else:
                    if random_connection is None:
                        raise RuntimeError("random-context SQLite index was not opened")
                    target_evidence = retrieved.get("retrieved_evidence")
                    target_evidence = target_evidence if isinstance(target_evidence, list) else []
                    sampling_started = time.perf_counter()
                    evidence, random_doc_count_matched, random_char_budget_matched = random_context_evidence(
                        random_connection,
                        case_id,
                        target_evidence,
                        seed=str(config["test_execution"]["random_context_seed"]),
                    )
                    retrieval_latency_ms = (time.perf_counter() - sampling_started) * 1000
                    retrieved = {"retrieved_evidence": evidence}
                decision = {
                    "seed": str(config["test_execution"]["random_context_seed"]),
                    "sampling": "deterministic corpus-row sample; query-independent",
                    "random_doc_count_matched": random_doc_count_matched,
                    "random_char_budget_matched": random_char_budget_matched,
                }
            elif arm_name == "cheap_router":
                route = route_cheap(question)
                action = route.action
                decision = route.to_dict()
                retrieved, retrieval_latency_ms = _selected_retrieval(action, case_id, bm25_rows, medcpt_rows)
                router_calls = 0
            else:
                router_calls = 1
                try:
                    response = asyncio.run(jev.evaluate(state=question, questions=JEV_QUESTIONS))
                    probabilities = {
                        name: float(response.answers[name]["noul"])
                        for name in JEV_QUESTIONS
                    }
                    route = route_from_jev_probabilities(probabilities)
                    action = route.action
                    decision = route.to_dict()
                    route_latency_ms = response.latency_ms
                    router_tokens = {"input": response.input_tokens, "output": response.output_tokens}
                    router_cost_usd = response.cost_usd
                    jev_served_model = response.model
                    retrieved, retrieval_latency_ms = _selected_retrieval(action, case_id, bm25_rows, medcpt_rows)
                except Exception as exc:  # noqa: BLE001 -- use fixed cheap failover and hide provider details
                    route = route_cheap(question, fallback_reason=type(exc).__name__)
                    action = route.action
                    decision = route.to_dict()
                    retrieved, retrieval_latency_ms = _selected_retrieval(action, case_id, bm25_rows, medcpt_rows)

            evidence = []
            retrieval_missing = False
            if action in {
                RetrievalAction.RAG_BM25,
                RetrievalAction.RAG_MEDCPT,
                RetrievalAction.RANDOM_CONTEXT,
            }:
                if retrieved is None:
                    retrieval_missing = True
                else:
                    raw_evidence = retrieved.get("retrieved_evidence")
                    evidence = raw_evidence if isinstance(raw_evidence, list) else []
            context_characters = evidence_context_characters(evidence)
            started = time.perf_counter()
            if retrieval_missing:
                output = {}
                status = "failed"
                prediction = None
                failure_type = "missing_retrieval_artifact"
            else:
                try:
                    output = provider.answer(case=answer_case, evidence=evidence)
                    verify_served_model(output, expected_model_id=expected_served_model)
                    status = "completed"
                    prediction = output["prediction"]
                    failure_type = None
                except Exception as exc:  # noqa: BLE001 -- persist class only; no secret-bearing error text
                    output = {}
                    status = "failed"
                    prediction = None
                    failure_type = type(exc).__name__
            answer_wall_ms = output.get("answer_latency_ms", round((time.perf_counter() - started) * 1000))
            record = {
                "case_id": case_id,
                "subdataset": answer_case["subdataset"],
                "partition": partition_name,
                "arm": arm_name,
                "result_identity": arm_identity,
                "status": status,
                "prediction": prediction,
                "is_correct": status == "completed" and is_correct(prediction, gold_label(case)),
                "invalid_answer": bool(output.get("invalid_answer")) if status == "completed" else None,
                "abstained": prediction == "ABSTAIN",
                "failure_type": failure_type,
                "selected_action": action.value,
                "route_decision": decision,
                "route_latency_ms": route_latency_ms,
                "router_calls": router_calls,
                "router_input_tokens": router_tokens["input"],
                "router_output_tokens": router_tokens["output"],
                "router_cost_usd": router_cost_usd,
                "jev_served_model": jev_served_model,
                "retrieval_calls": int(action is not RetrievalAction.CLOSED_BOOK),
                "retrieved_chunks": len(evidence),
                "retrieval_latency_ms": retrieval_latency_ms,
                "context_characters": context_characters,
                "answer_input_tokens": output.get("answer_input_tokens"),
                "answer_output_tokens": output.get("answer_output_tokens"),
                "answer_latency_ms": answer_wall_ms,
                "component_latency_proxy_ms": route_latency_ms + retrieval_latency_ms + answer_wall_ms,
                "random_context_doc_count_matched": (
                    random_doc_count_matched if arm_name == "random_context" else None
                ),
                "random_context_char_budget_matched": (
                    random_char_budget_matched if arm_name == "random_context" else None
                ),
                "requested_model": output.get("requested_model", provider.settings.model),
                "served_model": output.get("served_model"),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if index % 50 == 0 or index == len(cases):
                print(f"{partition_name} {arm_name}: {index}/{len(cases)} cases", flush=True)

    if random_connection is not None:
        random_connection.close()
    rows = read_jsonl(output_path)
    metrics = summarize_results(rows)
    write_json_atomic(output_dir / "metrics.json", metrics)
    write_json_atomic(
        manifest_path,
        {
            "status": "COMPLETED",
            "partition": partition_name,
            "arm": arm_name,
            "result_identity": arm_identity,
            "config_sha256": config_sha256,
            "selected_candidate": selected_candidate,
            "case_count": len(rows),
            "failures": sum(row.get("status") != "completed" for row in rows),
        },
    )
    print(f"{partition_name} {arm_name} completed: {len(rows)} cases; metrics remain unopened", flush=True)


def _selected_retrieval(
    action: RetrievalAction,
    case_id: str,
    bm25_rows: dict[str, dict[str, Any]],
    medcpt_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, float]:
    if action is RetrievalAction.CLOSED_BOOK:
        return None, 0.0
    if action is RetrievalAction.RAG_BM25:
        row = bm25_rows.get(case_id)
        return row, per_case_retrieval_latency(row or {}, "bm25")
    row = medcpt_rows.get(case_id)
    return row, per_case_retrieval_latency(row or {}, "medcpt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("pilot", "select-generator", "prepare-inputs", "retrieve", "jev-preflight", "run-arm"),
        required=True,
    )
    parser.add_argument("--candidate", choices=("QWEN3_LOCAL",))
    parser.add_argument("--partition", choices=("DEV", "TEST"))
    parser.add_argument("--arm", choices=ARM_NAMES)
    parser.add_argument("--retriever", choices=("bm25", "medcpt"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch-root", type=Path, default=SCRATCH_DEFAULT)
    parser.add_argument("--llama-api-key-file", type=Path, default=LOCAL_API_KEY_DEFAULT)
    args = parser.parse_args()
    config, config_sha256 = load_frozen_config()
    verify_frozen_source_identity(
        config,
        cases_path=args.cases,
        split_manifest_path=args.split_manifest,
    )
    ensure_storage(args.scratch_root, reserve_gib=1.0)

    if args.phase == "pilot":
        if not args.candidate:
            parser.error("--candidate is required for pilot")
        run_generator_pilot(
            args.candidate,
            cases_path=args.cases,
            split_manifest_path=args.split_manifest,
            scratch_root=args.scratch_root,
            local_api_key_path=args.llama_api_key_file,
            config_sha256=config_sha256,
        )
    elif args.phase == "select-generator":
        selection = select_generator(scratch_root=args.scratch_root, config=config, config_sha256=config_sha256)
        print(json.dumps({"selected_candidate": selection["selected_candidate"], "config_sha256": config_sha256}))
    elif args.phase == "prepare-inputs":
        if not args.partition:
            parser.error("--partition is required for prepare-inputs")
        output = args.scratch_root / "inputs" / f"benchmark_{args.partition.lower()}.json"
        result = materialize(
            cases_path=args.cases,
            split_manifest_path=args.split_manifest,
            split=args.partition,
            output_path=output,
        )
        print(json.dumps(result, ensure_ascii=False))
    elif args.phase == "retrieve":
        if not args.partition or not args.retriever:
            parser.error("--partition and --retriever are required for retrieve")
        run_retrieval(
            retriever=args.retriever,
            split=args.partition,
            scratch_root=args.scratch_root,
            corpus_dir=CORPUS_DEFAULT,
            model_root=MODEL_DEFAULT,
        )
    elif args.phase == "jev-preflight":
        run_jev_preflight()
    else:
        if not args.partition or not args.arm:
            parser.error("--partition and --arm are required for run-arm")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        run_arm(
            partition=args.partition,
            arm=args.arm,
            cases_path=args.cases,
            split_manifest_path=args.split_manifest,
            scratch_root=args.scratch_root,
            local_api_key_path=args.llama_api_key_file,
            limit=args.limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
