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
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.e1_2_answer_provider import E1_2AnswerProvider, settings_for_candidate
from eval.e1_2_capability_router import (
    JEV_QUESTIONS,
    RetrievalAction,
    route_cheap,
    route_from_jev_probabilities,
)
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
JEV_MODEL = "jev-1.13.0"
ARM_NAMES = ("closed_book", "rag_bm25", "rag_medcpt", "cheap_router", "jev_router")


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
    if value.get("selected_candidate") not in {"QWEN3_LOCAL", "CONFIGURED_API_MODEL"}:
        raise ValueError("generator selection artifact has no valid selected_candidate")
    return value


def create_provider(candidate: str, scratch_root: Path) -> E1_2AnswerProvider:
    settings = settings_for_candidate(
        candidate,
        dotenv_path=ROOT / ".env",
        local_api_key_path=scratch_root / "llama-api-key.txt",
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
    config_sha256: str,
) -> None:
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

    provider = create_provider(candidate, scratch_root)
    config, _ = load_frozen_config()
    candidate_config = next(
        row for row in config["generator_selection"]["candidates"] if row["id"] == candidate
    )
    if provider.settings.model != candidate_config["requested_model"]:
        raise ValueError(f"configured model for {candidate} differs from the frozen pilot")
    with result_path.open("a", encoding="utf-8") as handle:
        for index, case in enumerate(pilot_cases, start=1):
            case_id = str(case["case_id"])
            if case_id in existing_rows:
                continue
            answer_case = case_for_answer(case)
            started = time.perf_counter()
            try:
                output = provider.answer(case=answer_case, evidence=[])
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
                "invalid_answer": bool(output.get("invalid_answer", True)),
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

    candidates = ("QWEN3_LOCAL", "CONFIGURED_API_MODEL")
    all_rows: dict[str, dict[str, dict[str, Any]]] = {}
    case_sets: list[set[str]] = []
    per_candidate: dict[str, Any] = {}
    for candidate in candidates:
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
        case_sets.append(set(by_case))
        all_rows[candidate] = by_case
        subset_accuracy: dict[str, float] = {}
        subset_coverage: dict[str, float] = {}
        for subset in CLEAN_SUBDATASETS:
            subset_rows = [row for row in rows if row.get("subdataset") == subset]
            subset_accuracy[subset] = sum(row.get("is_correct") is True for row in subset_rows) / len(subset_rows)
            subset_coverage[subset] = sum(row.get("status") == "completed" for row in subset_rows) / len(subset_rows)
        per_candidate[candidate] = {
            "macro_accuracy": sum(subset_accuracy.values()) / len(subset_accuracy),
            "subset_accuracy": subset_accuracy,
            "subset_completion_coverage": subset_coverage,
            "completed_case_count": sum(row.get("status") == "completed" for row in rows),
            "served_models": sorted({row.get("served_model") for row in rows if row.get("served_model")}),
        }
    if case_sets[0] != case_sets[1]:
        raise ValueError("candidate pilots did not use identical case IDs")
    local = per_candidate["QWEN3_LOCAL"]
    api = per_candidate["CONFIGURED_API_MODEL"]
    api_coverage = api["completed_case_count"] / 90
    local_coverage = local["completed_case_count"] / 90
    if local_coverage < float(config["generator_selection"]["minimum_completion_coverage_each_candidate"]):
        raise RuntimeError("local Qwen DEV pilot completion coverage is below the preregistered minimum")
    if api_coverage >= 0.95 and api["macro_accuracy"] >= local["macro_accuracy"] + 0.02:
        selected = "CONFIGURED_API_MODEL"
    else:
        selected = "QWEN3_LOCAL"
    result = {
        "schema_version": "e1-2-generator-selection-v1",
        "config_sha256": config_sha256,
        "selection_uses": "DEV closed-book matched 90-case pilot only",
        "selected_candidate": selected,
        "rule": config["generator_selection"]["selection_rule"],
        "candidates": per_candidate,
        "case_id_set_sha256": hashlib.sha256("\n".join(sorted(case_sets[0])).encode()).hexdigest(),
    }
    write_json_atomic(SELECTION_PATH, result, refuse_overwrite=True)
    return result


def run_jev_preflight() -> dict[str, Any]:
    from health_ai_copilot.routing.jev import JevConfig

    base: dict[str, Any] = {
        "requested_model": JEV_MODEL,
        "endpoint": "GET /v1/models",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "secret_values_recorded": False,
    }
    try:
        settings = JevConfig.from_env(dotenv_path=ROOT / ".env")
        request = Request(
            f"{settings.base_url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {settings.api_key}", "Accept": "application/json"},
            method="GET",
        )
        with urlopen(request, timeout=settings.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("models", payload.get("data", [])) if isinstance(payload, dict) else []
        base.update(
            {
                "status": "AUTHENTICATED",
                "http_status": 200,
                "available_model_names": sorted(
                    str(row.get("name", row.get("id", "")))
                    for row in models
                    if isinstance(row, dict)
                ),
            }
        )
    except HTTPError as exc:
        base.update({"status": "BLOCKED_CREDENTIALS", "http_status": exc.code})
    except Exception as exc:  # noqa: BLE001 -- never persist request details or secrets
        base.update({"status": "BLOCKED_PREFLIGHT", "error_type": type(exc).__name__})
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / "runs" / "e1_2" / f"jev_preflight_{timestamp}.json"
    write_json_atomic(path, base, refuse_overwrite=True)
    print(json.dumps({"status": base["status"], "artifact": str(path)}))
    return base


def _pilot_random_evidence(
    connection: sqlite3.Connection, case_id: str, *, seed: str, top_k: int = 5
) -> list[dict[str, Any]]:
    count = int(connection.execute("SELECT count(*) FROM chunks").fetchone()[0])
    if count < top_k:
        raise ValueError("random-context corpus has fewer chunks than top_k")
    seed_bytes = hashlib.sha256(f"{seed}\0{case_id}".encode()).digest()
    generator = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    rowids = generator.sample(range(1, count + 1), top_k)
    placeholders = ",".join("?" for _ in rowids)
    rows = connection.execute(
        f"SELECT rowid,id,title,content,contents FROM chunks WHERE rowid IN ({placeholders})",
        rowids,
    )
    by_id = {
        int(row[0]): {"id": row[1], "title": row[2], "content": row[3], "contents": row[4]}
        for row in rows
    }
    if len(by_id) != top_k:
        raise RuntimeError("random-context index lookup returned an incomplete sample")
    return [by_id[rowid] for rowid in rowids]


def run_random_diagnostic(
    *, cases_path: Path, split_manifest_path: Path, scratch_root: Path, selection: dict[str, Any]
) -> None:
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    pilot_cases = select_generator_pilot(load_partition(cases_path, split_manifest, "DEV"))
    selected = str(selection["selected_candidate"])
    provider = create_provider(selected, scratch_root)
    index = scratch_root / "index" / "medrag_textbooks_fts5.sqlite3"
    if not index.is_file():
        raise FileNotFoundError("build the frozen BM25 index before the random-context diagnostic")
    output_dir = scratch_root / "runs" / "e1_2" / "random_context_dev"
    output_path = output_dir / "case_results.jsonl"
    manifest_path = output_dir / "manifest.json"
    identity = hashlib.sha256(
        f"{selection['config_sha256']}\0random_context_dev\0{selected}".encode()
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("result_identity") != identity:
            raise RuntimeError("random-context DEV output identity mismatch")
    else:
        write_json_atomic(
            manifest_path,
            {"status": "RUNNING", "result_identity": identity, "case_count": len(pilot_cases)},
        )
    completed = {
        row["case_id"]: row
        for row in read_jsonl(output_path)
        if row.get("result_identity") == identity
    } if output_path.exists() else {}
    connection = sqlite3.connect(index)
    try:
        with output_path.open("a", encoding="utf-8") as handle:
            for case in pilot_cases:
                case_id = str(case["case_id"])
                if case_id in completed:
                    continue
                evidence = _pilot_random_evidence(connection, case_id, seed="e1-2-random-context-v1")
                answer_case = case_for_answer(case)
                started = time.perf_counter()
                try:
                    output = provider.answer(case=answer_case, evidence=evidence)
                    status = "completed"
                    prediction = output["prediction"]
                    failure_type = None
                except Exception as exc:  # noqa: BLE001 -- do not persist provider response/body
                    output = {}
                    status = "failed"
                    prediction = None
                    failure_type = type(exc).__name__
                record = {
                    "case_id": case_id,
                    "subdataset": answer_case["subdataset"],
                    "result_identity": identity,
                    "status": status,
                    "prediction": prediction,
                    "is_correct": status == "completed" and is_correct(prediction, gold_label(case)),
                    "invalid_answer": bool(output.get("invalid_answer", True)),
                    "answer_input_tokens": output.get("answer_input_tokens"),
                    "answer_output_tokens": output.get("answer_output_tokens"),
                    "answer_latency_ms": output.get("answer_latency_ms", round((time.perf_counter() - started) * 1000)),
                    "retrieved_chunks": top_k if (top_k := len(evidence)) else 0,
                    "context_characters": sum(len(str(item.get("contents", ""))) for item in evidence),
                    "failure_type": failure_type,
                    "served_model": output.get("served_model"),
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
    finally:
        connection.close()
    write_json_atomic(
        manifest_path,
        {"status": "COMPLETED", "result_identity": identity, "case_count": len(pilot_cases)},
    )
    print(f"DEV random-context diagnostic completed: {len(pilot_cases)} cases")


def _load_arm_retrievals(
    *, arm: str, partition: str, scratch_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    base = scratch_root / "retrieval" / partition.lower()
    bm25: dict[str, dict[str, Any]] = {}
    medcpt: dict[str, dict[str, Any]] = {}
    if arm in {"rag_bm25", "cheap_router", "jev_router"}:
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


def run_arm(
    *,
    partition: str,
    arm: str,
    cases_path: Path,
    split_manifest_path: Path,
    scratch_root: Path,
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
        raise RuntimeError("Jev arm is blocked until authenticated model-list preflight succeeds")

    config, config_sha256 = load_frozen_config()
    selection = load_selection()
    if selection.get("config_sha256") != config_sha256:
        raise ValueError("generator selection was made against another frozen config")
    selected_candidate = str(selection["selected_candidate"])
    provider = create_provider(selected_candidate, scratch_root)
    candidate_config = next(
        row for row in config["generator_selection"]["candidates"] if row["id"] == selected_candidate
    )
    if provider.settings.model != candidate_config["requested_model"]:
        raise ValueError("selected generator model differs from the frozen config")
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
                    jev_served_model = response.model
                    retrieved, retrieval_latency_ms = _selected_retrieval(action, case_id, bm25_rows, medcpt_rows)
                except Exception as exc:  # noqa: BLE001 -- use fixed cheap failover and hide provider details
                    route = route_cheap(question, fallback_reason=type(exc).__name__)
                    action = route.action
                    decision = route.to_dict()
                    retrieved, retrieval_latency_ms = _selected_retrieval(action, case_id, bm25_rows, medcpt_rows)

            evidence = []
            retrieval_missing = False
            if action is RetrievalAction.RAG_BM25 or action is RetrievalAction.RAG_MEDCPT:
                if retrieved is None:
                    retrieval_missing = True
                else:
                    raw_evidence = retrieved.get("retrieved_evidence")
                    evidence = raw_evidence if isinstance(raw_evidence, list) else []
            context_characters = sum(
                len(str(item.get("contents") or item.get("content") or ""))
                + len(str(item.get("title") or ""))
                for item in evidence
                if isinstance(item, dict)
            )
            started = time.perf_counter()
            if retrieval_missing:
                output = {}
                status = "failed"
                prediction = None
                failure_type = "missing_retrieval_artifact"
            else:
                try:
                    output = provider.answer(case=answer_case, evidence=evidence)
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
                "invalid_answer": bool(output.get("invalid_answer", True)),
                "failure_type": failure_type,
                "selected_action": action.value,
                "route_decision": decision,
                "route_latency_ms": route_latency_ms,
                "router_calls": router_calls,
                "router_input_tokens": router_tokens["input"],
                "router_output_tokens": router_tokens["output"],
                "jev_served_model": jev_served_model,
                "retrieval_calls": int(action is not RetrievalAction.CLOSED_BOOK),
                "retrieved_chunks": len(evidence),
                "retrieval_latency_ms": retrieval_latency_ms,
                "context_characters": context_characters,
                "answer_input_tokens": output.get("answer_input_tokens"),
                "answer_output_tokens": output.get("answer_output_tokens"),
                "answer_latency_ms": answer_wall_ms,
                "end_to_end_latency_ms": route_latency_ms + retrieval_latency_ms + answer_wall_ms,
                "requested_model": output.get("requested_model", provider.settings.model),
                "served_model": output.get("served_model"),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if index % 50 == 0 or index == len(cases):
                print(f"{partition_name} {arm_name}: {index}/{len(cases)} cases", flush=True)

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
        choices=("pilot", "select-generator", "prepare-inputs", "retrieve", "random-diagnostic", "jev-preflight", "run-arm"),
        required=True,
    )
    parser.add_argument("--candidate", choices=("QWEN3_LOCAL", "CONFIGURED_API_MODEL"))
    parser.add_argument("--partition", choices=("DEV", "TEST"))
    parser.add_argument("--arm", choices=ARM_NAMES)
    parser.add_argument("--retriever", choices=("bm25", "medcpt"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch-root", type=Path, default=SCRATCH_DEFAULT)
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
    elif args.phase == "random-diagnostic":
        selection = load_selection()
        if selection.get("config_sha256") != config_sha256:
            raise ValueError("generator selection was made against another frozen config")
        run_random_diagnostic(
            cases_path=args.cases,
            split_manifest_path=args.split_manifest,
            scratch_root=args.scratch_root,
            selection=selection,
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
            limit=args.limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
