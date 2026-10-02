"""Gold-blind, deterministic CAV-v1 verifier preflight over frozen BUILD inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.rag_e6.data import load_partition_corpus, load_partition_episodes
from eval.rag_e6.llm import (
    COMMON_SYSTEM_PROMPT,
    CallJournal,
    LocalLlamaCppClient,
    sha256_text,
)
from eval.rag_e6.reader import (
    cav_verifier_prompt,
    evidence_identity_sha256,
    issue_evidence_aliases,
    parse_cav_response,
)
from eval.rag_e6.reader_executor import (
    DEFAULT_QWEN_PATH,
    DEFAULT_RUNTIME_CORPUS_ROOT,
    DEFAULT_SERVER_MANIFEST,
    _verify_server_manifest,
)
from eval.rag_e6.split import sha256_file, write_immutable_json

U2F_ROOT = ROOT / "runs/integration/u2f-owned-v1-55955b2eff38"
SPLIT_MANIFEST = ROOT / "runs/rag_e6/split_manifest.json"
BASELINE_ROOT = ROOT / "runs/rag_e6/build_v5"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/rag_e6/smoke_cav_v1"
SAMPLE_SIZE = 24
MIN_VALID_RATE = 0.95


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path.name}:{line_number}")
            rows.append(value)
    return rows


def _completed_calls(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        call_id = row.get("call_id")
        if row.get("event") != "completed" or not isinstance(call_id, str):
            continue
        if call_id in result:
            raise ValueError(f"duplicate completed baseline call: {call_id}")
        result[call_id] = row
    return result


def _select_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < SAMPLE_SIZE:
        raise ValueError("frozen BUILD outputs are smaller than the preflight sample")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"CAV-v1-preflight\0{row['episode_id']}".encode()
        ).hexdigest(),
    )
    return ordered[:SAMPLE_SIZE]


def run_preflight(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    baseline_root: Path = BASELINE_ROOT,
    corpus_root: Path = DEFAULT_RUNTIME_CORPUS_ROOT,
    split_manifest_path: Path = SPLIT_MANIFEST,
    u2f_root: Path = U2F_ROOT,
    server_manifest_path: Path = DEFAULT_SERVER_MANIFEST,
    qwen_path: Path = DEFAULT_QWEN_PATH,
    llama_url: str = "http://127.0.0.1:8092/v1",
) -> dict[str, Any]:
    baseline_manifest_path = baseline_root / "build_manifest.json"
    baseline_output_path = baseline_root / "build_reader_outputs.jsonl"
    baseline_journal_path = baseline_root / "build_generation_calls.jsonl"
    baseline_manifest = _read_json(baseline_manifest_path)
    if (
        baseline_manifest.get("partition") != "BUILD"
        or baseline_manifest.get("episode_count") != 818
        or baseline_manifest.get("evaluator_truth_opened") is not False
        or baseline_manifest.get("reserved_test_ood_opened") is not False
        or baseline_manifest.get("reader_output_sha256") != sha256_file(baseline_output_path)
        or baseline_manifest.get("generation_call_journal_sha256")
        != sha256_file(baseline_journal_path)
    ):
        raise ValueError("frozen BUILD baseline artifacts failed gold-blind identity checks")
    if sha256_file(qwen_path) != baseline_manifest["generator"]["model_sha256"]:
        raise ValueError("preflight Qwen file does not match the frozen baseline model")
    split_manifest = _read_json(split_manifest_path)
    if split_manifest.get("evaluator_truth_opened") is not False:
        raise ValueError("subject split records evaluator-truth access")
    _verify_server_manifest(server_manifest_path, qwen_path=qwen_path)

    episodes = load_partition_episodes(
        u2f_root=u2f_root,
        split_manifest_path=split_manifest_path,
        partition="BUILD",
    )
    episode_by_id = {item.episode_id: item for item in episodes}
    corpus = load_partition_corpus(
        corpus_root / "build_runtime_corpus.jsonl", set(episode_by_id)
    )
    output_rows = _read_jsonl(baseline_output_path)
    output_by_id = {row.get("episode_id"): row for row in output_rows}
    if (
        len(output_by_id) != len(output_rows)
        or set(output_by_id) != set(episode_by_id)
        or len(episodes) != 818
    ):
        raise ValueError("baseline BUILD outputs do not align with the frozen runtime episodes")
    calls = _completed_calls(baseline_journal_path)
    sample = _select_sample(output_rows)
    journal = CallJournal(output_root / "cav_preflight_calls.jsonl")
    client = LocalLlamaCppClient(
        llama_url,
        model_name=str(_read_json(server_manifest_path)["model_api_id"]),
        effective_context_size=40960,
    )

    outcomes: Counter[str] = Counter()
    sample_rows = []
    for output in sample:
        episode_id = output["episode_id"]
        episode = episode_by_id[episode_id]
        baseline = output["arms"].get("VANILLA_STRONG")
        if baseline is None or baseline.get("query_sha256") != episode.query_sha256:
            raise ValueError("preflight baseline output differs from its runtime query")
        evidence_ids = baseline.get("ranked_evidence_ids", [])
        document_by_id = {item.doc_id: item for item in corpus[episode_id]}
        if len(evidence_ids) > 10 or any(doc_id not in document_by_id for doc_id in evidence_ids):
            raise ValueError("baseline evidence escaped the frozen top-10 runtime corpus")
        evidence = issue_evidence_aliases([
            {"doc_id": doc_id, "text": document_by_id[doc_id].text}
            for doc_id in evidence_ids
        ])
        evidence_sha = evidence_identity_sha256(evidence)
        if evidence_sha != baseline.get("evidence_identity_sha256"):
            raise ValueError("reconstructed baseline evidence identity does not match")
        baseline_call_ids = baseline.get("generation_call_ids", [])
        if len(baseline_call_ids) != 1 or baseline_call_ids[0] not in calls:
            raise ValueError("baseline output does not reference exactly one completed reader call")
        baseline_event = calls[baseline_call_ids[0]]
        if baseline_event.get("status") != "ok" or not baseline_event.get("text"):
            raise ValueError("baseline draft call has no usable frozen completion record")

        call_id = f"{episode_id}|CAV_PREFLIGHT|verify"
        event = journal.call_once(
            call_id=call_id,
            prompt=cav_verifier_prompt(episode.query, baseline_event["text"], evidence),
            system_prompt=COMMON_SYSTEM_PROMPT,
            client=client,
        )
        bad_call = event.get("status") != "ok" or event.get("finish_reason") == "length"
        decision = None if bad_call else parse_cav_response(event["text"], evidence)
        valid = (
            decision is not None
            and decision.action in {"KEEP", "REPAIR"}
            and not decision.contract_failure
            and (decision.action != "REPAIR" or decision.parsed is not None)
        )
        label = "valid" if valid else (
            "truncated" if event.get("finish_reason") == "length" else
            "call_failure" if bad_call else "contract_failure"
        )
        outcomes[label] += 1
        sample_rows.append({
            "episode_id": episode_id,
            "sample_key_sha256": hashlib.sha256(
                f"CAV-v1-preflight\0{episode_id}".encode()
            ).hexdigest(),
            "baseline_answer_sha256": baseline.get("answer_sha256"),
            "evidence_identity_sha256": evidence_sha,
            "verification_call_id": call_id,
            "verification_response_sha256": sha256_text(event.get("text", "")),
            "action": "FALLBACK" if decision is None else decision.action,
            "unknown_aliases": [] if decision is None else list(decision.unknown_aliases),
            "contract_valid": valid,
            "finish_reason": event.get("finish_reason"),
            "status": event.get("status"),
        })
        print(json.dumps({
            "completed": len(sample_rows),
            "total": SAMPLE_SIZE,
            "valid": outcomes["valid"],
            "last_episode": episode_id,
        }, sort_keys=True), flush=True)

    valid_rate = outcomes["valid"] / SAMPLE_SIZE
    passed = valid_rate >= MIN_VALID_RATE and outcomes["truncated"] == 0
    report = {
        "schema_version": "rag-e6a-cav-preflight-v1",
        "method": "CAV-v1",
        "partition": "BUILD",
        "sample_size": SAMPLE_SIZE,
        "sample_selection": "lowest SHA256('CAV-v1-preflight\\0' + episode_id), independent of evaluator labels",
        "valid_response_threshold": MIN_VALID_RATE,
        "valid_responses": outcomes["valid"],
        "valid_response_rate": valid_rate,
        "outcomes": dict(sorted(outcomes.items())),
        "zero_truncation_required": True,
        "passed": passed,
        "evaluator_truth_opened": False,
        "reserved_test_ood_opened": False,
        "baseline_manifest_sha256": sha256_file(baseline_manifest_path),
        "baseline_reader_output_sha256": sha256_file(baseline_output_path),
        "baseline_generation_journal_sha256": sha256_file(baseline_journal_path),
        "subject_split_manifest_sha256": sha256_file(split_manifest_path),
        "runtime_corpus_sha256": sha256_file(corpus_root / "build_runtime_corpus.jsonl"),
        "server_manifest_sha256": sha256_file(server_manifest_path),
        "model_sha256": baseline_manifest["generator"]["model_sha256"],
        "method_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sample": sample_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_immutable_json(output_root / "cav_preflight_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--llama-url", default="http://127.0.0.1:8092/v1")
    args = parser.parse_args()
    report = run_preflight(output_root=args.output_root, llama_url=args.llama_url)
    print(json.dumps({
        "passed": report["passed"],
        "valid_responses": report["valid_responses"],
        "sample_size": report["sample_size"],
        "valid_response_rate": report["valid_response_rate"],
        "outcomes": report["outcomes"],
        "evaluator_truth_opened": report["evaluator_truth_opened"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
