"""Run non-benchmark synthetic-path smoke tests against the pinned local model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.rag_e6.llm import (
    COMMON_SYSTEM_PROMPT,
    LAMER_SYSTEM_PROMPT,
    CallJournal,
    LocalLlamaCppClient,
)
from eval.rag_e6.reader import (
    assign_requirement_ids,
    claim_prompt,
    composer_prompt,
    decompose_prompt,
    issue_evidence_aliases,
    parse_claims,
    parse_last_final,
    parse_requirements,
    vanilla_prompt,
)
from eval.rag_e6.reader_executor import (
    DEFAULT_UPSTREAM_ROOT,
    _verify_server_manifest,
)
from eval.rag_e6.split import sha256_file, write_immutable_json
from eval.u3r_rag_transfer import DEFAULT_QWEN_PATH, lamer_prompt

DEFAULT_SERVER_MANIFEST = ROOT / "runs/rag_e6/runtime/gpu_server_manifest.json"
OUTPUT_ROOT = ROOT / "runs/rag_e6/smoke"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-url", default="http://127.0.0.1:8092/v1")
    parser.add_argument("--server-manifest", type=Path, default=DEFAULT_SERVER_MANIFEST)
    parser.add_argument("--qwen-path", type=Path, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    server = _verify_server_manifest(args.server_manifest, qwen_path=args.qwen_path)
    client = LocalLlamaCppClient(
        args.llama_url,
        model_name=str(server["model_api_id"]),
        effective_context_size=int(server["effective_context_size"]),
    )
    journal = CallJournal(args.output_root / "smoke_call_journal.jsonl")
    question = "For this synthetic example, state the recommended follow-up interval and its caveat."
    passages = [
        "Synthetic example only: the fictional follow-up interval is 12 months.",
        "Synthetic example only: the interval is illustrative and not real medical guidance.",
    ]
    lamer_text, lamer_prompt_sha = lamer_prompt(question, passages, args.upstream_root)
    lamer_event = journal.call_once(
        call_id="smoke|lamer_bridge",
        prompt=lamer_text,
        system_prompt=LAMER_SYSTEM_PROMPT,
        client=client,
    )
    evidence = issue_evidence_aliases([
        {"doc_id": "SMOKE-DOC-1", "text": passages[0]},
        {"doc_id": "SMOKE-DOC-2", "text": passages[1]},
    ])
    vanilla_event = journal.call_once(
        call_id="smoke|vanilla",
        prompt=vanilla_prompt(question, evidence),
        system_prompt=COMMON_SYSTEM_PROMPT,
        client=client,
    )
    vanilla = parse_last_final(vanilla_event["text"])
    decomposer_event = journal.call_once(
        call_id="smoke|decompose",
        prompt=decompose_prompt(question),
        system_prompt=COMMON_SYSTEM_PROMPT,
        client=client,
    )
    requirements = assign_requirement_ids(parse_requirements(decomposer_event["text"]))
    if not requirements:
        raise RuntimeError("synthetic decomposer smoke produced no requirements")
    requirement_id, requirement_text = requirements[0]
    claim_event = journal.call_once(
        call_id="smoke|claim|req_1",
        prompt=claim_prompt(requirement_text, evidence),
        system_prompt=COMMON_SYSTEM_PROMPT,
        client=client,
    )
    claims, unknown_aliases, claim_failure = parse_claims(
        claim_event["text"], requirement_id=requirement_id, evidence=evidence
    )
    compose_event = journal.call_once(
        call_id="smoke|compose",
        prompt=composer_prompt(question, claims),
        system_prompt=COMMON_SYSTEM_PROMPT,
        client=client,
    )
    composed = parse_last_final(compose_event["text"])
    checks = {
        "lamer_call_nonempty": lamer_event["status"] == "ok",
        "vanilla_final_contract": not vanilla.contract_failure,
        "decomposer_requirements": bool(requirements),
        "claim_valid_alias": bool(claims),
        "claim_no_contract_failure": not claim_failure,
        "composer_final_contract": not composed.contract_failure,
        "unknown_aliases_absent": not unknown_aliases,
    }
    report = {
        "schema_version": "rag-e6a-synthetic-reader-smoke-v1",
        "benchmark_or_dataset_rows_used": False,
        "model_sha256": server["model_sha256"],
        "server_manifest_sha256": sha256_file(args.server_manifest),
        "lamer_prompt_sha256": lamer_prompt_sha,
        "call_journal_sha256": sha256_file(args.output_root / "smoke_call_journal.jsonl"),
        "generation_calls": len(journal.completed),
        "nonempty_calls": sum(item.get("status") == "ok" for item in journal.completed.values()),
        "checks": checks,
        "pass": all(checks.values()),
        "evaluator_truth_opened": False,
        "retrieval_configuration_changed": False,
        "synthetic_evidence_source_ids": [item.document_id for item in evidence],
    }
    write_immutable_json(args.output_root / "smoke_report.json", report)
    print(json.dumps(report, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
