"""Execute frozen RAG views with Vanilla or CFEC readers, without evaluator data."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from eval.r2med_crb_data import PINNED_UPSTREAM_COMMIT
from eval.r2med_gar_generation import EXPECTED_UPSTREAM_FILES
from eval.r2med_multiview import (
    RankedDocument,
    ensure_java_home,
)
from eval.rag_e6.data import (
    E6Episode,
    load_partition_corpus,
    load_partition_episodes,
)
from eval.rag_e6.llm import (
    COMMON_SYSTEM_PROMPT,
    COMPLETION_CEILING,
    CONTEXT_CEILING,
    LAMER_SYSTEM_PROMPT,
    MODEL_NAME,
    MODEL_SHA256,
    PROMPT_BYTE_SAFETY_MARGIN,
    REASONING_ENABLED,
    SERVER_COMPLETION_CEILING,
    TEMPERATURE,
    TOP_P,
    CallJournal,
    LocalLlamaCppClient,
    sha256_text,
)
from eval.rag_e6.reader import (
    assign_requirement_ids,
    claim_prompt,
    claims_used_evidence,
    composer_prompt,
    decompose_prompt,
    evidence_identity_sha256,
    final_citations_match_claims,
    issue_evidence_aliases,
    parse_claims,
    parse_last_final,
    parse_requirements,
    resolve_aliases,
    vanilla_prompt,
)
from eval.rag_e6.split import canonical_json_bytes, sha256_file, write_immutable_json
from eval.u3r_rag_transfer import (
    ANSWER_CONTEXT_K,
    BM25_B,
    BM25_K1,
    DEFAULT_BGE_PATH,
    DEFAULT_QWEN_PATH,
    DEFAULT_UPSTREAM_ROOT,
    STANDARD_RRF_K,
    STANDARD_RRF_WEIGHTS,
    STRONG_RRF_K,
    STRONG_RRF_WEIGHTS,
    TOP_K,
    U3R_RUN_ROOT,
    DenseDocumentCache,
    U3RDocument,
    U3RRetriever,
    _text_sha256,
    lamer_prompt,
    load_upstream_lamer_prompt,
    sha256_bytes,
)

ARM_ORDER = (
    "VANILLA_OFF", "VANILLA_STANDARD", "VANILLA_STRONG",
    "CFEC_STANDARD", "CFEC_STRONG",
)
PARTITION_ARMS = {
    "BUILD": ARM_ORDER,
    "FROZEN_DEV": ARM_ORDER,
}
DEFAULT_LLAMA_URL = "http://127.0.0.1:8092/v1"
DEFAULT_SERVER_MANIFEST = Path("runs/rag_e6/runtime/gpu_server_manifest.json")
DEFAULT_RUNTIME_CORPUS_ROOT = Path("runs/rag_e6/corpus")
U3R_RUNTIME_SOURCE_MODULES = {
    "src/health_ai_copilot/research/integration/actions.py":
        "health_ai_copilot.research.integration.actions",
    "src/health_ai_copilot/research/integration/contracts.py":
        "health_ai_copilot.research.integration.contracts",
    "src/health_ai_copilot/research/integration/evidence_world.py":
        "health_ai_copilot.research.integration.evidence_world",
    "src/health_ai_copilot/research/integration/executor.py":
        "health_ai_copilot.research.integration.executor",
    "src/health_ai_copilot/research/integration/tools.py":
        "health_ai_copilot.research.integration.tools",
    "src/health_ai_copilot/research/integration/owned_universe/schema.py":
        "health_ai_copilot.research.integration.owned_universe.schema",
}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(row) + b"\n"
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected object at {path.name}:{line_number}")
            rows.append(row)
    return rows


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output requires inspection: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_bge_cpu(path: Path, *, expected_sha256: str, threads: int = 8):
    weight_path = path / "model.safetensors"
    if sha256_file(weight_path) != expected_sha256:
        raise ValueError("pinned BGE-large weight SHA-256 mismatch")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(path), device="cpu", local_files_only=True)
    model.eval()
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("BGE-large must stay on CPU while Qwen uses the 4090")
    return model


def _verify_server_manifest(path: Path, *, qwen_path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "rag-e6a-gpu-server-v1"
        or manifest.get("host") != "127.0.0.1"
        or manifest.get("port") != 8092
        or manifest.get("model_sha256") != MODEL_SHA256
        or manifest.get("model_path") != str(qwen_path.resolve())
        or manifest.get("gpu_device") != "Vulkan1"
        or manifest.get("n_gpu_layers") != "all"
        or manifest.get("flash_attention") != "on"
        or manifest.get("cache_type_k") != "q4_0"
        or manifest.get("cache_type_v") != "q4_0"
        or manifest.get("context_ceiling") != CONTEXT_CEILING
        or manifest.get("completion_ceiling") != SERVER_COMPLETION_CEILING
        or manifest.get("effective_context_size") != 40960
        or manifest.get("model_native_context_size") != 40960
        or manifest.get("server_process_visible_in_nvidia_smi") is not True
        or manifest.get("health_status") != "ok"
    ):
        raise ValueError("llama.cpp manifest does not prove the frozen E6A GPU runtime")
    return manifest


def _call_event(
    journal: CallJournal,
    client: LocalLlamaCppClient,
    *,
    call_id: str,
    prompt: str,
    system_prompt: str = COMMON_SYSTEM_PROMPT,
) -> dict[str, Any]:
    result = journal.call_once(
        call_id=call_id,
        prompt=prompt,
        system_prompt=system_prompt,
        client=client,
    )
    if result.get("status") == "context_contract_violation":
        raise RuntimeError("generation prompt/response exceeded the effective model context budget")
    return result


def _rank_ids_and_aliases(
    view, documents: tuple[U3RDocument, ...]
) -> tuple[list[str], list[str], list[list[str]], tuple[Any, ...]]:
    document_by_id = {item.doc_id: item for item in documents}
    evidence_ids = list(view.final_ids)
    if any(item not in document_by_id for item in evidence_ids):
        raise ValueError("retriever returned a document outside the visible episode corpus")
    evidence = issue_evidence_aliases([
        {"doc_id": doc_id, "text": document_by_id[doc_id].text}
        for doc_id in evidence_ids
    ], limit=10)
    return (
        evidence_ids,
        list(view.candidate_union_ids),
        [list(channel) for channel in view.channel_ids],
        evidence,
    )


def _answer_row(
    *,
    arm: str,
    episode: E6Episode,
    evidence,
    parsed,
    call_ids: list[str],
    unknown_aliases: tuple[str, ...],
    retrieval_ids: list[str],
    candidate_ids: list[str],
    channels: list[list[str]],
    call_journal: CallJournal,
) -> dict[str, Any]:
    used_ids, final_unknown = resolve_aliases(parsed.cited_aliases, evidence)
    unknown_all = tuple(dict.fromkeys((*unknown_aliases, *final_unknown)))
    call_records = [call_journal.completed[call_id] for call_id in call_ids]
    failed_call_contract = any(
        row.get("status") != "ok" or row.get("finish_reason") == "length"
        for row in call_records
    )
    factual_answer_without_citation = (
        bool(evidence)
        and parsed.answer != "INSUFFICIENT_EVIDENCE"
        and not used_ids
    )
    return {
        "arm": arm,
        "partition": episode.partition,
        "retrieval_action": arm.removeprefix("VANILLA_").removeprefix("CFEC_"),
        "query_sha256": episode.query_sha256,
        "answer": parsed.answer,
        "answer_sha256": sha256_text(parsed.answer),
        "output_contract_failure": (
            parsed.contract_failure
            or bool(unknown_all)
            or failed_call_contract
            or factual_answer_without_citation
        ),
        "cited_aliases": list(parsed.cited_aliases),
        "unknown_aliases": list(unknown_all),
        "used_evidence_ids": list(used_ids),
        "ranked_evidence_ids": retrieval_ids,
        "candidate_union_ids": candidate_ids,
        "channel_ids": channels,
        "evidence_identity_sha256": evidence_identity_sha256(evidence),
        "evidence_aliases": [
            {"alias": item.alias, "document_id": item.document_id,
             "text_sha256": sha256_text(item.text)}
            for item in evidence
        ],
        "generation_call_ids": call_ids,
        "generation_call_records_sha256": [
            sha256_bytes(canonical_json_bytes(call_journal.completed[call_id]))
            for call_id in call_ids
        ],
    }


def _run_vanilla(
    *,
    arm: str,
    episode: E6Episode,
    evidence,
    retrieval_ids: list[str],
    candidate_ids: list[str],
    channels: list[list[str]],
    journal: CallJournal,
    client: LocalLlamaCppClient,
) -> dict[str, Any]:
    call_id = f"{episode.episode_id}|{arm}|reader"
    event = _call_event(
        journal,
        client,
        call_id=call_id,
        prompt=vanilla_prompt(episode.query, evidence),
    )
    parsed = parse_last_final(event["text"])
    return _answer_row(
        arm=arm,
        episode=episode,
        evidence=evidence,
        parsed=parsed,
        call_ids=[call_id],
        unknown_aliases=(),
        retrieval_ids=retrieval_ids,
        candidate_ids=candidate_ids,
        channels=channels,
        call_journal=journal,
    )


def _run_cfec(
    *,
    arm: str,
    episode: E6Episode,
    evidence,
    retrieval_ids: list[str],
    candidate_ids: list[str],
    channels: list[list[str]],
    journal: CallJournal,
    client: LocalLlamaCppClient,
) -> tuple[dict[str, Any], str]:
    decomposition_call_id = f"{episode.episode_id}|CFEC_SHARED|decompose"
    decomposition_event = _call_event(
        journal,
        client,
        call_id=decomposition_call_id,
        prompt=decompose_prompt(episode.query),
    )
    requirement_texts = parse_requirements(decomposition_event["text"])
    requirements = assign_requirement_ids(requirement_texts)
    call_ids = [decomposition_call_id]
    claims = []
    unknown_aliases: list[str] = []
    claim_contract_failures = 0
    unsupported_requirements: list[str] = []
    for requirement_id, requirement_text in requirements:
        claim_call_id = f"{episode.episode_id}|{arm}|claim|{requirement_id}"
        claim_event = _call_event(
            journal,
            client,
            call_id=claim_call_id,
            prompt=claim_prompt(requirement_text, evidence),
        )
        new_claims, unknown, contract_failure = parse_claims(
            claim_event["text"],
            requirement_id=requirement_id,
            evidence=evidence,
        )
        claims.extend(new_claims)
        unknown_aliases.extend(item for item in unknown if item not in unknown_aliases)
        claim_contract_failures += int(contract_failure)
        if not new_claims:
            unsupported_requirements.append(requirement_id)
        call_ids.append(claim_call_id)

    compose_call_id = f"{episode.episode_id}|{arm}|compose"
    compose_event = _call_event(
        journal,
        client,
        call_id=compose_call_id,
        prompt=composer_prompt(episode.query, claims),
    )
    parsed = parse_last_final(compose_event["text"])
    arm_row = _answer_row(
        arm=arm,
        episode=episode,
        evidence=evidence,
        parsed=parsed,
        call_ids=call_ids + [compose_call_id],
        unknown_aliases=tuple(unknown_aliases),
        retrieval_ids=retrieval_ids,
        candidate_ids=candidate_ids,
        channels=channels,
        call_journal=journal,
    )
    arm_row.update({
        "requirements": [
            {"requirement_id": requirement_id, "text": requirement_text}
            for requirement_id, requirement_text in requirements
        ],
        "validated_claims": [
            {"requirement_id": item.requirement_id, "claim_text": item.claim_text,
             "cited_aliases": list(item.cited_aliases),
             "evidence_ids": list(item.evidence_ids)}
            for item in claims
        ],
        "used_evidence_ids": list(claims_used_evidence(claims)),
        "unsupported_requirement_ids": unsupported_requirements,
        "claim_output_contract_failures": claim_contract_failures,
        "decomposition_output_empty": not bool(decomposition_event["text"]),
        "decomposition_call_id": decomposition_call_id,
        "decomposition_call_shared_between_cfec_arms": True,
    })
    if (
        claim_contract_failures > 0
        or bool(unknown_aliases)
        or not final_citations_match_claims(arm_row["cited_aliases"], claims)
    ):
        arm_row["output_contract_failure"] = True
    return arm_row, decomposition_call_id


def _load_lamer_bridge(
    *,
    episode: E6Episode,
    documents: tuple[U3RDocument, ...],
    original_bm25: list[RankedDocument],
    journal: CallJournal,
    client: LocalLlamaCppClient,
    upstream_root: Path,
) -> dict[str, Any]:
    feedback = original_bm25[:10]
    document_by_id = {item.doc_id: item for item in documents}
    prompt, prompt_sha = lamer_prompt(
        episode.query,
        [document_by_id[item.doc_id].text for item in feedback],
        upstream_root,
    )
    call_id = f"{episode.episode_id}|STRONG_RETRIEVAL|lamer_bridge"
    event = _call_event(
        journal,
        client,
        call_id=call_id,
        prompt=prompt,
        system_prompt=LAMER_SYSTEM_PROMPT,
    )
    valid = event.get("status") == "ok" and bool(event.get("text"))
    bridge_text = str(event["text"]) if valid else episode.query
    return {
        "call_id": call_id,
        "generated_text": bridge_text,
        "generated_text_sha256": sha256_text(bridge_text),
        "valid": valid,
        "completed": valid and event.get("finish_reason") != "length",
        "truncated": event.get("finish_reason") == "length",
        "fallback_original_query": not valid,
        "feedback_doc_ids": [item.doc_id for item in feedback],
        "prompt_sha256": prompt_sha,
        "call_record_sha256": sha256_bytes(canonical_json_bytes(event)),
    }


def _code_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "eval/rag_e6/data.py",
        "eval/rag_e6/llm.py",
        "eval/rag_e6/reader.py",
        "eval/rag_e6/reader_executor.py",
        "eval/rag_e6/split.py",
        "eval/u3r_rag_transfer.py",
        "eval/r2med_gar_generation.py",
        "eval/r2med_multiview.py",
        "tools/research/rag_e6/materialize_partition.py",
        "tools/research/rag_e6/record_gpu_server.py",
        "tools/research/rag_e6/run_build.py",
        "tools/research/rag_e6/start_gpu_server.ps1",
    )
    return {name: sha256_file(root / name) for name in names}


def _verify_inherited_u3r_runtime_source(repository_root: Path) -> dict[str, Any]:
    manifest_path = repository_root / U3R_RUN_ROOT / "u3r_counterfactual_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "u3r-counterfactual-freeze-v1"
        or manifest.get("evaluator_truth_opened") is not False
        or manifest.get("llm_cannot_access_evaluator_truth") is not True
        or manifest.get("llm_cannot_select_actions") is not True
    ):
        raise ValueError("pinned U3-R runtime manifest failed gold-blind validation")
    expected_hashes = manifest.get("code_sha256")
    if not isinstance(expected_hashes, dict):
        raise TypeError("pinned U3-R manifest has no code identity map")
    verified: dict[str, str] = {}
    for relative_path, module_name in U3R_RUNTIME_SOURCE_MODULES.items():
        expected = expected_hashes.get(relative_path)
        module = import_module(module_name)
        actual_path = getattr(module, "__file__", None)
        if not isinstance(expected, str) or not isinstance(actual_path, str):
            raise TypeError(f"pinned runtime module lacks an expected identity: {relative_path}")
        actual = sha256_file(Path(actual_path))
        if actual != expected:
            raise ValueError(f"inherited U3-R runtime module hash mismatch: {relative_path}")
        verified[relative_path] = actual
    return {
        "freeze_manifest_sha256": sha256_file(manifest_path),
        "code_commit": manifest.get("code_commit"),
        "verified_source_sha256": dict(sorted(verified.items())),
        "module_paths": {
            relative_path: str(Path(import_module(module_name).__file__).resolve())
            for relative_path, module_name in sorted(U3R_RUNTIME_SOURCE_MODULES.items())
        },
    }


def _retrieval_identity() -> dict[str, Any]:
    return {
        "standard": {
            "bm25": {"analyzer": "Lucene", "k1": BM25_K1, "b": BM25_B},
            "bge_revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
            "bge_weights_sha256": "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7",
            "rrf_k": STANDARD_RRF_K,
            "weights": list(STANDARD_RRF_WEIGHTS),
            "top_k": ANSWER_CONTEXT_K,
        },
        "strong": {
            "method": "pinned R2MED LameR-MV",
            "upstream_commit": PINNED_UPSTREAM_COMMIT,
            "prompt_family": "Stack-Medical",
            "feedback_top_k": 10,
            "retrieval_channels": ["BM25_QUERY", "BM25_QUERY_PLUS_BRIDGE", "BGE_QUERY", "BGE_BRIDGE"],
            "rrf_k": STRONG_RRF_K,
            "weights": list(STRONG_RRF_WEIGHTS),
            "top_k": ANSWER_CONTEXT_K,
            "original_query_bm25_reused": True,
        },
        "retriever_candidate_top_k": TOP_K,
        "retrieval_config_changed": False,
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _load_runtime_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or episode_id in result:
            raise ValueError("malformed or duplicate E6A runtime checkpoint")
        result[episode_id] = row
    return result


def execute_partition(
    *,
    partition: str,
    u2f_root: Path,
    split_manifest_path: Path,
    corpus_root: Path,
    output_root: Path,
    bge_path: Path = DEFAULT_BGE_PATH,
    qwen_path: Path = DEFAULT_QWEN_PATH,
    upstream_root: Path = DEFAULT_UPSTREAM_ROOT,
    llama_url: str = DEFAULT_LLAMA_URL,
    server_manifest_path: Path,
    cpu_threads: int = 8,
) -> dict[str, Any]:
    if partition not in PARTITION_ARMS:
        raise ValueError("only BUILD and FROZEN_DEV have E6A execution arms")
    if partition == "FROZEN_DEV" and not (output_root.parent / "protocol_lock.json").exists():
        raise FileNotFoundError("FROZEN_DEV execution requires a committed protocol lock")
    repository_root = Path(__file__).resolve().parents[2]
    inherited_u3r_identity = _verify_inherited_u3r_runtime_source(repository_root)
    if sha256_file(qwen_path) != MODEL_SHA256:
        raise ValueError("pinned Qwen3-8B model SHA-256 mismatch")
    server_manifest = _verify_server_manifest(server_manifest_path, qwen_path=qwen_path)
    if sha256_file(bge_path / "model.safetensors") != _retrieval_identity()["standard"]["bge_weights_sha256"]:
        raise ValueError("pinned BGE-large model SHA-256 mismatch")

    episodes = load_partition_episodes(
        u2f_root=u2f_root,
        split_manifest_path=split_manifest_path,
        partition=partition,
    )
    corpus_path = corpus_root / f"{partition.lower()}_runtime_corpus.jsonl"
    corpus_manifest_path = corpus_root / f"{partition.lower()}_corpus_manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if (
        corpus_manifest.get("partition") != partition
        or corpus_manifest.get("runtime_corpus_sha256") != sha256_file(corpus_path)
        or corpus_manifest.get("evaluator_truth_opened") is not False
        or corpus_manifest.get("future_train_outcomes_opened") is not False
        or corpus_manifest.get("reserved_test_ood_materialized") is not False
    ):
        raise ValueError("runtime corpus manifest failed gold-blind verification")
    corpora = load_partition_corpus(corpus_path, {item.episode_id for item in episodes})

    upstream_identity = load_upstream_lamer_prompt(upstream_root)
    upstream_identity["pinned_commit"] = PINNED_UPSTREAM_COMMIT
    upstream_identity["verified_source_sha256"] = dict(sorted(EXPECTED_UPSTREAM_FILES.items()))
    client = LocalLlamaCppClient(
        llama_url,
        model_name=str(server_manifest["model_api_id"]),
        effective_context_size=int(server_manifest["effective_context_size"]),
    )
    journal_path = output_root / f"{partition.lower()}_generation_calls.jsonl"
    journal = CallJournal(journal_path)
    checkpoint_path = output_root / f"{partition.lower()}_episode_checkpoint.jsonl"
    runtime_rows = _load_runtime_checkpoint(checkpoint_path)
    episode_by_id = {item.episode_id: item for item in episodes}
    for episode_id, row in runtime_rows.items():
        episode = episode_by_id.get(episode_id)
        if episode is None or row.get("query_sha256") != episode.query_sha256:
            raise ValueError("E6A runtime checkpoint differs from frozen episode input")
        if set(row.get("arms", {})) != set(ARM_ORDER):
            raise ValueError("E6A runtime checkpoint is missing a reader arm")
        required_calls = {
            call_id for arm in row["arms"].values()
            for call_id in arm.get("generation_call_ids", ())
        }
        if row.get("retrieval_bridge", {}).get("call_id"):
            required_calls.add(row["retrieval_bridge"]["call_id"])
        if not required_calls.issubset(journal.completed):
            raise ValueError("runtime checkpoint references an absent model-call result")

    missing = [item for item in episodes if item.episode_id not in runtime_rows]
    if missing:
        ensure_java_home()
        model = _load_bge_cpu(
            bge_path,
            expected_sha256=_retrieval_identity()["standard"]["bge_weights_sha256"],
            threads=cpu_threads,
        )
        retriever = U3RRetriever(DenseDocumentCache(model))
    else:
        retriever = None
    started = time.monotonic()
    newly_completed = 0

    for episode in episodes:
        if episode.episode_id in runtime_rows:
            continue
        if retriever is None:
            raise RuntimeError("retrieval model was not initialized for unfinished episodes")
        documents = corpora[episode.episode_id]
        original_bm25 = retriever._bm25(documents, episode.query)
        retrieval_bridge = _load_lamer_bridge(
            episode=episode,
            documents=documents,
            original_bm25=original_bm25,
            journal=journal,
            client=client,
            upstream_root=upstream_root,
        )
        standard_view = retriever.standard(episode, documents)
        strong_view = retriever.strong(
            episode,
            documents,
            retrieval_bridge["generated_text"],
            original_bm25,
        )
        standard_ids, standard_candidates, standard_channels, standard_evidence = (
            _rank_ids_and_aliases(standard_view, documents)
        )
        strong_ids, strong_candidates, strong_channels, strong_evidence = (
            _rank_ids_and_aliases(strong_view, documents)
        )
        off = _run_vanilla(
            arm="VANILLA_OFF",
            episode=episode,
            evidence=(),
            retrieval_ids=[],
            candidate_ids=[],
            channels=[],
            journal=journal,
            client=client,
        )
        vanilla_standard = _run_vanilla(
            arm="VANILLA_STANDARD",
            episode=episode,
            evidence=standard_evidence,
            retrieval_ids=standard_ids,
            candidate_ids=standard_candidates,
            channels=standard_channels,
            journal=journal,
            client=client,
        )
        vanilla_strong = _run_vanilla(
            arm="VANILLA_STRONG",
            episode=episode,
            evidence=strong_evidence,
            retrieval_ids=strong_ids,
            candidate_ids=strong_candidates,
            channels=strong_channels,
            journal=journal,
            client=client,
        )
        cfec_standard, shared_decomposition_call = _run_cfec(
            arm="CFEC_STANDARD",
            episode=episode,
            evidence=standard_evidence,
            retrieval_ids=standard_ids,
            candidate_ids=standard_candidates,
            channels=standard_channels,
            journal=journal,
            client=client,
        )
        cfec_strong, second_decomposition_call = _run_cfec(
            arm="CFEC_STRONG",
            episode=episode,
            evidence=strong_evidence,
            retrieval_ids=strong_ids,
            candidate_ids=strong_candidates,
            channels=strong_channels,
            journal=journal,
            client=client,
        )
        if shared_decomposition_call != second_decomposition_call:
            raise ValueError("CFEC requirement decomposition must be shared within episode")
        if (
            vanilla_standard["evidence_identity_sha256"]
            != cfec_standard["evidence_identity_sha256"]
            or vanilla_strong["evidence_identity_sha256"]
            != cfec_strong["evidence_identity_sha256"]
        ):
            raise ValueError("Vanilla and CFEC did not receive byte-identical evidence")
        row = {
            "episode_id": episode.episode_id,
            "subject_id": episode.subject_id,
            "partition": partition,
            "split": episode.split,
            "query_sha256": episode.query_sha256,
            "visible_document_ids_sha256": sha256_bytes(canonical_json_bytes([
                [item.doc_id, _text_sha256(item.text)] for item in documents
            ])),
            "retrieval_bridge": retrieval_bridge,
            "retrieval_cost": {
                "standard_search_invocations": standard_view.search_invocations,
                "strong_search_invocations": strong_view.search_invocations,
                "strong_lamer_bridge_call_id": retrieval_bridge["call_id"],
            },
            "arms": {
                "VANILLA_OFF": off,
                "VANILLA_STANDARD": vanilla_standard,
                "VANILLA_STRONG": vanilla_strong,
                "CFEC_STANDARD": cfec_standard,
                "CFEC_STRONG": cfec_strong,
            },
            "shared_cfec_decomposition_call_id": shared_decomposition_call,
        }
        _append_jsonl(checkpoint_path, row)
        runtime_rows[episode.episode_id] = row
        newly_completed += 1
        if newly_completed % 8 == 0 or len(runtime_rows) == len(episodes):
            print(json.dumps({
                "partition": partition,
                "completed_this_process": newly_completed,
                "completed_total": len(runtime_rows),
                "episodes_total": len(episodes),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }, sort_keys=True), flush=True)

    if set(runtime_rows) != set(episode_by_id):
        raise ValueError("E6A runtime did not execute every selected episode")
    ordered_rows = [runtime_rows[item.episode_id] for item in episodes]
    output_path = output_root / f"{partition.lower()}_reader_outputs.jsonl"
    output_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in ordered_rows)
    _write_immutable_bytes(output_path, output_payload)

    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    code_hashes = _code_manifest()
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    partition_call_records = {
        call_id: row for call_id, row in journal.completed.items()
        if call_id.split("|", 1)[0] in episode_by_id
    }
    manifest = {
        "schema_version": "rag-e6a-runtime-freeze-v1",
        "partition": partition,
        "arms": list(ARM_ORDER),
        "episode_count": len(episodes),
        "arm_execution_count": len(episodes) * len(ARM_ORDER),
        "all_partition_episodes_executed": True,
        "primary_slice_applied_before_execution": False,
        "dataset_root_sha256": split_manifest["dataset_root_sha256"],
        "source_u2f_manifest_sha256": split_manifest["source_u2f_manifest_sha256"],
        "source_train_runtime_episodes_sha256": split_manifest["source_train_runtime_episodes_sha256"],
        "subject_split_manifest_sha256": sha256_file(split_manifest_path),
        "runtime_corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "runtime_corpus_sha256": sha256_file(corpus_path),
        "query_order_sha256": sha256_bytes(canonical_json_bytes([
            [item.episode_id, item.query_sha256] for item in episodes
        ])),
        "reader_output_sha256": sha256_bytes(output_payload),
        "generation_call_journal_sha256": sha256_file(journal_path),
        "generation_calls_by_stage": dict(sorted(Counter(
            call_id.split("|")[1] + "/" + call_id.split("|")[2]
            for call_id in partition_call_records
        ).items())),
        "code_commit": code_commit,
        "generator": {
            "model_name": MODEL_NAME,
            "model_api_id": client.model_name,
            "model_sha256": MODEL_SHA256,
            "context_ceiling": CONTEXT_CEILING,
            "completion_ceiling": COMPLETION_CEILING,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "reasoning_enabled": REASONING_ENABLED,
            "retry_count": 0,
            "prompt_byte_safety_margin": PROMPT_BYTE_SAFETY_MARGIN,
            "generation_calls_attempted": sum(
                int(row.get("provider_calls", 0)) for row in partition_call_records.values()
            ),
            "generation_calls_nonempty": sum(
                row.get("status") == "ok" for row in partition_call_records.values()
            ),
            "generation_calls_truncated": sum(
                row.get("finish_reason") == "length" for row in partition_call_records.values()
            ),
            "generation_calls_rejected_by_context_guard": sum(
                row.get("status") == "context_contract_violation"
                for row in partition_call_records.values()
            ),
            "actual_backend": server_manifest["backend"],
        },
        "retrieval": _retrieval_identity(),
        "upstream_lamer": upstream_identity,
        "inherited_u3r_runtime": inherited_u3r_identity,
        "server_manifest_sha256": sha256_file(server_manifest_path),
        "server_manifest": server_manifest,
        "bge_device": "cpu",
        "cpu_threads_for_bge": cpu_threads,
        "packages": {
            "torch": _package_version("torch"),
            "sentence-transformers": _package_version("sentence-transformers"),
            "pyserini": _package_version("pyserini"),
            "gensim": _package_version("gensim"),
        },
        "code_sha256": code_hashes,
        "evaluator_truth_opened": False,
        "future_train_outcomes_opened": False,
        "reserved_test_ood_materialized": False,
        "reserved_test_ood_opened": False,
        "llm_cannot_access_evaluator_truth": True,
        "llm_cannot_select_actions": True,
        "llm_cannot_change_retrieval_or_evidence_scope": True,
        "llm_cannot_change_requirement_ids_or_provenance": True,
        "sft_started": False,
        "opd_started": False,
        "grpo_started": False,
    }
    write_immutable_json(output_root / f"{partition.lower()}_manifest.json", manifest)
    return manifest
