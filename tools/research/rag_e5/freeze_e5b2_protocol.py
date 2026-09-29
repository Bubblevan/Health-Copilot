"""Freeze the corrected E5-B2 scorer and verify all B1 inputs before execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

from eval.r2med_crb_data import PINNED_UPSTREAM_COMMIT, UPSTREAM_PROMPT_FAMILY
from eval.r2med_gar_generation import (
    EXPECTED_UPSTREAM_FILES,
    load_upstream_prompt_catalog,
    prompt_sha256,
)
from eval.rag_e5.age import AGE_TEMPORAL_CONTRACT_SHA256
from eval.rag_e5.longitudinal_state import PROFILE_TEMPORAL_CONTRACT_SHA256
from eval.rag_e5.overlay import (
    _RUBRIC_TEMPLATES,
    BATCH_ID,
    CAPABILITY_CONTEXT,
    CORPUS_IDENTITY,
    READER_PROMPT,
    READER_SCHEMA,
    SCORER_VERSION,
    TASK_TEMPLATES,
    build_overlay_cases,
    score_case,
)
from eval.rag_e5.temporal import TEMPORAL_SEMANTICS_ID, TEMPORAL_SEMANTICS_SHA256
from tools.research.rag_e5.build_e5b1_overlay import (
    _packet_set_sha,
    _validate_packet,
)
from tools.research.rag_e5.build_e5b1_state_packets import (
    DEVELOPMENT_USERS,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRIVATE_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b1")
DEFAULT_CORPUS_ROOT = Path(
    r"D:\MyLab\Jianli\external\rag_e5\e5a3\corpus\public_health_plus_guideline"
)
DEFAULT_INDEX_ROOT = Path(
    r"D:\MyLab\Jianli\external\rag_e5\e5a3\indexes\public_health_plus_guideline"
)
DEFAULT_MODEL_ROOT = Path(r"E:\Health-Copilot-Models\models")
DEFAULT_UPSTREAM_ROOT = Path(r"D:\MyLab\Jianli\external\rag\R2MED")
BGE_MODEL_FILES = (
    "config_sentence_transformers.json",
    "config.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.txt",
    "1_Pooling/config.json",
)
LOCK_PATH = ROOT / "runs/rag_e5/e5b2_protocol_lock.json"
CONTRACT_PATH = ROOT / "runs/rag_e5/e5b2_scoring_contract.json"
TASK_HASH = "7ac2d95f98efc6fcced665f93977c76c4d219af0cfe7e16f05b1fdd67358979a"
STATE_SET_HASH = "860f5da64a6b94790944a5bd2d50aa4977a098bc51d1b6f822321bf4fbbd1d1c"
STANDARD_PROFILE_HASH = "6ddb91bb0c31f5bc5b69372df6a3bdd3b4f71d70cdbcece8ef22c5bfd9a94330"
STRONG_PROFILE_HASH = "d9e3de9bf986a1f08b1c217153422d6e86ad0a84bc1982d90bade897c9274a8b"
MODEL_BYTES = 5_027_783_488
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
BGE_SHA256 = "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7"
LOCKED_CODE_FILES = (
    "eval/rag_e5/overlay.py",
    "eval/rag_e5/counterfactual.py",
    "eval/rag_e5/retrieval.py",
    "eval/rag_e5/e5b2_evaluator.py",
    "eval/rag_e5/E5LuceneAnalyzerCli.java",
    "eval/rag_e5/age.py",
    "eval/rag_e5/longitudinal_state.py",
    "eval/rag_e5/temporal.py",
    "eval/r2med_crb.py",
    "eval/r2med_crb_data.py",
    "eval/r2med_gar_generation.py",
    "eval/r2med_multiview.py",
    "tools/research/rag_e5/freeze_e5b2_protocol.py",
    "tools/research/rag_e5/run_e5b2_counterfactual.py",
    "tools/research/rag_e5/score_e5b2_counterfactual.py",
    "tools/research/rag_e5/build_e5b1_overlay.py",
    "tools/research/rag_e5/build_e5b1_state_packets.py",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"expected an object at {path.name}:{line_number}")
                rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def _verify_model(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError(f"required frozen model is missing or has wrong size: {path.name}")
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"frozen model SHA-256 mismatch: {path.name}")


def _build_scoring_contract(teacher_cases: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    recommendations: dict[str, Any] = {}
    for teacher in teacher_cases:
        source_id = teacher.get("required_external_source_id")
        if not source_id:
            continue
        for recommendation_id in teacher.get("required_recommendation_ids", []):
            source = recommendations.setdefault(source_id, {})
            if recommendation_id in source:
                continue
            source[recommendation_id] = {
                "approved_chunk_ids": sorted(
                    chunk["chunk_id"]
                    for chunk in chunks
                    if chunk.get("source_id") == source_id
                    and chunk.get("recommendation_id") == recommendation_id
                ),
                "keypoints": _RUBRIC_TEMPLATES[source_id][recommendation_id],
            }
    return {
        "schema_version": "rag-e5-e5b2-scoring-contract-v2",
        "scorer_version": SCORER_VERSION,
        "normalization": "casefold_and_collapse_whitespace; numbers_match_integer_token",
        "guideline_content_score": "rubric_anchor_match_in_guidance_facts_independent_of_retrieval_and_citations",
        "grounding_score": "cited_supplied_chunk_matching_required_source_and_recommendation_fraction",
        "state_score": "exact_required_field_enum_match_fraction",
        "end_to_end_weights": {
            "T0": {"state": 1.0},
            "T1": {"guideline_content": 0.75, "grounding": 0.25},
            "T2": {"state": 0.5, "guideline_content": 0.25, "grounding": 0.25},
        },
        "content_only_weights": {
            "T0": {"state": 1.0},
            "T1": {"guideline_content": 1.0},
            "T2": {"state": 0.5, "guideline_content": 0.5},
        },
        "recommendations": recommendations,
    }


def freeze_protocol(
    *,
    private_root: Path = DEFAULT_PRIVATE_ROOT,
    corpus_root: Path = DEFAULT_CORPUS_ROOT,
    index_root: Path = DEFAULT_INDEX_ROOT,
    model_root: Path = DEFAULT_MODEL_ROOT,
) -> dict[str, Any]:
    task_manifest = _read_json(ROOT / "runs/rag_e5/e5b1_task_manifest.json")
    old_lock = _read_json(ROOT / "runs/rag_e5/e5b_counterfactual_lock.json")
    coverage = _read_json(ROOT / "runs/rag_e5/e5b1_state_coverage_report.json")
    corpus_manifest = _read_json(ROOT / "runs/rag_e5/external_corpus_manifest.json")
    profiles_path = ROOT / "runs/rag_e5/retrieval_action_profiles.json"
    profiles_manifest = _read_json(profiles_path)
    profile_by_action = {row["action"]: row for row in profiles_manifest["profiles"]}
    standard = profile_by_action["STANDARD"]
    strong = profile_by_action["STRONG"]
    if task_manifest.get("scorer_version") != "e5b1-deterministic-evidence-gated-v1":
        raise ValueError("B1 input manifest is not the expected pre-erratum baseline")
    if old_lock.get("scorer_version") != task_manifest.get("scorer_version"):
        raise ValueError("B1 lock and task manifest disagree on the original scorer")
    if task_manifest.get("ordered_case_ids_sha256") != TASK_HASH:
        raise ValueError("B1 frozen task ID set differs from the approved task-set SHA")
    if task_manifest.get("state_packet_set_sha256") != STATE_SET_HASH:
        raise ValueError("B1 frozen state packet set differs from the approved SHA")
    if corpus_manifest.get("active_external_corpus_identity") != CORPUS_IDENTITY:
        raise ValueError("active external corpus identity changed")
    if standard.get("config_sha256") != STANDARD_PROFILE_HASH:
        raise ValueError("STANDARD profile config hash changed")
    if strong.get("config_sha256") != STRONG_PROFILE_HASH:
        raise ValueError("STRONG profile config hash changed")

    runtime_cases = _read_jsonl(private_root / "runtime_cases.jsonl")
    teacher_cases = _read_jsonl(private_root / "teacher_cases.jsonl")
    if len(runtime_cases) != 60 or len(teacher_cases) != 60:
        raise ValueError("B1 runtime and teacher artifacts must each contain exactly 60 cases")
    case_ids = [row.get("case_id") for row in runtime_cases]
    if _canonical_sha256(case_ids) != TASK_HASH:
        raise ValueError("runtime case IDs differ from the frozen task-set SHA")
    runtime_fields = {
        "case_id", "user_id", "question", "decision_boundary", "state_packet_ref",
        "runtime_capability_context_ref",
    }
    if any(set(row) != runtime_fields for row in runtime_cases):
        raise ValueError("runtime case schema contains missing or teacher-only fields")
    if [row.get("case_id") for row in teacher_cases] != case_ids:
        raise ValueError("teacher/runtime case identity or ordering changed")
    if json.loads((private_root / "capability_context.json").read_text(encoding="utf-8")) != CAPABILITY_CONTEXT:
        raise ValueError("runtime capability context changed")
    if json.loads((private_root / "task_templates.json").read_text(encoding="utf-8")) != TASK_TEMPLATES:
        raise ValueError("frozen task questions/templates changed")
    if (private_root / "reader_prompt.txt").read_text(encoding="utf-8") != READER_PROMPT:
        raise ValueError("reader prompt changed")
    if json.loads((private_root / "reader_schema.json").read_text(encoding="utf-8")) != READER_SCHEMA:
        raise ValueError("reader schema changed")

    states = []
    for user_id in DEVELOPMENT_USERS:
        packet = _read_json(private_root / "state_packets" / f"{user_id}.json")
        _validate_packet(packet, expected_user_id=user_id)
        states.append(packet)
    state_set_sha256 = _packet_set_sha(states)
    if state_set_sha256 != STATE_SET_HASH or coverage.get("state_packet_set_sha256") != state_set_sha256:
        raise ValueError("private B1 state packet set changed")

    chunks_path = corpus_root / "chunks.jsonl"
    chunks = _read_jsonl(chunks_path)
    eligible_source_ids = {
        row["source_id"]
        for row in corpus_manifest["guideline_candidates"]
        if row.get("task_authoring_eligible") is True
    }
    reconstructed_runtime, reconstructed_teacher = build_overlay_cases(
        state_packets=states,
        eligible_chunks=[chunk for chunk in chunks if chunk.get("source_id") in eligible_source_ids],
    )
    if [row.to_dict() for row in reconstructed_runtime] != runtime_cases:
        raise ValueError("frozen task questions/runtime projections changed")
    if [row.to_dict() for row in reconstructed_teacher] != teacher_cases:
        raise ValueError("frozen teacher targets or required chunk IDs changed")

    old_rubric = _read_json(private_root / "scoring_rubric.json")
    contract = _build_scoring_contract(teacher_cases, chunks)
    old_recommendations = old_rubric.get("recommendations", {})
    new_recommendations = contract["recommendations"]
    if {
        source_id: {
            rec_id: record["keypoints"] for rec_id, record in recommendation_rows.items()
        }
        for source_id, recommendation_rows in old_recommendations.items()
    } != {
        source_id: {
            rec_id: record["keypoints"] for rec_id, record in recommendation_rows.items()
        }
        for source_id, recommendation_rows in new_recommendations.items()
    }:
        raise ValueError("frozen rubric anchors changed during scorer correction")

    model_path = model_root / "qwen3-8b" / "Qwen3-8B-Q4_K_M.gguf"
    bge_path = model_root / "bge-large-en-v1.5" / "model.safetensors"
    _verify_model(model_path, expected_bytes=MODEL_BYTES, expected_sha256=MODEL_SHA256)
    _verify_model(bge_path, expected_bytes=1_340_616_616, expected_sha256=BGE_SHA256)
    bge_model_root = model_root / "bge-large-en-v1.5"
    bge_model_files = {
        relative: _sha256_file(bge_model_root / relative)
        for relative in BGE_MODEL_FILES
    }
    python_runtime = {
        "python": platform.python_version(),
        **{
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "torch", "transformers", "sentence-transformers")
        },
    }

    index_manifests = {
        name: _sha256_file(index_root / name)
        for name in (
            "bm25_index_manifest.json",
            "dense_index_manifest.json",
        )
    }
    bge_embedding_path = index_root / "bge_large_embeddings.npy"
    bm25_matrix_path = index_root / "bm25_document_term_weights.npz"
    bm25_token_path = index_root / "bm25_lucene_tokens.jsonl"
    bm25_model_path = index_root / "bm25_dictionary_model.json"
    bm25_manifest = _read_json(index_root / "bm25_index_manifest.json")

    scorer_source = (ROOT / "eval/rag_e5/overlay.py").read_bytes()
    scorer_sha256 = _sha256_bytes(scorer_source)
    code_records = []
    for relative in LOCKED_CODE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"execution code must exist before freeze: {relative}")
        code_records.append({"path": relative, "sha256": _sha256_file(path)})

    task_templates_sha256 = _canonical_sha256(TASK_TEMPLATES)
    question_set_sha256 = _canonical_sha256(
        [{"case_id": row["case_id"], "question": row["question"]} for row in runtime_cases]
    )
    scorer_function_sha256 = _sha256_bytes(
        __import__("inspect").getsource(score_case).encode("utf-8")
    )
    bridge_prompt_sha256 = strong["frozen_config"]["prompt_binding"]["template_sha256"]
    if bridge_prompt_sha256 != "317a15290b029422df211a96550f2f689d98fff61618a1e1847c7f54f8443122":
        raise ValueError("frozen LameR prompt template hash changed")
    if PINNED_UPSTREAM_COMMIT != "11244a4925a39082967a6c9d38ef01f279c316a5":
        raise ValueError("the pinned R2MED upstream commit changed")
    upstream_prompts = load_upstream_prompt_catalog(DEFAULT_UPSTREAM_ROOT)
    if prompt_sha256(upstream_prompts["lamer"]["MedQA-Diag"]) != bridge_prompt_sha256:
        raise ValueError("pinned upstream LameR prompt does not match the frozen profile")

    lock: dict[str, Any] = {
        "schema_version": "rag-e5-e5b2-counterfactual-lock-v1",
        "status": "FROZEN_PROTOCOL_NO_OUTCOMES",
        "batch_id": BATCH_ID,
        "base_main": "56a809024e3d006451724380e94f2b4ed9c2cee7",
        "case_count": 60,
        "expected_arms": 180,
        "ordered_case_ids_sha256": TASK_HASH,
        "question_set_sha256": question_set_sha256,
        "task_template_sha256": task_templates_sha256,
        "state_packet_set_sha256": state_set_sha256,
        "runtime_cases_file_sha256": _sha256_file(private_root / "runtime_cases.jsonl"),
        "teacher_cases_file_sha256": _sha256_file(private_root / "teacher_cases.jsonl"),
        "runtime_capability_context_sha256": _canonical_sha256(CAPABILITY_CONTEXT),
        "state_packet_builder_version": task_manifest["state_packet_builder_version"],
        "state_packet_config_sha256": task_manifest["state_packet_config_sha256"],
        "temporal_semantics_id": TEMPORAL_SEMANTICS_ID,
        "temporal_semantics_sha256": TEMPORAL_SEMANTICS_SHA256,
        "profile_temporal_contract_sha256": PROFILE_TEMPORAL_CONTRACT_SHA256,
        "age_temporal_contract_sha256": AGE_TEMPORAL_CONTRACT_SHA256,
        "external_corpus_identity": CORPUS_IDENTITY,
        "external_corpus_manifest_sha256": _sha256_file(ROOT / "runs/rag_e5/external_corpus_manifest.json"),
        "external_corpus_chunks_sha256": _sha256_file(chunks_path),
        "bm25_artifacts": {
            "manifest_sha256": index_manifests["bm25_index_manifest.json"],
            "analyzer_jar_sha256": bm25_manifest["analyzer"]["analyzer_jar_sha256"],
            "analyzer_cli_class_sha256": _sha256_file(
                index_root / "java-classes" / "E5LuceneAnalyzerCli.class"
            ),
            "matrix_sha256": _sha256_file(bm25_matrix_path),
            "token_artifact_sha256": _sha256_file(bm25_token_path),
            "dictionary_model_sha256": _sha256_file(bm25_model_path),
            "analyzer": "Anserini DefaultEnglishAnalyzer with Porter stemming",
            "k1": 0.9,
            "b": 0.4,
        },
        "bge_artifacts": {
            "weights_sha256": BGE_SHA256,
            "model_files_sha256": bge_model_files,
            "embedding_matrix_sha256": _sha256_file(bge_embedding_path),
            "dense_manifest_sha256": index_manifests["dense_index_manifest.json"],
            "revision": standard["frozen_config"]["dense"]["revision"],
            "query_instruction": "Represent this sentence for searching relevant passages: ",
        },
        "answer_model": {
            "model": "Qwen/Qwen3-8B-GGUF",
            "revision": "6a569868d07d3bd59e8b97fb001bf8c0b254bb20",
            "file": "Qwen3-8B-Q4_K_M.gguf",
            "bytes": MODEL_BYTES,
            "sha256": MODEL_SHA256,
            "temperature": 0.0,
            "reasoning": "disabled",
            "max_output_tokens": 256,
            "reader_calls": 180,
            "bridge_calls": 60,
            "retry_on_error": False,
            "endpoint_binding": "loopback_only",
            "stateless_request": True,
            "prompt_cache": False,
            "llama_cpp_version": old_lock["answer_model"]["llama_cpp_version"],
        },
        "reader_prompt_sha256": _sha256_bytes((private_root / "reader_prompt.txt").read_bytes()),
        "reader_schema_sha256": _canonical_sha256(READER_SCHEMA),
        "reader_evidence_context": "top_five_fused_chunks_in_rank_order",
        "strong_generator": {
            "method": "LameR-MV",
            "prompt_template_sha256": bridge_prompt_sha256,
            "calls_per_case": 1,
            "feedback_depth": 10,
            "input_boundary": "current_question_and_same_query_bm25_top10_only",
            "fallback": "original_query_on_empty_or_truncated_bridge",
        },
        "upstream": {
            "repository": "R2MED/R2MED",
            "commit": PINNED_UPSTREAM_COMMIT,
            "prompt_family_mapping": UPSTREAM_PROMPT_FAMILY,
            "source_file_sha256": EXPECTED_UPSTREAM_FILES,
            "lamer_medqa_prompt_sha256": bridge_prompt_sha256,
        },
        "action_profiles": {
            "manifest_sha256": _sha256_file(profiles_path),
            "STANDARD": STANDARD_PROFILE_HASH,
            "STRONG": STRONG_PROFILE_HASH,
        },
        "python_runtime": python_runtime,
        "execution_order": "sorted_case_id_then_OFF_STANDARD_STRONG",
        "scorer_version": SCORER_VERSION,
        "scorer_module_sha256": scorer_sha256,
        "scorer_function_sha256": scorer_function_sha256,
        "scoring_contract_sha256": _canonical_sha256(contract),
        "scoring_contract": contract,
        "primary_quality": {
            "T0": "state_score",
            "T1": "0.75*guideline_content_score+0.25*grounding_score",
            "T2": "0.50*state_score+0.25*guideline_content_score+0.25*grounding_score",
        },
        "content_only_quality": {
            "T0": "state_score",
            "T1": "guideline_content_score",
            "T2": "0.50*state_score+0.50*guideline_content_score",
        },
        "source_cohort": "202607_only",
        "202608_opened": False,
        "model_calls": 0,
        "retrieval_calls": 0,
        "outcomes_or_oracle_action_created": False,
        "locked_code": code_records,
    }
    lock["counterfactual_lock_sha256"] = _canonical_sha256(lock)
    audit = {
        "schema_version": "rag-e5-e5b2-preflight-audit-v1",
        "task_set_changed": False,
        "question_templates_changed": False,
        "state_packet_set_changed": False,
        "teacher_targets_changed": False,
        "rubric_anchors_changed": False,
        "reader_prompt_or_schema_changed": False,
        "retrieval_profiles_changed": False,
        "external_corpus_identity_changed": False,
        "model_identity_changed": False,
        "scorer_version": SCORER_VERSION,
        "scorer_module_sha256": scorer_sha256,
        "scorer_function_sha256": scorer_function_sha256,
        "counterfactual_lock_sha256": lock["counterfactual_lock_sha256"],
        "cases": 60,
        "arms": 180,
        "reader_calls": 180,
        "bridge_calls": 60,
        "maximum_model_calls": 240,
        "model_calls": 0,
        "retrieval_calls": 0,
        "outcomes_created": False,
        "202608_opened": False,
        "preflight": "PASS",
    }
    _write_json(LOCK_PATH, lock)
    _write_json(CONTRACT_PATH, contract)
    _write_json(ROOT / "runs/rag_e5/e5b2_preflight_audit.json", audit)
    return {"lock": lock, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    args = parser.parse_args()
    result = freeze_protocol(
        private_root=args.private_root,
        corpus_root=args.corpus_root,
        index_root=args.index_root,
        model_root=args.model_root,
    )
    print(json.dumps(result["audit"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
