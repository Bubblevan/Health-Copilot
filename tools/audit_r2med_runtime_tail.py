"""Probe R2MED PMC-Clinical DEV runtime sensitivity without writing source text."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.e1_2_r2med_metrics import query_metrics
from tools import run_e1_2_r2med_retrieval as hc_runner

DEFAULT_SCRATCH = Path(r"E:\Health-Copilot-E1.2")
DEFAULT_MODEL_ROOT = Path(r"D:\MyLab\Jianli\models")
DEFAULT_OUTPUT_ROOT = DEFAULT_SCRATCH / "parity" / "runtime-tail"
EXPECTED_SUBSET = "PMC-Clinical"
EXPECTED_WIRING = "ARTICLE_ARTICLE"
MAX_LENGTH = 512
RANKING_DEPTH = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_article_model(model_root: Path, torch: Any, attention: str | None) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer, BertConfig

    model_dir = model_root / "MedCPT-Article-Encoder"
    weights = model_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"MedCPT Article Encoder weights missing: {weights}")
    config = BertConfig.from_pretrained(model_dir, local_files_only=True)
    options: dict[str, Any] = {
        "config": config,
        "local_files_only": True,
        "use_safetensors": True,
    }
    if attention is not None:
        options["attn_implementation"] = attention
    model = AutoModel.from_pretrained(model_dir, **options)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model


def _encode(
    rows: list[str],
    *,
    tokenizer: Any,
    model: Any,
    torch: Any,
    batch_size: int,
    document_pairs: bool,
    progress_label: str,
) -> np.ndarray:
    device = next(model.parameters()).device
    output = np.empty((len(rows), int(model.config.hidden_size)), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        inputs: Any = [("", text) for text in chunk] if document_pairs else chunk
        tokens = tokenizer(
            inputs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=MAX_LENGTH,
        ).to(device)
        with torch.inference_mode():
            vectors = model(**tokens).last_hidden_state[:, 0, :]
        output[start : start + len(chunk)] = vectors.detach().to(
            dtype=torch.float32, device="cpu"
        ).numpy()
        done = start + len(chunk)
        if done == len(rows) or done % max(batch_size * 250, 1) == 0:
            print(f"{progress_label}: {done}/{len(rows)}", flush=True)
    return output


def encode_run(
    data: dict[str, Any],
    *,
    tokenizer: Any,
    model: Any,
    torch: Any,
    batch_size: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    query_vectors = _encode(
        [row["text"] for row in data["queries"]],
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        batch_size=batch_size,
        document_pairs=False,
        progress_label=f"{label} queries",
    )
    document_vectors = _encode(
        [row["text"] for row in data["documents"]],
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        batch_size=batch_size,
        document_pairs=True,
        progress_label=f"{label} documents",
    )
    return query_vectors, document_vectors


def rank_and_score(
    query_vectors: np.ndarray,
    document_vectors: np.ndarray,
    doc_ids: list[str],
    qrels: dict[str, dict[str, int]],
    query_ids: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    document_norms = np.linalg.norm(document_vectors, axis=1).astype(np.float32)
    if np.any(document_norms <= 1e-12):
        raise ValueError("zero document embedding encountered")
    doc_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    ranked: dict[str, Any] = {}
    score_rows = np.empty((len(query_ids), len(doc_ids)), dtype=np.float32)
    for row_index, (query_vector, query_id) in enumerate(zip(query_vectors, query_ids, strict=True)):
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm <= 1e-12:
            raise ValueError("zero query embedding encountered")
        scores = np.asarray(document_vectors @ query_vector, dtype=np.float32)
        scores = scores / np.float32(query_norm)
        scores = scores / document_norms
        score_rows[row_index] = scores
        depth = min(RANKING_DEPTH, len(doc_ids))
        indices = np.argpartition(scores, len(doc_ids) - depth)[len(doc_ids) - depth :]
        indices = sorted(indices.tolist(), key=lambda idx: (-float(scores[idx]), idx))
        top_rows = [(doc_ids[idx], float(scores[idx])) for idx in indices]
        metrics = query_metrics([doc_id for doc_id, _ in top_rows], qrels[query_id])
        relevant_ranks: dict[str, int] = {}
        for relevant_id, relevance in qrels[query_id].items():
            if relevance <= 0:
                continue
            target_index = doc_index[relevant_id]
            target_score = scores[target_index]
            greater = int(np.count_nonzero(scores > target_score))
            tied_before = int(np.count_nonzero(scores[:target_index] == target_score))
            relevant_ranks[relevant_id] = greater + tied_before + 1
        ranked[query_id] = {
            "top": top_rows,
            "relevant_ranks": relevant_ranks,
            **metrics,
        }
    return ranked, score_rows


def comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    query_ids: list[str],
    doc_ids: list[str],
    label: str,
) -> dict[str, Any]:
    query_index = {query_id: idx for idx, query_id in enumerate(query_ids)}
    doc_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    changed_queries: list[dict[str, Any]] = []
    relevant_pair_changes = 0
    top10_changed = 0
    all_rank_changed = 0

    for query_id in query_ids:
        left = baseline[query_id]
        right = candidate[query_id]
        left_top = left["top"]
        right_top = right["top"]
        top10_diff = [row[0] for row in left_top[:10]] != [row[0] for row in right_top[:10]]
        top100_diff = [row[0] for row in left_top] != [row[0] for row in right_top]
        top10_changed += int(top10_diff)
        all_rank_changed += int(top100_diff)
        relevant_changes = []
        for doc_id, baseline_rank in left["relevant_ranks"].items():
            candidate_rank = right["relevant_ranks"][doc_id]
            if baseline_rank != candidate_rank:
                relevant_pair_changes += 1
                idx = doc_index[doc_id]
                relevant_changes.append(
                    {
                        "doc_id": doc_id,
                        "baseline_rank": baseline_rank,
                        "candidate_rank": candidate_rank,
                        "baseline_score": float(baseline_scores[query_index[query_id], idx]),
                        "candidate_score": float(candidate_scores[query_index[query_id], idx]),
                    }
                )
        if top100_diff or relevant_changes:
            changed_ids = {row[0] for row in left_top[:10]} ^ {
                row[0] for row in right_top[:10]
            }
            left_scores = {doc_id: score for doc_id, score in left_top[:10]}
            right_scores = {doc_id: score for doc_id, score in right_top[:10]}
            cutoff_left = left_top[9][1] - left_top[10][1] if len(left_top) > 10 else None
            cutoff_right = right_top[9][1] - right_top[10][1] if len(right_top) > 10 else None
            changed_queries.append(
                {
                    "query_id": query_id,
                    "top100_order_changed": top100_diff,
                    "top10_changed": top10_diff,
                    "top10_changed_doc_score_margins": [
                        {
                            "doc_id": doc_id,
                            "baseline_rank": next(
                                (i for i, row in enumerate(left_top[:10], 1) if row[0] == doc_id), None
                            ),
                            "candidate_rank": next(
                                (i for i, row in enumerate(right_top[:10], 1) if row[0] == doc_id), None
                            ),
                            "baseline_score": left_scores.get(doc_id),
                            "candidate_score": right_scores.get(doc_id),
                        }
                        for doc_id in sorted(changed_ids)
                    ],
                    "relevant_doc_rank_changes": relevant_changes,
                    "baseline_top10_cutoff_margin": cutoff_left,
                    "candidate_top10_cutoff_margin": cutoff_right,
                }
            )

    deltas = np.abs(candidate_scores - baseline_scores)
    metric_deltas: dict[str, float] = {}
    for metric in ("ndcg@10", "mrr@10"):
        metric_deltas[metric] = float(
            np.mean([candidate[qid][metric] - baseline[qid][metric] for qid in query_ids])
        )
    return {
        "compared_to": "batch32_default_attention",
        "label": label,
        "changed_query_count_definition": "queries whose top-100 order changed",
        "changed_query_count": all_rank_changed,
        "top10_changed_query_count": top10_changed,
        "relevant_doc_rank_changed_count": relevant_pair_changes,
        "max_retrieval_score_abs_delta": float(deltas.max(initial=0.0)),
        "mean_metric_delta": metric_deltas,
        "changed_queries": changed_queries,
    }


def runtime_info(torch: Any, transformers: Any, model: Any) -> dict[str, Any]:
    device_info: dict[str, Any] = {"name": None, "total_memory_bytes": None}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device_info = {
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    driver = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
        driver = result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cuda_driver": driver,
        "transformers": str(transformers.__version__),
        "sentence_transformers": importlib.metadata.version("sentence-transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "safetensors": importlib.metadata.version("safetensors"),
        "gpu": device_info,
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "dtype": str(next(model.parameters()).dtype),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def run(args: argparse.Namespace) -> Path:
    import torch
    import transformers

    _config, source_manifest, config_hash, source_hash = hc_runner.load_protocol()
    entries = hc_runner.dataset_entries(source_manifest, "DEV")
    matches = [entry for entry in entries if entry["name"] == EXPECTED_SUBSET]
    if len(matches) != 1 or "test" in matches[0]["directory"].lower():
        raise ValueError("refusing to continue unless exactly the PMC-Clinical DEV source is selected")
    entry = matches[0]
    data = hc_runner.load_dataset(entry, args.scratch_root / "sources")
    queries = data["queries"]
    doc_ids = [row["id"] for row in data["documents"]]
    query_ids = [row["id"] for row in queries]
    outdir = args.output_root / datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")
    outdir.mkdir(parents=True, exist_ok=False)

    runs: dict[str, dict[str, Any]] = {}
    vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    base_runtime: dict[str, Any] | None = None
    execution_records: list[dict[str, Any]] = []

    def one_run(label: str, batch_size: int, attention: str | None) -> None:
        nonlocal base_runtime
        started = time.perf_counter()
        tokenizer, model = load_article_model(args.model_root, torch, attention)
        if base_runtime is None:
            base_runtime = runtime_info(torch, transformers, model)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        q_vectors, d_vectors = encode_run(
            data,
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            batch_size=batch_size,
            label=label,
        )
        ranked, scores = rank_and_score(
            q_vectors,
            d_vectors,
            doc_ids,
            data["qrels"],
            query_ids,
        )
        vectors[label] = (q_vectors, d_vectors)
        runs[label] = ranked
        execution_records.append(
            {
                "label": label,
                "batch_size": batch_size,
                "attention_requested": attention or "default",
                "attention_actual": getattr(model.config, "_attn_implementation", None),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
                ),
                "query_count": len(query_ids),
                "document_count": len(doc_ids),
                "metrics": {
                    "ndcg@10": float(np.mean([ranked[qid]["ndcg@10"] for qid in query_ids])),
                    "mrr@10": float(np.mean([ranked[qid]["mrr@10"] for qid in query_ids])),
                },
            }
        )
        if label == "batch32_default":
            vectors["scores_batch32_default"] = (scores, np.empty((0,), dtype=np.float32))
        del model, tokenizer, q_vectors, d_vectors, ranked, scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    failures: list[dict[str, str]] = []
    for batch_size in (16, 32, 64):
        label = f"batch{batch_size}_default"
        try:
            one_run(label, batch_size, None)
        except torch.cuda.OutOfMemoryError as exc:
            failures.append({"label": label, "failure": "CUDA_OOM", "detail": str(exc)[:500]})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"{label}: CUDA OOM; stopping batch-size escalation", flush=True)
            break
    if "batch32_default" not in runs:
        raise RuntimeError("batch32 default baseline did not complete")

    baseline_vectors = vectors["batch32_default"]
    baseline_score_matrix = vectors["scores_batch32_default"][0]
    comparisons: list[dict[str, Any]] = []
    for label in ("batch16_default", "batch64_default"):
        if label not in runs:
            continue
        q_vec, d_vec = vectors[label]
        q_base, d_base = baseline_vectors
        comparisons.append(
            {
                **comparison(
                    runs["batch32_default"],
                    runs[label],
                    baseline_score_matrix,
                    vectors.get(f"scores_{label}", (None, None))[0]
                    if f"scores_{label}" in vectors
                    else score_matrix_for(runs[label], q_vec, d_vec, doc_ids, data["qrels"], query_ids),
                    query_ids,
                    doc_ids,
                    label,
                ),
                "max_embedding_abs_delta": {
                    "query": float(np.max(np.abs(q_vec - q_base), initial=0.0)),
                    "document": float(np.max(np.abs(d_vec - d_base), initial=0.0)),
                },
            }
        )

    try:
        one_run("batch32_eager", 32, "eager")
    except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError, TypeError) as exc:
        failures.append({"label": "batch32_eager", "failure": type(exc).__name__, "detail": str(exc)[:500]})
    if "batch32_eager" in runs:
        q_vec, d_vec = vectors["batch32_eager"]
        eager_score_matrix = score_matrix_for(
            runs["batch32_eager"], q_vec, d_vec, doc_ids, data["qrels"], query_ids
        )
        comparisons.append(
            {
                **comparison(
                    runs["batch32_default"],
                    runs["batch32_eager"],
                    baseline_score_matrix,
                    eager_score_matrix,
                    query_ids,
                    doc_ids,
                    "batch32_eager",
                ),
                "max_embedding_abs_delta": {
                    "query": float(np.max(np.abs(q_vec - baseline_vectors[0]), initial=0.0)),
                    "document": float(np.max(np.abs(d_vec - baseline_vectors[1]), initial=0.0)),
                },
            }
        )

    output = {
        "schema_version": "r2med-runtime-tail-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "partition": "DEV",
        "test_opened": False,
        "subset": EXPECTED_SUBSET,
        "wiring": EXPECTED_WIRING,
        "source_identity": {
            "frozen_config_sha256": config_hash,
            "source_manifest_sha256": source_hash,
            "corpus_sha256": entry["files"]["corpus.jsonl"]["sha256"],
            "query_sha256": entry["files"]["query.jsonl"]["sha256"],
            "qrels_sha256": entry["files"]["qrels.jsonl"]["sha256"],
            "article_encoder_sha256": sha256_file(args.model_root / "MedCPT-Article-Encoder" / "model.safetensors"),
        },
        "runtime": base_runtime,
        "protocol": {
            "query_encoder": "MedCPT-Article-Encoder",
            "document_encoder": "MedCPT-Article-Encoder",
            "query_input": "raw query text",
            "document_input": "empty title plus document text as paired tokenizer input",
            "query_max_length": MAX_LENGTH,
            "document_max_length": MAX_LENGTH,
            "pooling": "CLS",
            "similarity": "cosine (BEIR-style; normalized by vector norms)",
            "ranking_depth": RANKING_DEPTH,
            "score_tie_break": "score descending, corpus row order ascending",
        },
        "runs": execution_records,
        "comparisons_to_batch32_default": comparisons,
        "failures": failures,
        "runtime_numerical_sensitivity_confirmed": any(
            row["changed_query_count"] > 0
            or row["top10_changed_query_count"] > 0
            or row["relevant_doc_rank_changed_count"] > 0
            or any(abs(delta) > 0 for delta in row["mean_metric_delta"].values())
            for row in comparisons
        ),
        "notes": [
            "Only PMC-Clinical DEV was read; no TEST partition was loaded.",
            "Query text, corpus text, and qrel contents are not written to this artifact.",
            "Metrics are reported over PMC-Clinical DEV only; no cross-subset aggregate is implied.",
        ],
    }
    write_json(outdir / "runtime_tail_audit.json", output)
    return outdir


def score_matrix_for(
    _ranked: dict[str, Any],
    query_vectors: np.ndarray,
    document_vectors: np.ndarray,
    doc_ids: list[str],
    qrels: dict[str, dict[str, int]],
    query_ids: list[str],
) -> np.ndarray:
    _result, scores = rank_and_score(query_vectors, document_vectors, doc_ids, qrels, query_ids)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    output = run(args)
    print(f"DEV-only runtime-tail audit written: {output}", flush=True)


if __name__ == "__main__":
    main()
