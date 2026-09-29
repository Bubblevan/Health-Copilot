"""Build exact-flat BGE-large embedding indexes and merge A3 qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTERNAL_ROOT = Path("D:/MyLab/Jianli/external/rag_e5")
DEFAULT_MODEL_ROOT = Path("E:/Health-Copilot-Models/models/bge-large-en-v1.5")
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DENSE_DEPTH = 100
GUIDELINE_SMOKE_QUERIES = (
    ("Guideline for the pharmacological treatment of hypertension in adults", "who-hypertension-pharmacological-2021"),
    ("blood pressure threshold for initiation of pharmacological treatment", "who-hypertension-pharmacological-2021"),
    ("laboratory testing before and during pharmacological treatment", "who-hypertension-pharmacological-2021"),
    ("first-line drug classes for adults with hypertension", "who-hypertension-pharmacological-2021"),
    ("WHO guidelines on physical activity and sedentary behaviour", "who-physical-activity-sedentary-2020"),
    ("physical activity recommendation for adults aged 18 to 64", "who-physical-activity-sedentary-2020"),
    ("sedentary behaviour recommendation for older adults aged 65 and older", "who-physical-activity-sedentary-2020"),
    ("physical activity recommendations for pregnant and postpartum women", "who-physical-activity-sedentary-2020"),
    ("physical activity recommendation for adults living with disability", "who-physical-activity-sedentary-2020"),
    ("Total fat intake for the prevention of unhealthy weight gain in adults and children", "who-total-fat-weight-gain-2023"),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: Any) -> str:
    data = _canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256(data)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _smoke_queries(view_id: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    if view_id == "GUIDELINE_ONLY":
        return [
            {"query_id": f"guideline-{i:02d}", "query": query, "expected_source_id": source}
            for i, (query, source) in enumerate(GUIDELINE_SMOKE_QUERIES, start=1)
        ]
    public = sorted(
        (row for row in chunks if row["source_family"] == "public_health"),
        key=lambda row: row["source_id"],
    )
    if view_id == "PUBLIC_HEALTH_ONLY":
        return [
            {"query_id": f"public-health-{i:02d}", "query": " ".join(row["section_path"]), "expected_source_id": row["source_id"]}
            for i, row in enumerate(public[:10], start=1)
        ]
    mixed = [
        {"query_id": f"combined-guideline-{i:02d}", "query": query, "expected_source_id": source}
        for i, (query, source) in enumerate(GUIDELINE_SMOKE_QUERIES[:5], start=1)
    ]
    mixed.extend(
        {"query_id": f"combined-public-{i:02d}", "query": " ".join(row["section_path"]), "expected_source_id": row["source_id"]}
        for i, row in enumerate(public[:5], start=1)
    )
    return mixed


@torch.inference_mode()
def _embed_texts(
    texts: list[str], *, tokenizer: Any, model: Any, prefix: str = "", batch_size: int = 12
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for offset in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[offset : offset + batch_size]]
        encoded = tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
        if encoded["input_ids"].shape[1] > 512:
            raise ValueError("BGE input exceeds the frozen 512-token limit")
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        output = model(**encoded)
        cls_vector = output.last_hidden_state[:, 0, :].float()
        normalized = torch.nn.functional.normalize(cls_vector, p=2, dim=-1)
        vectors.append(normalized.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 1024), dtype=np.float32)


def build(*, external_root: Path, model_root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the approved GPU-based A3 index build")
    corpus_path = external_root / "e5a3/corpus_build_report.json"
    bm25_report_path = external_root / "e5a3/bm25_qualification_report.json"
    corpus_report = json.loads(corpus_path.read_text(encoding="utf-8"))
    bm25_report = json.loads(bm25_report_path.read_text(encoding="utf-8"))
    if bm25_report.get("external_corpus_identity") != corpus_report["external_corpus_identity"]:
        raise ValueError("BM25 qualification report is bound to a different corpus identity")

    tokenizer = AutoTokenizer.from_pretrained(str(model_root), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_root), local_files_only=True)
    model.eval()
    model.to("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    weight_rows = [
        {"file": path.relative_to(model_root).as_posix(), "sha256": _file_sha256(path)}
        for path in sorted(model_root.rglob("*.safetensors"))
    ]
    if len(weight_rows) != 1 or weight_rows[0]["file"] != "model.safetensors":
        raise ValueError("the frozen BGE profile requires the single-file model.safetensors artifact")
    weights_sha = weight_rows[0]["sha256"]
    expected_weights_sha = "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7"
    if weights_sha != expected_weights_sha:
        raise ValueError("local BGE-large weights do not match the frozen profile SHA-256")
    dense_config = {
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "embedding_precision": "float32",
        "pooling": "last_hidden_state CLS token",
        "normalized": True,
        "max_length": 512,
        "query_instruction": BGE_QUERY_INSTRUCTION,
        "document_instruction": "none",
        "model_weight_files": weight_rows,
    }

    build_root = external_root / "e5a3/indexes"
    bm25_views = {row["corpus_view_id"]: row for row in bm25_report["views"]}
    dense_views: list[dict[str, Any]] = []
    for view in corpus_report["views"]:
        view_id = view["corpus_view_id"]
        bm25_view = bm25_views[view_id]
        if bm25_view["corpus_sha256"] != view["corpus_sha256"]:
            raise ValueError(f"BM25 index and corpus view differ: {view_id}")
        chunks = _read_jsonl(Path(view["chunks_jsonl_path"]))
        documents = _read_jsonl(Path(view["search_documents_jsonl_path"]))
        doc_ids = [row["id"] for row in documents]
        doc_order_sha = _sha256(_canonical_json(doc_ids))
        if doc_order_sha != json.loads(Path(bm25_view["bm25_index_manifest_path"]).read_text(encoding="utf-8"))["document_order_sha256"]:
            raise ValueError(f"BM25/dense document order mismatch: {view_id}")
        queries = _smoke_queries(view_id, chunks)
        doc_contents = [row["contents"] for row in documents]
        doc_vectors = _embed_texts(doc_contents, tokenizer=tokenizer, model=model)
        doc_vectors_cuda = torch.from_numpy(doc_vectors).to("cuda")
        directory = build_root / view_id.lower()
        directory.mkdir(parents=True, exist_ok=True)
        vector_path = directory / "bge_large_embeddings.npy"
        np.save(vector_path, doc_vectors, allow_pickle=False)
        vector_sha = _file_sha256(vector_path)

        chunk_by_id = {row["chunk_id"]: row for row in chunks}
        rankings: list[dict[str, Any]] = []
        hit_count = 0
        for query in queries:
            query_vector = _embed_texts(
                [query["query"]], tokenizer=tokenizer, model=model,
                prefix=BGE_QUERY_INSTRUCTION, batch_size=1,
            )[0]
            query_vector_cuda = torch.from_numpy(query_vector).to("cuda")
            scores = torch.mv(doc_vectors_cuda, query_vector_cuda).cpu().tolist()
            order = sorted(range(len(doc_ids)), key=lambda index: (-scores[index], doc_ids[index]))
            top100 = [
                {"chunk_id": doc_ids[index], "score": scores[index]}
                for index in order[:DENSE_DEPTH]
            ]
            top10_sources = [chunk_by_id[row["chunk_id"]]["source_id"] for row in top100[:10]]
            passed = query["expected_source_id"] in top10_sources
            hit_count += int(passed)
            rankings.append(
                {
                    **query,
                    "expected_source_hit_at_10": passed,
                    "top10_source_ids": top10_sources,
                    "top100": top100,
                }
            )
        rankings_sha = _sha256(_canonical_json(rankings))
        dense_identity = _sha256(
            _canonical_json(
                {
                    "corpus_sha256": view["corpus_sha256"],
                    "model": "BAAI/bge-large-en-v1.5",
                    "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
                    "weights_sha256": weights_sha,
                    "model_config": dense_config,
                    "document_order_sha256": doc_order_sha,
                    "embedding_matrix_sha256": vector_sha,
                }
            )
        )
        dense_manifest = {
            "schema_version": "rag-e5-dense-index-v1",
            "corpus_view_id": view_id,
            "corpus_sha256": view["corpus_sha256"],
            "source_families": view["source_families"],
            "model": "BAAI/bge-large-en-v1.5",
            "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
            "weights_sha256": weights_sha,
            "embedding_dimension": int(doc_vectors.shape[1]),
            "document_order_sha256": doc_order_sha,
            "embedding_matrix_path": str(vector_path),
            "embedding_matrix_sha256": vector_sha,
            "document_count": len(doc_ids),
            "top_k": DENSE_DEPTH,
            "search_type": "exact flat cosine over normalized vectors",
            **dense_config,
            "smoke_query_count": len(rankings),
            "smoke_source_hit_at_10": hit_count,
            "smoke_rankings_sha256": rankings_sha,
            "dense_index_identity": dense_identity,
        }
        dense_manifest_path = directory / "dense_index_manifest.json"
        dense_manifest_sha = _write_json(dense_manifest_path, dense_manifest)
        dense_smoke_path = directory / "dense_source_title_smoke.json"
        dense_smoke_sha = _write_json(
            dense_smoke_path,
            {
                "schema_version": "rag-e5-dense-source-smoke-v1",
                "corpus_view_id": view_id,
                "corpus_sha256": view["corpus_sha256"],
                "query_count": len(queries),
                "hits_at_10": hit_count,
                "pass": hit_count == len(queries),
                "rankings": rankings,
            },
        )
        bm25_smoke_path = Path(bm25_view["bm25_smoke_path"])
        if _file_sha256(bm25_smoke_path) != bm25_view["bm25_smoke_sha256"]:
            raise ValueError(f"BM25 smoke report SHA mismatch: {view_id}")
        bm25_smoke = json.loads(bm25_smoke_path.read_text(encoding="utf-8"))
        if bm25_smoke["query_count"] != len(queries):
            raise ValueError(f"BM25/dense source smoke query counts differ: {view_id}")
        combined_smoke_path = directory / "source_title_smoke.json"
        combined_smoke_sha = _write_json(
            combined_smoke_path,
            {
                "schema_version": "rag-e5-source-smoke-v1",
                "corpus_view_id": view_id,
                "corpus_sha256": view["corpus_sha256"],
                "query_count_per_retriever": len(queries),
                "bm25": bm25_smoke,
                "dense": json.loads(dense_smoke_path.read_text(encoding="utf-8")),
            },
        )
        dense_views.append(
            {
                **bm25_view,
                "dense_index_path": str(vector_path),
                "dense_index_manifest_path": str(dense_manifest_path),
                "dense_index_manifest_sha256": dense_manifest_sha,
                "dense_index_identity": dense_identity,
                "dense_smoke_hits_at_10": hit_count,
                "dense_smoke_query_count": len(queries),
                "dense_smoke_path": str(dense_smoke_path),
                "dense_smoke_sha256": dense_smoke_sha,
                "smoke_query_count": len(queries),
                "smoke_report_path": str(combined_smoke_path),
                "smoke_report_sha256": combined_smoke_sha,
                "qualification_pass": bm25_view["bm25_smoke_hits_at_10"] == 10 and hit_count == 10,
            }
        )

    report = {
        "schema_version": "rag-e5-e5a3-index-qualification-v1",
        "stage": "E5-A3-index-qualification",
        "external_corpus_identity": corpus_report["external_corpus_identity"],
        "source_manifest_sha256": corpus_report["source_manifest_sha256"],
        "owner_decision_record_sha256": corpus_report["owner_decision_record_sha256"],
        "bm25_profile": bm25_report["bm25_profile"],
        "dense_profile": {
            "model": "BAAI/bge-large-en-v1.5",
            "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
            "weights_sha256": weights_sha,
            "dimension": 1024,
            "max_length": 512,
            "pooling": "CLS",
            "normalized": True,
            "query_instruction": BGE_QUERY_INSTRUCTION,
            "document_instruction": None,
            "search_type": "exact flat cosine over normalized vectors",
            **dense_config,
        },
        "views": dense_views,
        "bm25_indexes_ready": bm25_report["bm25_indexes_ready"],
        "dense_indexes_ready": all(row["dense_smoke_hits_at_10"] == 10 for row in dense_views),
        "source_title_smoke_pass": all(row["qualification_pass"] for row in dense_views),
        "index_output_root": str(build_root),
    }
    _write_json(external_root / "e5a3/index_qualification_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    args = parser.parse_args()
    result = build(external_root=args.external_root, model_root=args.model_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
