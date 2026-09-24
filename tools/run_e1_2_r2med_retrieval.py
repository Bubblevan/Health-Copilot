"""Run the predeclared E1.2 retrieval arms on R2MED without emitting source text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.e1_2_protocol import assert_committed_artifact
from eval.e1_2_r2med_metrics import query_metrics, summarize_metrics
from tools.run_medcpt_textbooks_retrieval import load_models as load_medcpt_models

CONFIG_PATH = ROOT / "runs/e1_2/frozen_test_config.json"
SOURCE_MANIFEST_PATH = ROOT / "runs/e1_2/r2med_source_manifest.json"
DEV_LOCK_PATH = ROOT / "runs/e1_2/r2med_dev_selection.json"
DEFAULT_SCRATCH = Path(r"E:\Health-Copilot-E1.2")
DEFAULT_MEDCPT_ROOT = Path(r"D:\MyLab\Jianli\models")
DEFAULT_BGE_ROOT = DEFAULT_SCRATCH / "models/bge-base-en-v1.5"
ARMS = ("bm25", "bge_dense", "medcpt_dense", "rrf_medcpt_rerank")
FIXED_ARMS = ARMS[:3]
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60
CANDIDATE_DEPTH = 100
TOP_K = 10
MEDCPT_QUERY_MAX_LENGTH = 64
MEDCPT_DOCUMENT_MAX_LENGTH = 512
MEDCPT_RERANKER_MAX_LENGTH = 512
EMBEDDING_BATCH_SIZE = 32
RERANKER_BATCH_SIZE = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary artifact already exists: {temporary}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path.name}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def load_protocol() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    config_bytes = CONFIG_PATH.read_bytes()
    source_bytes = SOURCE_MANIFEST_PATH.read_bytes()
    config = json.loads(config_bytes)
    source_manifest = json.loads(source_bytes)
    if config.get("status") != "FROZEN_BEFORE_ANY_E1_2_ANSWER_CALLS":
        raise ValueError("E1.2 configuration is not frozen")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    expected_source_hash = config["track_a_retrieval"]["source_manifest_sha256"]
    if expected_source_hash != source_hash:
        raise ValueError("R2MED source manifest does not match the frozen configuration")
    expected_parameters = {
        "top_k": TOP_K,
        "candidate_depth": CANDIDATE_DEPTH,
        "rrf_k": RRF_K,
        "bge_query_prefix": BGE_QUERY_PREFIX,
        "bge_max_sequence_length": 512,
        "medcpt_query_max_sequence_length": MEDCPT_QUERY_MAX_LENGTH,
        "medcpt_document_max_sequence_length": MEDCPT_DOCUMENT_MAX_LENGTH,
        "medcpt_reranker_max_sequence_length": MEDCPT_RERANKER_MAX_LENGTH,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "reranker_batch_size": RERANKER_BATCH_SIZE,
    }
    if config["track_a_retrieval"].get("retrieval_parameters") != expected_parameters:
        raise ValueError("R2MED retrieval parameters differ from the frozen protocol")
    return config, source_manifest, config_hash, source_hash


def dataset_entries(source_manifest: dict[str, Any], partition: str) -> list[dict[str, Any]]:
    field = "dev_subsets" if partition == "DEV" else "test_subsets"
    return list(source_manifest[field])


def verify_source_files(entry: dict[str, Any], source_root: Path) -> dict[str, Path]:
    dataset_dir = source_root / entry["directory"]
    verified: dict[str, Path] = {}
    for name in ("corpus.jsonl", "query.jsonl", "qrels.jsonl"):
        path = dataset_dir / name
        expected = entry["files"][name]
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"R2MED source file is missing or has the wrong size: {entry['name']}/{name}")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"R2MED source file hash mismatch: {entry['name']}/{name}")
        verified[name] = path
    return verified


def load_dataset(entry: dict[str, Any], source_root: Path) -> dict[str, Any]:
    paths = verify_source_files(entry, source_root)
    corpus_rows = read_jsonl(paths["corpus.jsonl"])
    query_rows = read_jsonl(paths["query.jsonl"])
    qrel_rows = read_jsonl(paths["qrels.jsonl"])
    if len(corpus_rows) != entry["corpus_document_count"]:
        raise ValueError(f"corpus row count changed for {entry['name']}")
    if len(query_rows) != entry["query_count"] or len(qrel_rows) != entry["qrels_record_count"]:
        raise ValueError(f"query/qrels row count changed for {entry['name']}")

    documents: list[dict[str, str]] = []
    documents_by_id: dict[str, str] = {}
    duplicate_corpus_rows_removed = 0
    for row in corpus_rows:
        doc_id, text = row.get("id"), row.get("text")
        if not isinstance(doc_id, str) or not doc_id or not isinstance(text, str) or not text:
            raise ValueError(f"invalid corpus schema in {entry['name']}")
        if doc_id in documents_by_id:
            if documents_by_id[doc_id] != text:
                raise ValueError(f"conflicting duplicate corpus ID in {entry['name']}")
            duplicate_corpus_rows_removed += 1
            continue
        documents_by_id[doc_id] = text
        documents.append({"id": doc_id, "text": text})
    doc_ids = set(documents_by_id)

    queries: list[dict[str, str]] = []
    query_ids: set[str] = set()
    for row in query_rows:
        query_id, text = row.get("id"), row.get("text")
        if not isinstance(query_id, str) or not query_id or not isinstance(text, str) or not text:
            raise ValueError(f"invalid query schema in {entry['name']}")
        if query_id in query_ids:
            raise ValueError(f"duplicate query ID in {entry['name']}")
        query_ids.add(query_id)
        queries.append({"id": query_id, "text": text})

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrel_rows:
        query_id, doc_id, score = row.get("q_id"), row.get("p_id"), row.get("score")
        if query_id not in query_ids or doc_id not in doc_ids:
            raise ValueError(f"qrels reference an unknown query or document in {entry['name']}")
        if isinstance(score, bool) or not isinstance(score, int) or score < 0:
            raise ValueError(f"invalid graded relevance score in {entry['name']}")
        if doc_id in qrels[query_id]:
            raise ValueError(f"duplicate qrels pair in {entry['name']}")
        qrels[query_id][doc_id] = score
    if set(qrels) != query_ids:
        raise ValueError(f"queries without any qrels found in {entry['name']}")

    return {
        "entry": entry,
        "paths": paths,
        "documents": documents,
        "queries": queries,
        "qrels": qrels,
        "source_integrity": {
            "corpus_source_rows": len(corpus_rows),
            "unique_document_ids": len(documents),
            "identical_duplicate_rows_removed": duplicate_corpus_rows_removed,
        },
    }


def ensure_index(data: dict[str, Any], index_root: Path) -> tuple[Path, bool]:
    entry = data["entry"]
    index_root.mkdir(parents=True, exist_ok=True)
    index_path = index_root / f"{entry['directory']}.sqlite3"
    corpus_hash = entry["files"]["corpus.jsonl"]["sha256"]
    if index_path.is_file():
        connection = sqlite3.connect(index_path)
        try:
            meta = dict(connection.execute("SELECT key,value FROM index_meta"))
            count = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
            if meta != {"corpus_sha256": corpus_hash, "document_count": str(len(data["documents"]))}:
                raise ValueError(f"existing SQLite index identity mismatch: {index_path.name}")
            if count != len(data["documents"]):
                raise ValueError(f"existing SQLite index document count mismatch: {index_path.name}")
        finally:
            connection.close()
        return index_path, True

    building = index_path.with_name(index_path.name + ".building")
    if building.exists():
        raise FileExistsError(f"incomplete R2MED index exists; inspect before retry: {building}")
    connection = sqlite3.connect(building)
    inserted = 0
    try:
        connection.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, text TEXT NOT NULL)")
        connection.execute(
            "CREATE VIRTUAL TABLE documents_fts USING fts5(id UNINDEXED, text, "
            "tokenize='porter unicode61 remove_diacritics 2')"
        )
        connection.execute("CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("BEGIN")
        batch: list[tuple[str, str]] = []
        for document in data["documents"]:
            batch.append((document["id"], document["text"]))
            if len(batch) >= 1000:
                connection.executemany("INSERT INTO documents VALUES (?,?)", batch)
                connection.executemany("INSERT INTO documents_fts (id,text) VALUES (?,?)", batch)
                inserted += len(batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO documents VALUES (?,?)", batch)
            connection.executemany("INSERT INTO documents_fts (id,text) VALUES (?,?)", batch)
            inserted += len(batch)
        if inserted != len(data["documents"]):
            raise ValueError(f"unexpected indexed document count for {entry['name']}")
        connection.executemany(
            "INSERT INTO index_meta VALUES (?,?)",
            (("corpus_sha256", corpus_hash), ("document_count", str(inserted))),
        )
        connection.commit()
    except Exception:
        connection.close()
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{building}{suffix}")
            if candidate.exists():
                candidate.unlink()
        raise
    connection.close()
    building.replace(index_path)
    return index_path, False


def build_fts_query(query: str) -> str:
    terms = list(dict.fromkeys(token.lower() for token in TOKEN_RE.findall(query)))
    return " OR ".join(f'"{term}"' for term in terms)


def retrieve_bm25(connection: sqlite3.Connection, query: str, depth: int) -> list[tuple[str, float]]:
    fts_query = build_fts_query(query)
    if not fts_query:
        return []
    rows = connection.execute(
        "SELECT id,bm25(documents_fts) FROM documents_fts "
        "WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts) LIMIT ?",
        (fts_query, depth),
    )
    return [(str(doc_id), float(score)) for doc_id, score in rows]


def top_dense(
    query_vectors: np.ndarray, embeddings: np.ndarray, depth: int
) -> list[list[tuple[int, float]]]:
    count = embeddings.shape[0]
    k = min(depth, count)
    output: list[list[tuple[int, float]]] = []
    for query_vector in query_vectors:
        scores = np.asarray(embeddings @ query_vector, dtype=np.float32)
        indices = np.argpartition(scores, count - k)[count - k :]
        indices = sorted(indices.tolist(), key=lambda index: float(scores[index]), reverse=True)
        output.append([(int(index), float(scores[index])) for index in indices])
    return output


def _cache_paths(scratch_root: Path, model: str, entry: dict[str, Any]) -> tuple[Path, Path]:
    directory = scratch_root / "cache" / "r2med" / model / entry["directory"]
    return directory / "document_embeddings.npy", directory / "manifest.json"


def load_or_create_bge_embeddings(
    data: dict[str, Any], model: Any, scratch_root: Path
) -> np.ndarray:
    entry = data["entry"]
    weights_hash = entry_model_hash(data, "bge_model_safetensors_sha256")
    spec = {
        "model": "BAAI/bge-base-en-v1.5",
        "model_sha256": weights_hash,
        "corpus_sha256": entry["files"]["corpus.jsonl"]["sha256"],
        "pooling": "SentenceTransformer model card configuration",
        "normalized": True,
        "document_prefix": "none",
        "maximum_sequence_length": int(model.max_seq_length),
    }
    return _load_or_encode_embeddings(data, model, scratch_root, "bge", spec)


def entry_model_hash(data: dict[str, Any], name: str) -> str:
    del data
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return str(config["track_a_retrieval"]["test_model_hashes"][name])


def _load_or_encode_embeddings(
    data: dict[str, Any], model: Any, scratch_root: Path, model_name: str, spec: dict[str, Any]
) -> np.ndarray:
    entry = data["entry"]
    cache_path, manifest_path = _cache_paths(scratch_root, model_name, entry)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        if not manifest_path.is_file():
            raise ValueError(f"embedding cache is missing its manifest: {cache_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        embeddings = np.load(cache_path, mmap_mode="r")
        if manifest.get("spec") != spec or embeddings.shape != (len(data["documents"]), 768):
            raise ValueError(f"embedding cache identity or shape mismatch: {cache_path}")
        return embeddings
    partial_path = cache_path.with_suffix(".partial.npy")
    if partial_path.exists():
        raise FileExistsError(f"partial embedding cache exists; inspect before retry: {partial_path}")
    embeddings = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.float32, shape=(len(data["documents"]), 768)
    )
    batch_size = EMBEDDING_BATCH_SIZE
    for start in range(0, len(data["documents"]), batch_size):
        texts = [row["text"] for row in data["documents"][start : start + batch_size]]
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=bool(spec["normalized"]),
            show_progress_bar=False,
        )
        if vectors.shape != (len(texts), 768):
            raise ValueError(f"unexpected {model_name} embedding shape")
        embeddings[start : start + len(texts)] = vectors
        if (start // batch_size + 1) % 250 == 0 or start + len(texts) == len(data["documents"]):
            embeddings.flush()
            print(
                f"{entry['name']} {model_name} document embeddings: "
                f"{min(start + len(texts), len(data['documents']))}/{len(data['documents'])}",
                flush=True,
            )
    embeddings.flush()
    del embeddings
    partial_path.replace(cache_path)
    write_json_atomic(
        manifest_path,
        {"spec": spec, "shape": [len(data["documents"]), 768], "created_at_utc": datetime.now(UTC).isoformat()},
    )
    return np.load(cache_path, mmap_mode="r")


def encode_medcpt_queries(queries: list[dict[str, str]], models: dict[str, Any], torch: Any) -> np.ndarray:
    vectors: list[np.ndarray] = []
    tokenizer = models["query_tokenizer"]
    model = models["query_model"]
    for start in range(0, len(queries), EMBEDDING_BATCH_SIZE):
        texts = [row["text"] for row in queries[start : start + EMBEDDING_BATCH_SIZE]]
        tokens = tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=MEDCPT_QUERY_MAX_LENGTH,
        )
        tokens = tokens.to(models["device"])
        with torch.inference_mode():
            encoded = model(**tokens).last_hidden_state[:, 0, :]
        vectors.append(encoded.detach().to(dtype=torch.float32, device="cpu").numpy())
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 768), dtype=np.float32)


def load_or_create_medcpt_embeddings(
    data: dict[str, Any], models: dict[str, Any], scratch_root: Path, torch: Any
) -> np.ndarray:
    entry = data["entry"]
    article_hash = models["model_manifest"]["article_encoder"]["sha256"]
    spec = {
        "model": "MedCPT-Article-Encoder",
        "model_sha256": article_hash,
        "corpus_sha256": entry["files"]["corpus.jsonl"]["sha256"],
        "document_representation": "empty title plus R2MED text as second token sequence",
        "pooling": "CLS vector",
        "normalized": False,
        "maximum_sequence_length": MEDCPT_DOCUMENT_MAX_LENGTH,
    }
    cache_path, manifest_path = _cache_paths(scratch_root, "medcpt", entry)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        if not manifest_path.is_file():
            raise ValueError(f"embedding cache is missing its manifest: {cache_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        embeddings = np.load(cache_path, mmap_mode="r")
        if manifest.get("spec") != spec or embeddings.shape != (len(data["documents"]), 768):
            raise ValueError(f"embedding cache identity or shape mismatch: {cache_path}")
        return embeddings
    partial_path = cache_path.with_suffix(".partial.npy")
    if partial_path.exists():
        raise FileExistsError(f"partial embedding cache exists; inspect before retry: {partial_path}")
    article_model = models["article_model"]
    tokenizer = models["article_tokenizer"]
    device = models["device"]
    embeddings = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.float32, shape=(len(data["documents"]), 768)
    )
    batch_size = 32
    for start in range(0, len(data["documents"]), batch_size):
        batch = data["documents"][start : start + batch_size]
        pairs = [("", row["text"]) for row in batch]
        tokens = tokenizer(
            pairs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=MEDCPT_DOCUMENT_MAX_LENGTH,
        ).to(device)
        with torch.inference_mode():
            encoded = article_model(**tokens).last_hidden_state[:, 0, :]
        vectors = encoded.detach().to(dtype=torch.float32, device="cpu").numpy()
        if vectors.shape != (len(batch), 768):
            raise ValueError("unexpected MedCPT article embedding shape")
        embeddings[start : start + len(batch)] = vectors
        if (start // batch_size + 1) % 250 == 0 or start + len(batch) == len(data["documents"]):
            embeddings.flush()
            print(
                f"{entry['name']} MedCPT document embeddings: "
                f"{min(start + len(batch), len(data['documents']))}/{len(data['documents'])}",
                flush=True,
            )
    embeddings.flush()
    del embeddings
    partial_path.replace(cache_path)
    write_json_atomic(
        manifest_path,
        {"spec": spec, "shape": [len(data["documents"]), 768], "created_at_utc": datetime.now(UTC).isoformat()},
    )
    return np.load(cache_path, mmap_mode="r")


def encode_bge_queries(queries: list[dict[str, str]], model: Any) -> np.ndarray:
    texts = [BGE_QUERY_PREFIX + row["text"] for row in queries]
    return model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def reciprocal_rank_fusion(
    first: list[tuple[str, float]], second: list[tuple[str, float]], *, depth: int = CANDIDATE_DEPTH
) -> list[tuple[str, float]]:
    fused: dict[str, float] = defaultdict(float)
    for ranking in (first, second):
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (RRF_K + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:depth]


def rerank_medcpt(
    query: str,
    candidates: list[tuple[str, float]],
    document_texts: dict[str, str],
    models: dict[str, Any],
    torch: Any,
) -> list[tuple[str, float]]:
    if not candidates:
        return []
    tokenizer = models["cross_tokenizer"]
    model = models["cross_model"]
    device = next(model.parameters()).device
    scored: list[tuple[str, float]] = []
    batch_size = RERANKER_BATCH_SIZE
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        docs = [document_texts[doc_id] for doc_id, _ in batch]
        tokens = tokenizer(
            [query] * len(batch),
            docs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=MEDCPT_RERANKER_MAX_LENGTH,
        ).to(device)
        with torch.inference_mode():
            logits = model(**tokens).logits.reshape(-1).detach().to(dtype=torch.float32, device="cpu")
        scored.extend((doc_id, float(score)) for (doc_id, _), score in zip(batch, logits.tolist(), strict=True))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:TOP_K]


def arm_paths(scratch_root: Path, partition: str, entry: dict[str, Any], arm: str) -> tuple[Path, Path]:
    directory = scratch_root / "results" / "track_a" / partition.lower() / entry["directory"]
    return directory / f"{arm}.jsonl", directory / f"{arm}.manifest.json"


def load_rank_rows(path: Path, expected_identity: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("result_identity") != expected_identity:
            raise ValueError(f"ranked output identity mismatch: {path.name}")
        query_id = str(row["query_id"])
        if query_id in rows:
            raise ValueError(f"duplicate query result in {path.name}")
        rows[query_id] = row
    return rows


def run_dataset(
    *,
    partition: str,
    entry: dict[str, Any],
    arms: tuple[str, ...],
    data: dict[str, Any],
    config_hash: str,
    source_hash: str,
    scratch_root: Path,
    bge_root: Path,
    medcpt_root: Path,
    bge_model: Any | None,
    medcpt_models: dict[str, Any] | None,
    torch: Any | None,
) -> dict[str, dict[str, Any]]:
    index_path, index_reused = ensure_index(data, scratch_root / "index" / "r2med")
    connection = sqlite3.connect(index_path)
    document_texts = {row["id"]: row["text"] for row in data["documents"]}
    queries = data["queries"]
    subset = entry["name"]
    outputs: dict[str, dict[str, Any]] = {}
    base_rankings: dict[str, dict[str, list[tuple[str, float]]]] = {}

    if "rrf_medcpt_rerank" in arms and partition == "DEV":
        base_rankings = load_dev_fixed_rankings(
            partition=partition,
            entry=entry,
            config_hash=config_hash,
            source_hash=source_hash,
            scratch_root=scratch_root,
            expected_query_ids={str(query["id"]) for query in queries},
            medcpt_hashes={
                "query_encoder_sha256": medcpt_models["model_manifest"]["query_encoder"]["sha256"],
                "article_encoder_sha256": medcpt_models["model_manifest"]["article_encoder"]["sha256"],
            },
        )

    need_bm25 = "bm25" in arms or ("rrf_medcpt_rerank" in arms and partition == "TEST")
    if need_bm25:
        bm25_rankings: dict[str, list[tuple[str, float]]] = {}
        bm25_latencies: dict[str, float] = {}
        for query in queries:
            started = time.perf_counter()
            bm25_rankings[query["id"]] = retrieve_bm25(connection, query["text"], CANDIDATE_DEPTH)
            bm25_latencies[query["id"]] = (time.perf_counter() - started) * 1000
        base_rankings["bm25"] = bm25_rankings
        if "bm25" in arms:
            outputs["bm25"] = persist_arm(
                partition=partition,
                entry=entry,
                arm="bm25",
                queries=queries,
                qrels=data["qrels"],
                rankings=bm25_rankings,
                latencies=bm25_latencies,
                config_hash=config_hash,
                source_hash=source_hash,
                scratch_root=scratch_root,
                index_reused=index_reused,
                model_hashes={},
            )

    if "bge_dense" in arms:
        if bge_model is None:
            raise ValueError("BGE model was not loaded")
        embeddings = load_or_create_bge_embeddings(data, bge_model, scratch_root)
        started = time.perf_counter()
        query_vectors = encode_bge_queries(queries, bge_model)
        rankings_by_index = top_dense(query_vectors, embeddings, CANDIDATE_DEPTH)
        elapsed_ms = (time.perf_counter() - started) * 1000
        bge_rankings = {
            query["id"]: [
                (data["documents"][doc_index]["id"], score) for doc_index, score in ranking
            ]
            for query, ranking in zip(queries, rankings_by_index, strict=True)
        }
        base_rankings["bge_dense"] = bge_rankings
        outputs["bge_dense"] = persist_arm(
            partition=partition,
            entry=entry,
            arm="bge_dense",
            queries=queries,
            qrels=data["qrels"],
            rankings=bge_rankings,
            latencies={query["id"]: elapsed_ms / max(len(queries), 1) for query in queries},
            config_hash=config_hash,
            source_hash=source_hash,
            scratch_root=scratch_root,
            index_reused=index_reused,
            model_hashes={"bge_model_safetensors_sha256": entry_model_hash(data, "bge_model_safetensors_sha256")},
        )

    if "medcpt_dense" in arms or ("rrf_medcpt_rerank" in arms and partition == "TEST"):
        if medcpt_models is None or torch is None:
            raise ValueError("MedCPT models were not loaded")
        embeddings = load_or_create_medcpt_embeddings(data, medcpt_models, scratch_root, torch)
        started = time.perf_counter()
        query_vectors = encode_medcpt_queries(queries, medcpt_models, torch)
        rankings_by_index = top_dense(query_vectors, embeddings, CANDIDATE_DEPTH)
        elapsed_ms = (time.perf_counter() - started) * 1000
        medcpt_rankings = {
            query["id"]: [
                (data["documents"][doc_index]["id"], score) for doc_index, score in ranking
            ]
            for query, ranking in zip(queries, rankings_by_index, strict=True)
        }
        base_rankings["medcpt_dense"] = medcpt_rankings
        medcpt_hashes = {
            "query_encoder_sha256": medcpt_models["model_manifest"]["query_encoder"]["sha256"],
            "article_encoder_sha256": medcpt_models["model_manifest"]["article_encoder"]["sha256"],
        }
        if "medcpt_dense" in arms:
            outputs["medcpt_dense"] = persist_arm(
                partition=partition,
                entry=entry,
                arm="medcpt_dense",
                queries=queries,
                qrels=data["qrels"],
                rankings=medcpt_rankings,
                latencies={query["id"]: elapsed_ms / max(len(queries), 1) for query in queries},
                config_hash=config_hash,
                source_hash=source_hash,
                scratch_root=scratch_root,
                index_reused=index_reused,
                model_hashes=medcpt_hashes,
            )

    if "rrf_medcpt_rerank" in arms:
        if medcpt_models is None or torch is None:
            raise ValueError("MedCPT models were not loaded")
        if set(base_rankings) != {"bm25", "medcpt_dense"}:
            raise ValueError("RRF reranking requires BM25 and MedCPT dense top-100 rankings")
        fused: dict[str, list[tuple[str, float]]] = {}
        reranked: dict[str, list[tuple[str, float]]] = {}
        latencies: dict[str, float] = {}
        start_all = time.perf_counter()
        for query in queries:
            query_id = query["id"]
            started = time.perf_counter()
            fused[query_id] = reciprocal_rank_fusion(
                base_rankings["bm25"][query_id], base_rankings["medcpt_dense"][query_id]
            )
            reranked[query_id] = rerank_medcpt(
                query["text"], fused[query_id], document_texts, medcpt_models, torch
            )
            latencies[query_id] = (time.perf_counter() - started) * 1000
        if partition == "DEV":
            rerank_rows = persist_arm(
                partition=partition,
                entry=entry,
                arm="rrf_medcpt_rerank",
                queries=queries,
                qrels=data["qrels"],
                rankings=reranked,
                latencies=latencies,
                config_hash=config_hash,
                source_hash=source_hash,
                scratch_root=scratch_root,
                index_reused=index_reused,
                model_hashes={
                    "query_encoder_sha256": medcpt_models["model_manifest"]["query_encoder"]["sha256"],
                    "article_encoder_sha256": medcpt_models["model_manifest"]["article_encoder"]["sha256"],
                    "cross_encoder_sha256": medcpt_models["model_manifest"]["cross_encoder"]["sha256"],
                },
            )
        else:
            rerank_rows = persist_arm(
                partition=partition,
                entry=entry,
                arm="rrf_medcpt_rerank",
                queries=queries,
                qrels=data["qrels"],
                rankings=reranked,
                latencies=latencies,
                config_hash=config_hash,
                source_hash=source_hash,
                scratch_root=scratch_root,
                index_reused=index_reused,
                model_hashes={
                    "query_encoder_sha256": medcpt_models["model_manifest"]["query_encoder"]["sha256"],
                    "article_encoder_sha256": medcpt_models["model_manifest"]["article_encoder"]["sha256"],
                    "cross_encoder_sha256": medcpt_models["model_manifest"]["cross_encoder"]["sha256"],
                },
            )
        outputs["rrf_medcpt_rerank"] = rerank_rows
        print(
            f"{subset} frozen fusion+rerank elapsed: "
            f"{time.perf_counter() - start_all:.1f}s",
            flush=True,
        )

    connection.close()
    return outputs


def load_dev_fixed_rankings(
    *,
    partition: str,
    entry: dict[str, Any],
    config_hash: str,
    source_hash: str,
    scratch_root: Path,
    expected_query_ids: set[str],
    medcpt_hashes: dict[str, str],
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    if len(expected_query_ids) != int(entry["query_count"]):
        raise ValueError(f"expected query IDs do not match the frozen count for {entry['name']}")
    result: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for arm in ("bm25", "medcpt_dense"):
        result_path, manifest_path = arm_paths(scratch_root, partition, entry, arm)
        if not result_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"run all fixed DEV arms before custom RRF: missing {arm}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_model_hashes = {} if arm == "bm25" else medcpt_hashes
        if (
            manifest.get("status") != "COMPLETED"
            or manifest.get("partition") != partition
            or manifest.get("subset") != entry["name"]
            or manifest.get("arm") != arm
            or manifest.get("query_count") != len(expected_query_ids)
            or manifest.get("config_sha256") != config_hash
            or manifest.get("source_manifest_sha256") != source_hash
            or manifest.get("model_hashes") != expected_model_hashes
        ):
            raise ValueError(f"fixed result identity mismatch for {arm}")
        identity = str(manifest["result_identity"])
        rows = load_rank_rows(result_path, identity)
        if set(rows) != expected_query_ids:
            raise ValueError(f"fixed DEV query IDs do not match {entry['name']} for {arm}")
        for query_id, row in rows.items():
            ranked = row.get("ranked")
            if not isinstance(ranked, list) or len(ranked) < min(CANDIDATE_DEPTH, int(entry["corpus_document_count"])):
                raise ValueError(f"fixed DEV ranking is truncated for {arm} / {query_id}")
            if any(not isinstance(item, list) or len(item) != 2 for item in ranked):
                raise ValueError(f"fixed DEV ranking has an invalid row for {arm} / {query_id}")
            if len({str(item[0]) for item in ranked}) != len(ranked):
                raise ValueError(f"fixed DEV ranking has duplicate document IDs for {arm} / {query_id}")
        result[arm] = {
            query_id: [(str(doc_id), float(score)) for doc_id, score in row["ranked"]]
            for query_id, row in rows.items()
        }
    return result


def persist_arm(
    *,
    partition: str,
    entry: dict[str, Any],
    arm: str,
    queries: list[dict[str, str]],
    qrels: dict[str, dict[str, int]],
    rankings: dict[str, list[tuple[str, float]]],
    latencies: dict[str, float],
    config_hash: str,
    source_hash: str,
    scratch_root: Path,
    index_reused: bool,
    model_hashes: dict[str, str],
) -> dict[str, Any]:
    result_path, manifest_path = arm_paths(scratch_root, partition, entry, arm)
    run_spec = {
        "config_sha256": config_hash,
        "source_manifest_sha256": source_hash,
        "subset": entry["name"],
        "partition": partition,
        "arm": arm,
        "top_k": TOP_K,
        "candidate_depth": CANDIDATE_DEPTH,
        "rrf_k": RRF_K if arm == "rrf_medcpt_rerank" else None,
        "model_hashes": model_hashes,
    }
    identity = hashlib.sha256(
        json.dumps(run_spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {"status": "RUNNING", "result_identity": identity, **run_spec}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("result_identity") != identity:
            raise ValueError(f"existing output belongs to another frozen identity: {manifest_path}")
        if existing.get("status") == "COMPLETED":
            rows = load_rank_rows(result_path, identity)
            if len(rows) != len(queries):
                raise ValueError(f"completed result row count mismatch: {result_path}")
            return summarize_metrics(list(rows.values()))
    elif result_path.exists() and result_path.stat().st_size:
        raise ValueError(f"result file exists without matching manifest: {result_path}")
    else:
        write_json_atomic(manifest_path, manifest)

    existing_rows = load_rank_rows(result_path, identity) if result_path.exists() else {}
    with result_path.open("a", encoding="utf-8") as handle:
        for query in queries:
            query_id = query["id"]
            if query_id in existing_rows:
                continue
            ranking = rankings.get(query_id, [])
            top_rows = ranking[:CANDIDATE_DEPTH]
            metrics = query_metrics([doc_id for doc_id, _ in top_rows], qrels[query_id])
            record = {
                "query_id": query_id,
                "subset": entry["name"],
                "result_identity": identity,
                "ranked": [[doc_id, round(float(score), 8)] for doc_id, score in top_rows],
                **metrics,
                "retrieval_latency_ms": round(float(latencies.get(query_id, 0.0)), 3),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    rows = load_rank_rows(result_path, identity)
    if set(rows) != {query["id"] for query in queries}:
        raise ValueError(f"result query IDs do not match the source set: {result_path}")
    summary = summarize_metrics(list(rows.values()))
    manifest.update(
        {
            "status": "COMPLETED",
            "query_count": len(rows),
            "metrics": summary,
            "index_reused": index_reused,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    write_json_atomic(manifest_path, manifest)
    return summary


def ensure_test_dev_lock(config_hash: str, source_hash: str) -> None:
    if not DEV_LOCK_PATH.is_file():
        raise FileNotFoundError("run and lock R2MED fixed baselines and DEV method before opening TEST")
    lock = json.loads(DEV_LOCK_PATH.read_text(encoding="utf-8"))
    if (
        lock.get("status") != "LOCKED_BEFORE_TEST"
        or lock.get("config_sha256") != config_hash
        or lock.get("source_manifest_sha256") != source_hash
        or lock.get("selected_fixed_baseline") not in FIXED_ARMS
        or not lock.get("custom_method_dev_complete")
    ):
        raise ValueError("R2MED DEV selection lock is absent, incomplete, or incompatible")
    assert_committed_artifact(DEV_LOCK_PATH, repo_root=ROOT)


def run_partition(
    partition: str,
    arms: tuple[str, ...],
    *,
    scratch_root: Path,
    source_root: Path,
    bge_root: Path,
    medcpt_root: Path,
) -> dict[str, Any]:
    if partition not in {"DEV", "TEST"}:
        raise ValueError("partition must be DEV or TEST")
    if not arms or any(arm not in ARMS for arm in arms) or len(set(arms)) != len(arms):
        raise ValueError(f"arms must be unique values from {ARMS}")
    config, source_manifest, config_hash, source_hash = load_protocol()
    if partition == "TEST":
        ensure_test_dev_lock(config_hash, source_hash)
        if set(arms) != set(ARMS):
            raise ValueError("the frozen TEST must execute every predeclared arm together")

    disk = shutil.disk_usage(scratch_root if scratch_root.exists() else scratch_root.parent)
    minimum_free = int(disk.total * 0.20)
    if disk.free < minimum_free:
        raise OSError("scratch volume is below the preregistered 20% free-space floor")
    current_use = sum(path.stat().st_size for path in scratch_root.rglob("*") if path.is_file())
    if current_use > 50 * 1024**3:
        raise OSError("R2MED scratch use exceeds the preregistered 50 GiB cap")

    source_entries = dataset_entries(source_manifest, partition)
    expected_names = config["track_a_retrieval"]["test_subsets" if partition == "TEST" else "dev_subsets"]
    if [entry["name"] for entry in source_entries] != expected_names:
        raise ValueError("R2MED subset order/selection differs from the frozen configuration")

    bge_model = None
    medcpt_models = None
    torch = None
    if "bge_dense" in arms:
        from sentence_transformers import SentenceTransformer

        bge_model = SentenceTransformer(
            str(bge_root),
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            model_kwargs={"local_files_only": True},
        )
        bge_model.max_seq_length = int(
            config["track_a_retrieval"]["retrieval_parameters"]["bge_max_sequence_length"]
        )
        expected_hash = config["track_a_retrieval"]["test_model_hashes"]["bge_model_safetensors_sha256"]
        if sha256_file(bge_root / "model.safetensors") != expected_hash:
            raise ValueError("BGE model weights differ from the frozen model hash")
    if "medcpt_dense" in arms or "rrf_medcpt_rerank" in arms:
        import torch as torch_module

        torch = torch_module
        medcpt_models = load_medcpt_models(medcpt_root, torch)
        expected_hashes = config["track_a_retrieval"]["test_model_hashes"]
        actual = medcpt_models["model_manifest"]
        for key, model_name in (
            ("medcpt_query_sha256", "query_encoder"),
            ("medcpt_article_sha256", "article_encoder"),
            ("medcpt_cross_encoder_sha256", "cross_encoder"),
        ):
            if actual[model_name]["sha256"] != expected_hashes[key]:
                raise ValueError(f"local MedCPT {model_name} differs from the frozen model hash")

    summaries_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    per_dataset: dict[str, Any] = {}
    data_integrity_by_subset: dict[str, dict[str, int]] = {}
    for entry in source_entries:
        print(f"R2MED {partition} subset {entry['name']}: validating source and indexing", flush=True)
        data = load_dataset(entry, source_root)
        data_integrity_by_subset[entry["name"]] = data["source_integrity"]
        outputs = run_dataset(
            partition=partition,
            entry=entry,
            arms=arms,
            data=data,
            config_hash=config_hash,
            source_hash=source_hash,
            scratch_root=scratch_root,
            bge_root=bge_root,
            medcpt_root=medcpt_root,
            bge_model=bge_model,
            medcpt_models=medcpt_models,
            torch=torch,
        )
        per_dataset[entry["name"]] = outputs
        for arm in arms:
            result_path, manifest_path = arm_paths(scratch_root, partition, entry, arm)
            if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")).get("status") != "COMPLETED":
                raise RuntimeError(f"arm did not complete: {entry['name']} / {arm}")
            rows = read_jsonl(result_path)
            summaries_by_arm[arm].extend(rows)

    aggregate = {
        "schema_version": "e1-2-r2med-results-v1",
        "partition": partition,
        "config_sha256": config_hash,
        "source_manifest_sha256": source_hash,
        "data_integrity_by_subset": data_integrity_by_subset,
        "arms": {
            arm: {
                "summary": summarize_metrics(rows),
                "by_subset": {
                    subset: values[arm]
                    for subset, values in per_dataset.items()
                },
            }
            for arm, rows in summaries_by_arm.items()
        },
        "latency_semantics": "Retrieval stage latency only; dense query vectors are precomputed per batch, so report this as offline retrieval throughput, not production end-to-end latency.",
    }
    output_path = scratch_root / "results" / "track_a" / f"{partition.lower()}_aggregate.json"
    write_json_atomic(output_path, aggregate)
    print(json.dumps({"partition": partition, "output": str(output_path), "arms": list(arms)}))
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("DEV", "TEST"), required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SCRATCH / "sources")
    parser.add_argument("--bge-root", type=Path, default=DEFAULT_BGE_ROOT)
    parser.add_argument("--medcpt-root", type=Path, default=DEFAULT_MEDCPT_ROOT)
    args = parser.parse_args()
    run_partition(
        args.partition,
        tuple(args.arms),
        scratch_root=args.scratch_root,
        source_root=args.source_root,
        bge_root=args.bge_root,
        medcpt_root=args.medcpt_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
