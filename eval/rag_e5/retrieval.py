"""Frozen E5-B2 Lucene BM25, BGE-large, and action-profile retrieval."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from eval.r2med_crb import BGE_QUERY_PREFIX
from eval.r2med_multiview import RankedDocument, dense_search


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def weighted_rrf(
    channels: Sequence[Sequence[RankedDocument]],
    *,
    weights: Sequence[int],
    k: int,
    top_k: int,
) -> list[RankedDocument]:
    if not channels or len(channels) != len(weights):
        raise ValueError("RRF channels and weights must have the same nonzero length")
    if k <= 0 or top_k <= 0 or any(weight <= 0 for weight in weights):
        raise ValueError("RRF requires positive k, depth, and channel weights")
    scores: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    for channel, weight in zip(channels, weights, strict=True):
        ids = [item.doc_id for item in channel]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieval channel contains duplicate chunk IDs")
        for rank, item in enumerate(channel, start=1):
            scores[item.doc_id] = scores.get(item.doc_id, 0.0) + weight / (k + rank)
            first_rank[item.doc_id] = min(first_rank.get(item.doc_id, rank), rank)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], first_rank[chunk_id], chunk_id))
    return [RankedDocument(chunk_id, scores[chunk_id]) for chunk_id in ordered[:top_k]]


class E5B1LuceneBM25:
    """Use the frozen 4.3.2 document matrix and exact Lucene tokenization for queries."""

    def __init__(self, index_root: Path, *, chunk_ids: Sequence[str]) -> None:
        manifest = json.loads((index_root / "bm25_index_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("implementation") != "gensim.models.LuceneBM25Model + SparseMatrixSimilarity":
            raise ValueError("active BM25 index is not the frozen LuceneBM25 implementation")
        if (manifest.get("k1"), manifest.get("b")) != (0.9, 0.4):
            raise ValueError("active BM25 k1/b differ from the frozen profile")
        model_path = index_root / "bm25_dictionary_model.json"
        matrix_path = index_root / "bm25_document_term_weights.npz"
        token_path = index_root / "bm25_lucene_tokens.jsonl"
        for path, field in (
            (model_path, "dictionary_model_sha256"),
            (matrix_path, "document_term_matrix_sha256"),
            (token_path, "token_artifact_sha256"),
        ):
            if _sha256_file(path) != manifest.get(field):
                raise ValueError(f"frozen BM25 artifact hash mismatch: {path.name}")
        tokens = _read_jsonl(token_path)
        token_ids = [row.get("chunk_id") for row in tokens]
        if token_ids != list(chunk_ids):
            raise ValueError("BM25 token artifact order differs from the active corpus")
        self.doc_ids = tuple(token_ids)
        model = json.loads(model_path.read_text(encoding="utf-8"))
        self.term_to_id = {term: index for index, term in enumerate(model["terms"])}
        self.idfs = np.asarray(model["idfs"], dtype=np.float64)
        self.avgdl = float(model["avg_document_length"])
        self.k1 = float(model["k1"])
        self.b = float(model["b"])
        self.matrix = sparse.load_npz(matrix_path).tocsr()
        if self.matrix.shape != (len(self.doc_ids), len(self.term_to_id)):
            raise ValueError("frozen BM25 matrix dimensions do not match its dictionary/corpus")
        java_home = Path(os.environ.get("JAVA_HOME", r"D:\jdk-21.0.4"))
        java = java_home / "bin" / "java.exe"
        if not java.is_file():
            raise RuntimeError("the frozen Lucene analyzer requires the pinned local JDK 21")
        pyserini_root = Path(
            os.environ.get(
                "E5_PYSERINI_ROOT",
                r"E:\Health-Copilot-Models\cache\python-packages\pyserini",
            )
        )
        analyzer_jar = pyserini_root / "resources" / "jars" / "anserini-1.3.0-fatjar.jar"
        expected_jar_sha = manifest.get("analyzer", {}).get("analyzer_jar_sha256")
        if not analyzer_jar.is_file() or _sha256_file(analyzer_jar) != expected_jar_sha:
            raise ValueError("pinned Anserini analyzer jar differs from the frozen Lucene index")
        analyzer_class = index_root / "java-classes" / "E5LuceneAnalyzerCli.class"
        if not analyzer_class.is_file():
            raise FileNotFoundError("the frozen index's Lucene analyzer CLI class is missing")
        self.analyzer = _PersistentLuceneAnalyzer(java, analyzer_class.parent, analyzer_jar)

    def search(self, query: str, *, top_k: int = 100) -> list[RankedDocument]:
        if top_k <= 0:
            return []
        tokens = self.analyzer.analyze(query)
        frequencies = Counter(tokens)
        query_length = len(tokens)
        term_ids: list[int] = []
        values: list[float] = []
        for term, frequency in frequencies.items():
            term_id = self.term_to_id.get(term)
            if term_id is None:
                continue
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * query_length / self.avgdl
            )
            term_ids.append(term_id)
            values.append(
                self.idfs[term_id] * frequency / denominator
            )
        query_vector = sparse.csr_matrix(
            (
                np.asarray(values, dtype=np.float64),
                (np.zeros(len(term_ids), dtype=np.int32), np.asarray(term_ids, dtype=np.int32)),
            ),
            shape=(1, len(self.term_to_id)),
        )
        scores = (self.matrix @ query_vector.T).toarray().reshape(-1)
        order = sorted(range(len(self.doc_ids)), key=lambda index: (-float(scores[index]), self.doc_ids[index]))
        return [
            RankedDocument(self.doc_ids[index], float(scores[index]))
            for index in order[:top_k]
        ]

    def close(self) -> None:
        self.analyzer.close()


class _PersistentLuceneAnalyzer:
    """Keep one pinned Java analyzer alive for exact, low-overhead query tokenization."""

    def __init__(self, java: Path, classes: Path, jar: Path) -> None:
        classpath = os.pathsep.join((str(classes), str(jar)))
        self.process = subprocess.Popen(
            [
                str(java),
                "-Dfile.encoding=UTF-8",
                "--add-modules=jdk.incubator.vector",
                "-classpath",
                classpath,
                "E5LuceneAnalyzerCli",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="ascii",
            bufsize=1,
        )
        self._sequence = 0

    def analyze(self, text: str) -> list[str]:
        if self.process.poll() is not None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("pinned Lucene analyzer process is not running")
        self._sequence += 1
        row_id = str(self._sequence)
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        self.process.stdin.write(f"{row_id}\t{encoded}\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("pinned Lucene analyzer returned no result")
        decoded = base64.b64decode(line.strip(), validate=True).decode("utf-8")
        returned_id, tokens = decoded.split("\t", maxsplit=1)
        if returned_id != row_id:
            raise RuntimeError("pinned Lucene analyzer changed query response order")
        return tokens.split()

    def close(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)


class E5BGEQueryEncoder:
    """Load the exact local BGE-large weights without network access, on CPU."""

    def __init__(self, model_root: Path, *, expected_weights_sha256: str) -> None:
        weight_path = model_root / "model.safetensors"
        if _sha256_file(weight_path) != expected_weights_sha256:
            raise ValueError("BGE-large weights do not match the frozen action profile")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for BGE-large retrieval") from exc
        self.model = SentenceTransformer(
            str(model_root), device="cpu", local_files_only=True
        )

    def encode(self, text: str) -> np.ndarray:
        vectors = self.model.encode(
            [BGE_QUERY_PREFIX + text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
            precision="float32",
        )
        vector = np.asarray(vectors[0], dtype=np.float32)
        if vector.shape != (1024,):
            raise ValueError("BGE-large query vector dimension differs from the frozen 1024")
        return vector


class E5B2Retriever:
    """The two frozen retrieval actions over the already active combined corpus."""

    def __init__(
        self,
        *,
        corpus_root: Path,
        index_root: Path,
        bge_model_root: Path,
        standard_profile: Mapping[str, Any],
        strong_profile: Mapping[str, Any],
    ) -> None:
        self.chunks = _read_jsonl(corpus_root / "chunks.jsonl")
        self.chunk_by_id = {row["chunk_id"]: row for row in self.chunks}
        if len(self.chunk_by_id) != len(self.chunks):
            raise ValueError("active external corpus has duplicate chunk IDs")
        docs = _read_jsonl(corpus_root / "search_documents.jsonl")
        self.doc_ids = [row["id"] for row in docs]
        if self.doc_ids != [row["chunk_id"] for row in self.chunks]:
            raise ValueError("active search-document order differs from corpus chunks")
        self.document_texts = {row["chunk_id"]: row["text"] for row in self.chunks}
        dense_manifest_path = index_root / "dense_index_manifest.json"
        dense_manifest = json.loads(dense_manifest_path.read_text(encoding="utf-8"))
        embedding_path = index_root / "bge_large_embeddings.npy"
        if _sha256_file(embedding_path) != dense_manifest.get("embedding_matrix_sha256"):
            raise ValueError("frozen BGE document embedding matrix hash mismatch")
        self.document_vectors = np.load(embedding_path, mmap_mode="r")
        if self.document_vectors.shape != (len(self.doc_ids), 1024):
            raise ValueError("active BGE document embeddings have an unexpected shape")
        self.bm25 = E5B1LuceneBM25(index_root, chunk_ids=self.doc_ids)
        if dense_manifest.get("weights_sha256") != standard_profile["frozen_config"]["dense"]["weights_sha256"]:
            raise ValueError("BGE weights differ across dense index/profile identities")
        if dense_manifest.get("corpus_view_id") != "PUBLIC_HEALTH_PLUS_GUIDELINE":
            raise ValueError("dense index is not the frozen combined corpus view")
        self.encoder = E5BGEQueryEncoder(
            bge_model_root,
            expected_weights_sha256=standard_profile["frozen_config"]["dense"]["weights_sha256"],
        )
        self.standard_profile = standard_profile
        self.strong_profile = strong_profile

    def _fuse(
        self,
        channels: Sequence[Sequence[RankedDocument]],
        *,
        weights: Sequence[int],
        k: int,
        output_depth: int,
    ) -> list[RankedDocument]:
        return weighted_rrf(channels, weights=weights, k=k, top_k=output_depth)

    def retrieve_standard(self, query: str) -> dict[str, Any]:
        config = self.standard_profile["frozen_config"]
        bm25 = self.bm25.search(query, top_k=config["bm25"]["top_k"])
        query_vector = self.encoder.encode(query)
        dense = dense_search(
            query_vector,
            np.asarray(self.document_vectors),
            self.doc_ids,
            top_k=config["dense"]["top_k"],
        )
        rrf = config["rrf"]
        fused = self._fuse(
            (bm25, dense), weights=rrf["weights"], k=rrf["k"],
            output_depth=rrf["output_depth"],
        )
        return {
            "channels": {"bm25_original": bm25, "bge_original": dense},
            "ranking": fused,
            "profile_sha256": self.standard_profile["config_sha256"],
        }

    def feedback_passages(self, query: str, *, depth: int) -> tuple[list[RankedDocument], list[str]]:
        feedback = self.bm25.search(query, top_k=depth)
        passages = [
            " ".join(self.document_texts[row.doc_id].replace("\n", " ").split()[:512])
            for row in feedback
        ]
        return feedback, passages

    def retrieve_strong(self, query: str, bridge: str) -> dict[str, Any]:
        config = self.strong_profile["frozen_config"]
        bm25_original = self.bm25.search(query, top_k=config["bm25"]["top_k"])
        bm25_bridge = self.bm25.search(
            f"{query} {bridge}", top_k=config["bm25"]["top_k"]
        )
        bge_original_vector = self.encoder.encode(query)
        bge_bridge_vector = self.encoder.encode(bridge)
        bge_original = dense_search(
            bge_original_vector, np.asarray(self.document_vectors), self.doc_ids,
            top_k=config["dense"]["top_k"],
        )
        bge_bridge = dense_search(
            bge_bridge_vector, np.asarray(self.document_vectors), self.doc_ids,
            top_k=config["dense"]["top_k"],
        )
        rrf = config["rrf"]
        channels = (bm25_original, bm25_bridge, bge_original, bge_bridge)
        fused = self._fuse(
            channels, weights=rrf["weights"], k=rrf["k"],
            output_depth=rrf["output_depth"],
        )
        return {
            "channels": {
                "bm25_original": bm25_original,
                "bm25_bridge": bm25_bridge,
                "bge_original": bge_original,
                "bge_bridge": bge_bridge,
            },
            "ranking": fused,
            "profile_sha256": self.strong_profile["config_sha256"],
        }

    def supplied_chunks(self, ranking: Sequence[RankedDocument], *, top_k: int = 5) -> list[dict[str, Any]]:
        return [dict(self.chunk_by_id[row.doc_id]) for row in ranking[:top_k]]
