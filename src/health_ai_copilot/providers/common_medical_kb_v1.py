"""Provider factory for the frozen, non-commercial Common Medical KB V1."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..knowledge.loader import load_knowledge_cards
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense import DenseIndex, DenseIndexManifest, DenseRetriever
from ..retrieval.hybrid import HybridRetriever
from .retrieval import CommonMedicalKBManifest, FrozenMedicalRAGProvider

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BUILD_ID = "common-medical-kb-v1-bge-hybrid"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _BGEEmbeddingBackend:
    """Pinned BGE encoder with the frozen E5 query prefix, loaded locally on CPU."""

    def __init__(self, model_root: Path, *, revision: str, weights_sha256: str) -> None:
        weight_path = model_root / "model.safetensors"
        if _sha256(weight_path) != weights_sha256:
            raise ValueError("Common KB BGE weights do not match the frozen manifest")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("install the retrieval extra for Common Medical KB V1") from exc
        self.model = SentenceTransformer(
            str(model_root), device="cpu", local_files_only=True
        )
        if int(self.model.get_embedding_dimension()) != 1024:
            raise ValueError("Common KB BGE model dimension differs from the frozen 1024")
        self.identity = f"BAAI/bge-large-en-v1.5@{revision}:{weights_sha256}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._encode([_BGE_QUERY_PREFIX + query])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            precision="float32",
        )
        if vectors.ndim != 2 or vectors.shape[1] != 1024:
            raise ValueError("Common KB BGE output dimension differs from the frozen 1024")
        return vectors.tolist()


def _index_hash(manifest: dict[str, Any]) -> str:
    payload = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check_build_environment(config: dict[str, Any]) -> None:
    expected = config.get("build_environment", {})
    if expected.get("python") != platform.python_version():
        raise ValueError("Common KB Python version differs from its frozen build")
    for name, expected_version in expected.get("packages", {}).items():
        try:
            current_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"Common KB runtime package is missing: {name}") from exc
        if current_version != expected_version:
            raise ValueError(
                f"Common KB runtime package version changed: {name} {current_version} != {expected_version}"
            )


@lru_cache(maxsize=1)
def _load_provider(config_json: str) -> FrozenMedicalRAGProvider:
    config = json.loads(config_json)
    _check_build_environment(config)
    repo_root = Path(__file__).resolve().parents[3]
    kb_root = Path(
        os.environ.get(
            "HEALTH_COPILOT_COMMON_KB_V1_ROOT",
            config["artifact_root"],
        )
    )
    model_root = Path(
        os.environ.get(
            "HEALTH_COPILOT_COMMON_KB_V1_MODEL_ROOT",
            config["dense_encoder"]["model_root"],
        )
    )
    dense_config = config["dense_encoder"]
    if (
        dense_config["model"] != "BAAI/bge-large-en-v1.5"
        or dense_config["revision"] != "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
        or dense_config["weights_sha256"]
        != "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7"
        or dense_config["dimension"] != 1024
        or dense_config["query_prefix"] != _BGE_QUERY_PREFIX
        or dense_config["document_prefix"] != ""
        or dense_config["device"] != "cpu"
    ):
        raise ValueError("Common KB dense encoder config differs from the frozen implementation")
    bm25_config = config["bm25_index_config"]
    if (
        bm25_config.get("implementation")
        != "health_ai_copilot.retrieval.bm25.BM25Retriever"
        or bm25_config.get("tokenizer") != "jieba-nfkc-lowercase-v1"
        or bm25_config.get("fields") != ["title", "content", "tags"]
    ):
        raise ValueError("Common KB BM25 config differs from the frozen implementation")
    if (
        config["rrf_config"].get("weights") != [1, 1]
        or config["rrf_config"].get("candidate_depth") != config["retrieval_top_k"]
        or config["rrf_config"].get("k") != 60
    ):
        raise ValueError("Common KB RRF config differs from the frozen implementation")

    metadata = json.loads((kb_root / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get("corpus_id") != config.get("corpus_id"):
        raise ValueError("Common KB external artifact corpus identity differs from config")
    if metadata.get("index_hash") != config.get("index_hash"):
        raise ValueError("Common KB external artifact index identity differs from config")
    identity_payload = config.get("index_identity_payload")
    if not isinstance(identity_payload, dict) or _index_hash(identity_payload) != config.get("index_hash"):
        raise ValueError("Common KB index identity payload does not match its frozen hash")
    if metadata.get("index_identity_payload") != identity_payload:
        raise ValueError("Common KB artifact index identity payload differs from config")
    if metadata.get("dense_index_manifest") != config.get("dense_index_manifest"):
        raise ValueError("Common KB dense index manifest differs from config")
    if metadata.get("source_list") != config.get("source_list"):
        raise ValueError("Common KB source list differs from the qualified artifact")
    if metadata.get("document_hashes") != config.get("document_hashes"):
        raise ValueError("Common KB document hashes differ from the qualified artifact")
    for relative_path, expected_hash in config.get("implementation_hashes", {}).items():
        if _sha256(repo_root / relative_path) != expected_hash:
            raise ValueError(f"Common KB implementation changed after qualification: {relative_path}")
    for relative_path, expected_hash in config["dense_encoder"]["model_file_hashes"].items():
        if _sha256(model_root / relative_path) != expected_hash:
            raise ValueError(f"Common KB encoder config changed after qualification: {relative_path}")
    dense_dir = kb_root / "dense_index"
    vectors_path = dense_dir / "vectors.json"
    if _sha256(vectors_path) != metadata.get("dense_vectors_sha256"):
        raise ValueError("Common KB dense vector artifact hash mismatch")

    cards_by_id = {card.id: card for card in load_knowledge_cards(repo_root / "data/knowledge_cards")}
    source_ids = tuple(config["source_list"])
    expected_hashes = dict(config["document_hashes"])
    cards = []
    for source_id in source_ids:
        card = cards_by_id.get(source_id)
        if card is None:
            raise ValueError(f"Common KB source card is missing: {source_id}")
        card_path = repo_root / "data/knowledge_cards" / f"{source_id}.json"
        if _sha256(card_path) != expected_hashes.get(source_id):
            raise ValueError(f"Common KB source card hash changed: {source_id}")
        cards.append(card)
    if len(cards) != len(expected_hashes):
        raise ValueError("Common KB document hash inventory has extra or missing IDs")

    encoder = _BGEEmbeddingBackend(
        model_root,
        revision=config["dense_encoder"]["revision"],
        weights_sha256=config["dense_encoder"]["weights_sha256"],
    )
    dense_manifest = DenseIndexManifest(
        **metadata["dense_index_manifest"]
    )
    dense_index = DenseIndex.load(dense_dir, expected=dense_manifest)
    dense_retriever = DenseRetriever(dense_index, encoder)
    bm25 = BM25Retriever(
        cards,
        k1=float(config["bm25_index_config"]["k1"]),
        b=float(config["bm25_index_config"]["b"]),
    )
    retriever = HybridRetriever(
        bm25,
        dense_retriever,
        rrf_k=int(config["rrf_config"]["k"]),
    )
    manifest = CommonMedicalKBManifest(
        corpus_id=config["corpus_id"],
        source_list=source_ids,
        use_status=config["use_status"],
        document_hashes=tuple(expected_hashes.values()),
        index_hash=config["index_hash"],
        bm25_config=config["bm25_index_config"],
        dense_encoder_version=(
            f"{config['dense_encoder']['model']}@{config['dense_encoder']['revision']}"
        ),
        rrf_config=config["rrf_config"],
        top_k=int(config["retrieval_top_k"]),
    )
    return FrozenMedicalRAGProvider(retriever, manifest)


def common_medical_kb_v1_provider_factory(config: dict[str, Any]) -> FrozenMedicalRAGProvider:
    """Return the singleton provider after validating source, model, and index hashes."""
    if config.get("status") != "READY" or config.get("readiness") != "YES":
        raise ValueError("Common Medical KB V1 is not marked READY")
    external = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _load_provider(external)
