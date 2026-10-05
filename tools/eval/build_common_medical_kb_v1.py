"""Freeze COMMON_MEDICAL_KB_V1 from the U2-D-approved CDC/WHO cards only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.providers.common_medical_kb_v1 import _BGEEmbeddingBackend
from health_ai_copilot.retrieval.dense import DenseIndex
from health_ai_copilot.retrieval.documents import document_from_knowledge_card

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BGE_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
EXPECTED_BGE_WEIGHTS_SHA256 = "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7"
BUILD_ID = "common-medical-kb-v1-bge-hybrid"
BM25_CONFIG = {
    "implementation": "health_ai_copilot.retrieval.bm25.BM25Retriever",
    "tokenizer": "jieba-nfkc-lowercase-v1",
    "fields": ["title", "content", "tags"],
    "k1": 1.5,
    "b": 0.75,
}
DENSE_CONFIG = {
    "model": "BAAI/bge-large-en-v1.5",
    "revision": EXPECTED_BGE_REVISION,
    "weights_sha256": EXPECTED_BGE_WEIGHTS_SHA256,
    "dimension": 1024,
    "query_prefix": "Represent this sentence for searching relevant passages: ",
    "document_prefix": "",
    "device": "cpu",
}
MODEL_IDENTITY_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "1_Pooling/config.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)
RRF_CONFIG = {"k": 60, "weights": [1, 1], "candidate_depth": 5}
BUILD_ENVIRONMENT = {
    "python": platform.python_version(),
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("jieba", "sentence-transformers", "transformers", "torch", "numpy", "safetensors")
    },
    "embedding_device": "cpu",
}
USE_STATUS = (
    "U2-D owner-qualified non-commercial external retrieval only; preserve CDC/WHO "
    "attribution and source links; no source-document redistribution or training."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _approved_card_inventory(repo_root: Path) -> tuple[list[dict], list]:
    qualification_path = repo_root / "docs/research/integration/u2d_external_evidence_qualification.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    family = next(item for item in qualification["families"] if item["family"] == "PUBLIC_HEALTH")
    if family.get("qualified_asset_count") != 26 or family.get("excluded_asset_count") != 4:
        raise ValueError("U2-D source qualification changed; re-audit the Common KB source set")

    source_dir = repo_root / "data/knowledge_cards"
    metadata_by_id = {}
    for path in sorted(source_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        publisher = row.get("publisher")
        host = (urlparse(row.get("source_url", "")).hostname or "").lower()
        if publisher == "Centers for Disease Control and Prevention" and (
            host == "cdc.gov" or host.endswith(".cdc.gov")
        ):
            kind = "CDC"
        elif publisher == "World Health Organization" and (
            host == "who.int" or host.endswith(".who.int")
        ):
            kind = "WHO"
        else:
            continue
        metadata_by_id[row["id"]] = {
            "id": row["id"],
            "publisher": kind,
            "source_url": row["source_url"],
            "sha256": _sha256(path),
            "reviewed_at": row.get("reviewed_at"),
            "path": path.name,
        }
    counts = Counter(row["publisher"] for row in metadata_by_id.values())
    if counts != {"CDC": 21, "WHO": 5}:
        raise ValueError(f"qualified source inventory differs from U2-D: {dict(counts)}")
    cards_by_id = {card.id: card for card in load_knowledge_cards(source_dir)}
    ids = sorted(metadata_by_id)
    return [metadata_by_id[item] for item in ids], [cards_by_id[item] for item in ids]


def build(*, repo_root: Path, artifact_root: Path, model_root: Path) -> dict:
    sources, cards = _approved_card_inventory(repo_root)
    if any(not row["reviewed_at"] for row in sources):
        raise ValueError("all Common KB source cards must retain their human review date")
    if _sha256(model_root / "model.safetensors") != EXPECTED_BGE_WEIGHTS_SHA256:
        raise ValueError("local BGE weights differ from the frozen E5 BGE identity")
    model_file_hashes = {
        relative_path: _sha256(model_root / relative_path)
        for relative_path in MODEL_IDENTITY_FILES
    }

    corpus_hash = _canonical_hash(
        {"source_list": [row["id"] for row in sources], "document_hashes": [row["sha256"] for row in sources]}
    )
    corpus_id = f"COMMON_MEDICAL_KB_V1_CDC_WHO_BP_26_{corpus_hash[:12]}"
    encoder = _BGEEmbeddingBackend(
        model_root,
        revision=EXPECTED_BGE_REVISION,
        weights_sha256=EXPECTED_BGE_WEIGHTS_SHA256,
    )
    dense_index = DenseIndex.build(
        [document_from_knowledge_card(card) for card in cards],
        encoder,
        knowledge_pack_version=corpus_id,
        build_commit=BUILD_ID,
    )
    dense_dir = artifact_root / "dense_index"
    dense_index.save(dense_dir)
    dense_vectors_sha256 = _sha256(dense_dir / "vectors.json")

    implementation_paths = [
        "src/health_ai_copilot/knowledge/loader.py",
        "src/health_ai_copilot/providers/common_medical_kb_v1.py",
        "src/health_ai_copilot/providers/retrieval.py",
        "src/health_ai_copilot/retrieval/bm25.py",
        "src/health_ai_copilot/retrieval/dense.py",
        "src/health_ai_copilot/retrieval/documents.py",
        "src/health_ai_copilot/retrieval/hybrid.py",
        "src/health_ai_copilot/retrieval/tokenizer.py",
        "tools/eval/build_common_medical_kb_v1.py",
    ]
    implementation_hashes = {path: _sha256(repo_root / path) for path in implementation_paths}
    index_identity_payload = {
        "corpus_sha256": corpus_hash,
        "source_manifest_sha256": _sha256(
            repo_root / "docs/research/integration/u2d_external_evidence_qualification.json"
        ),
        "bm25_config": BM25_CONFIG,
        "dense_encoder": {**DENSE_CONFIG, "model_file_hashes": model_file_hashes},
        "dense_index_manifest": dense_index.manifest.__dict__,
        "dense_vectors_sha256": dense_vectors_sha256,
        "rrf_config": RRF_CONFIG,
        "build_environment": BUILD_ENVIRONMENT,
        "retrieval_top_k": 5,
        "implementation_hashes": implementation_hashes,
    }
    index_hash = _canonical_hash(index_identity_payload)
    config = {
        "schema_version": "common-medical-kb-v1",
        "status": "READY",
        "readiness": "YES",
        "corpus_id": corpus_id,
        "corpus_scope": "Hypertension patient education only; 21 CDC and 5 WHO cards.",
        "corpus_sha256": corpus_hash,
        "source_list": [row["id"] for row in sources],
        "document_hashes": {row["id"]: row["sha256"] for row in sources},
        "source_inventory": sources,
        "source_qualification": "docs/research/integration/u2d_external_evidence_qualification.json",
        "source_qualification_sha256": index_identity_payload["source_manifest_sha256"],
        "use_status": USE_STATUS,
        "bm25_index_config": BM25_CONFIG,
        "dense_encoder": {
            **DENSE_CONFIG,
            "model_file_hashes": model_file_hashes,
            "model_root": str(model_root),
        },
        "rrf_config": RRF_CONFIG,
        "retrieval_top_k": 5,
        "build_environment": BUILD_ENVIRONMENT,
        "index_hash": index_hash,
        "index_identity_payload": index_identity_payload,
        "artifact_root": str(artifact_root),
        "dense_index_manifest": dense_index.manifest.__dict__,
        "dense_vectors_sha256": dense_vectors_sha256,
        "implementation_hashes": implementation_hashes,
        "provider_factory": (
            "health_ai_copilot.providers.common_medical_kb_v1:"
            "common_medical_kb_v1_provider_factory"
        ),
    }
    metadata = {
        "schema_version": "common-medical-kb-v1-artifacts",
        "corpus_id": corpus_id,
        "corpus_sha256": corpus_hash,
        "index_hash": index_hash,
        "source_list": config["source_list"],
        "document_hashes": config["document_hashes"],
        "dense_index_manifest": dense_index.manifest.__dict__,
        "dense_vectors_sha256": dense_vectors_sha256,
        "implementation_hashes": implementation_hashes,
        "index_identity_payload": index_identity_payload,
        "authorization": USE_STATUS,
        "gold_or_eval_rows_opened": False,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_path = repo_root / "configs/eval/common_medical_kb_v1.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/root/gpufree-share/data/health-copilot/common-medical-kb-v1"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/root/gpufree-share/models/bge-large-en-v1.5"),
    )
    args = parser.parse_args()
    result = build(repo_root=ROOT, artifact_root=args.artifact_root, model_root=args.model_root)
    print(
        json.dumps(
            {
                "corpus_id": result["corpus_id"],
                "corpus_sha256": result["corpus_sha256"],
                "index_hash": result["index_hash"],
                "document_count": len(result["source_list"]),
                "bm25_k1_b": [BM25_CONFIG["k1"], BM25_CONFIG["b"]],
                "dense_weights_sha256": DENSE_CONFIG["weights_sha256"],
                "rrf_config": RRF_CONFIG,
                "top_k": result["retrieval_top_k"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
