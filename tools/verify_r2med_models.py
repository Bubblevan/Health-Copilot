"""Verify exact model artifacts before any R2MED generator/retriever call."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

E_ROOT = Path(r"E:\Health-Copilot-RAG")
QWEN = {
    "repo": "Qwen/Qwen3-8B-GGUF",
    "revision": "6a569868d07d3bd59e8b97fb001bf8c0b254bb20",
    "file": "Qwen3-8B-Q4_K_M.gguf",
    "bytes": 5_027_783_488,
    "sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
}
BGE = {
    "repo": "BAAI/bge-large-en-v1.5",
    "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    "weight_file": "model.safetensors",
    "weight_bytes": 1_340_616_616,
    "weight_sha256": "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7",
    "core_files": {
        "1_Pooling/config.json": (191, "9bd85925f325e25246d94c4918dc02ab98f2a1b7"),
        "config.json": (779, "ab101ed8d012df683ef32f92cea51e97615ad07c"),
        "modules.json": (349, "952a9b81c0bfd99800fabf352f69c7ccd46c5e43"),
        "sentence_bert_config.json": (52, "ea85692bff64b0d1917833c31ddbca8ab10f5455"),
        "special_tokens_map.json": (125, "a8b3208c2884c4efb86e49300fdd3dc877220cdf"),
        "tokenizer.json": (711_396, "688882a79f44442ddc1f60d70334a7ff5df0fb47"),
        "tokenizer_config.json": (366, "37fca74771bc76a8e01178ce3a6055a0995f8093"),
        "vocab.txt": (231_508, "fb140275c155a9c7c5a3b3e0e77a9e839594a938"),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_models(
    qwen_path: Path = E_ROOT / "models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf",
    bge_root: Path = E_ROOT / "models/bge-large-en-v1.5",
) -> dict[str, Any]:
    qwen_path = qwen_path.resolve()
    bge_root = bge_root.resolve()
    if not qwen_path.is_file() or qwen_path.stat().st_size != QWEN["bytes"]:
        raise ValueError("Qwen GGUF is absent or has the wrong byte count")
    qwen_hash = sha256_file(qwen_path)
    if qwen_hash != QWEN["sha256"]:
        raise ValueError("Qwen GGUF SHA-256 mismatch")

    weight_path = bge_root / BGE["weight_file"]
    if not weight_path.is_file() or weight_path.stat().st_size != BGE["weight_bytes"]:
        raise ValueError("BGE-large model.safetensors is absent or has the wrong byte count")
    if sha256_file(weight_path) != BGE["weight_sha256"]:
        raise ValueError("BGE-large weights do not match the pinned Hugging Face revision")

    bge_files: dict[str, dict[str, Any]] = {}
    for relative, (size, expected_blob) in BGE["core_files"].items():
        path = bge_root / relative
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"pinned BGE-large file missing or wrong size: {relative}")
        actual_blob = git_blob_sha1(path)
        if actual_blob != expected_blob:
            raise ValueError(f"pinned BGE-large file revision mismatch: {relative}")
        bge_files[relative] = {"bytes": size, "git_blob_sha1": actual_blob, "sha256": sha256_file(path)}

    space = shutil.disk_usage(E_ROOT.anchor)
    free_pct = 100 * space.free / space.total
    if free_pct < 20:
        raise OSError(f"E: free space {free_pct:.1f}% is below the 20% sprint floor")
    return {
        "schema_version": "r2med-gar-verified-models-v1",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "qwen": {**QWEN, "path": str(qwen_path), "verified_sha256": qwen_hash},
        "bge_large": {
            "repo": BGE["repo"],
            "revision": BGE["revision"],
            "path": str(bge_root),
            "weights_file": BGE["weight_file"],
            "weights_bytes": BGE["weight_bytes"],
            "weights_sha256": BGE["weight_sha256"],
            "core_files": bge_files,
        },
        "storage": {
            "root": str(E_ROOT),
            "free_bytes": space.free,
            "free_percent": round(free_pct, 2),
            "minimum_free_percent": 20,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, default=E_ROOT / "models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf")
    parser.add_argument("--bge-root", type=Path, default=E_ROOT / "models/bge-large-en-v1.5")
    args = parser.parse_args()
    manifest = verify_models(args.qwen, args.bge_root)
    output = E_ROOT / "models/verified_models.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified Qwen SHA256 {manifest['qwen']['verified_sha256']}")
    print(f"Verified BGE-large revision {manifest['bge_large']['revision']} and weights SHA256 {manifest['bge_large']['weights_sha256']}")
    print(f"E: free {manifest['storage']['free_percent']}%")


if __name__ == "__main__":
    main()
