"""Verify and pin the R2MED sprint's public inputs and upstream identity."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from eval.r2med_crb_data import (
    PARTITIONS,
    PINNED_UPSTREAM_COMMIT,
    UPSTREAM_PROMPT_FAMILY,
    sha256_file,
)
from eval.r2med_gar_generation import EXPECTED_UPSTREAM_FILES

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT_DEFAULT = Path(r"D:\MyLab\Jianli\external\rag\R2MED")
SOURCE_ROOT_DEFAULT = Path(r"E:\Health-Copilot-E1.2\sources")
OLD_SOURCE_MANIFEST = ROOT / "runs/e1_2/r2med_source_manifest.json"
OLD_FROZEN_CONFIG = ROOT / "runs/e1_2/frozen_test_config.json"
OUTPUT_PATH = ROOT / "runs/rag_r2med_crb/source_manifest.json"
EXPECTED_COUNTS = {
    "DEV": {"PMC-Treatment": 150, "PMC-Clinical": 114, "IIYi-Clinical": 129},
    "TEST": {"MedQA-Diag": 118, "MedXpertQA-Exam": 97, "Medical-Sciences": 88},
}
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
    "weights_file": "model.safetensors",
    "weights_bytes": 1_340_616_616,
    "weights_sha256": "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7",
}


def git_text(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def assert_committed_clean(path: Path) -> None:
    relative = path.relative_to(ROOT).as_posix()
    git_text("ls-files", "--error-unmatch", relative)
    if git_text("status", "--porcelain", "--", relative):
        raise ValueError(f"required source identity is modified in the worktree: {relative}")


def _source_entries(
    manifest: dict[str, Any], partition: str, source_root: Path
) -> list[dict[str, Any]]:
    old_key = "dev_subsets" if partition == "DEV" else "test_subsets"
    old_entries = manifest.get(old_key)
    if not isinstance(old_entries, list):
        raise TypeError(f"existing R2MED manifest has no {old_key}")
    selected: list[dict[str, Any]] = []
    for subset in PARTITIONS[partition]:
        match = [entry for entry in old_entries if entry.get("name") == subset]
        if len(match) != 1:
            raise ValueError(f"expected one E1.2 source manifest entry for {subset}")
        entry = match[0]
        if int(entry.get("query_count", -1)) != EXPECTED_COUNTS[partition][subset]:
            raise ValueError(f"unexpected frozen query count for {subset}")
        source_dir = source_root / entry["directory"]
        for filename in ("query.jsonl", "corpus.jsonl"):
            path = source_dir / filename
            identity = entry["files"][filename]
            if not path.is_file() or path.stat().st_size != identity["bytes"]:
                raise ValueError(f"E1.2 input size mismatch: {subset}/{filename}")
            if sha256_file(path) != identity["sha256"]:
                raise ValueError(f"E1.2 input checksum mismatch: {subset}/{filename}")
        selected.append(entry)
    return selected


def build_manifest(
    upstream_root: Path = UPSTREAM_ROOT_DEFAULT,
    source_root: Path = SOURCE_ROOT_DEFAULT,
) -> dict[str, Any]:
    upstream_commit = git_text("rev-parse", "HEAD", cwd=upstream_root)
    if upstream_commit != PINNED_UPSTREAM_COMMIT:
        raise ValueError(f"R2MED upstream commit mismatch: {upstream_commit}")
    upstream_files: dict[str, str] = {}
    for relative, expected_hash in EXPECTED_UPSTREAM_FILES.items():
        path = upstream_root / relative
        digest = sha256_file(path)
        if digest != expected_hash:
            raise ValueError(f"R2MED upstream SHA-256 mismatch: {relative}")
        upstream_files[relative] = digest

    for artifact in (OLD_SOURCE_MANIFEST, OLD_FROZEN_CONFIG):
        assert_committed_clean(artifact)
    source_bytes = OLD_SOURCE_MANIFEST.read_bytes()
    config_bytes = OLD_FROZEN_CONFIG.read_bytes()
    old_manifest = json.loads(source_bytes)
    frozen_config = json.loads(config_bytes)
    expected_source_hash = frozen_config["track_a_retrieval"]["source_manifest_sha256"]
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != expected_source_hash:
        raise ValueError("E1.2 source manifest does not match its committed frozen config")

    datasets = {
        partition: _source_entries(old_manifest, partition, source_root)
        for partition in ("DEV", "TEST")
    }
    usage = shutil.disk_usage(source_root.anchor)
    free_pct = 100 * usage.free / usage.total
    if free_pct < 20:
        raise OSError(f"E: free space is {free_pct:.1f}%; the sprint requires at least 20%")

    return {
        "schema_version": "r2med-gar-source-manifest-v1",
        "test_status": "PUBLIC_BENCHMARK_REUSED",
        "upstream": {
            "repository": "https://github.com/R2MED/R2MED.git",
            "commit": upstream_commit,
            "files_sha256": upstream_files,
        },
        "prompt_family_mapping": UPSTREAM_PROMPT_FAMILY,
        "dataset_source": {
            "root": str(source_root),
            "existing_e1_2_manifest": str(OLD_SOURCE_MANIFEST),
            "existing_e1_2_manifest_sha256": source_hash,
            "existing_frozen_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "datasets": datasets,
        "models": {
            "generator": {**QWEN, "status": "SHA_AND_SIZE_PINNED"},
            "dense_retriever": {**BGE, "status": "REVISION_PINNED"},
        },
        "storage": {
            "root": "E:\\Health-Copilot-RAG",
            "free_bytes_at_manifest_creation": usage.free,
            "free_percent_at_manifest_creation": round(free_pct, 2),
            "minimum_free_percent": 20,
        },
        "counts": {
            partition: {subset: EXPECTED_COUNTS[partition][subset] for subset in PARTITIONS[partition]}
            for partition in ("DEV", "TEST")
        },
    }


def main() -> None:
    manifest = build_manifest()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if OUTPUT_PATH.exists():
        if OUTPUT_PATH.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to replace frozen source manifest: {OUTPUT_PATH}")
        print(f"Source manifest unchanged: {OUTPUT_PATH}")
        return
    OUTPUT_PATH.write_text(encoded, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
