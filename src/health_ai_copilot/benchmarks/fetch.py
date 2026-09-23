"""Explicit E0 materialization and source-evidence collection.

No function in this module is called by import, registry listing, inspection,
normalization or audit. Network access is reached only by explicit CLI actions.
"""

from __future__ import annotations

import json
import shutil
import stat
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .contracts import BenchmarkManifest, RawArtifactIdentity, canonical_json, sha256_file

FETCH_PROTOCOL_VERSION = "e0-fetch-v1"
EXTRACTION_IMPLEMENTATION_VERSION = "safe-zip-extractor-v1"


class FetchError(RuntimeError):
    """Raised when explicit materialization cannot produce a verified artifact."""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_destination(root: Path, relative_name: str) -> Path:
    if not relative_name or "\x00" in relative_name:
        raise FetchError("archive member has an invalid name")
    if PurePosixPath(relative_name).is_absolute() or PureWindowsPath(relative_name).is_absolute():
        raise FetchError(f"archive member uses an absolute path: {relative_name}")
    if "\\" in relative_name:
        raise FetchError(f"archive member uses a Windows path separator: {relative_name}")
    parts = PurePosixPath(relative_name).parts
    if ".." in parts:
        raise FetchError(f"archive member escapes extraction root: {relative_name}")
    destination = (root / Path(*parts)).resolve()
    if destination != root.resolve() and root.resolve() not in destination.parents:
        raise FetchError(f"archive member escapes extraction root: {relative_name}")
    return destination


def safe_extract_zip(archive_path: Path, extraction_root: Path) -> dict[str, Any]:
    """Extract a zip only after validating every member path and file type."""

    extraction_root = extraction_root.resolve()
    if extraction_root.exists():
        shutil.rmtree(extraction_root)
    extraction_root.mkdir(parents=True, exist_ok=True)
    members: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = sorted(archive.infolist(), key=lambda item: item.filename)
            for info in infos:
                target = _safe_destination(extraction_root, info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise FetchError(f"archive member is a symlink: {info.filename}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if target.exists():
                    raise FetchError(f"archive member would overwrite an existing file: {info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                members.append(
                    {
                        "path": info.filename,
                        "size_bytes": target.stat().st_size,
                        "sha256": sha256_file(target),
                    }
                )
    except zipfile.BadZipFile as exc:
        raise FetchError(f"invalid zip archive: {archive_path}") from exc
    return {
        "extraction_implementation": EXTRACTION_IMPLEMENTATION_VERSION,
        "archive_sha256": sha256_file(archive_path),
        "member_count": len(members),
        "members": members,
    }


def _http_get(url: str, *, max_bytes: int | None = None) -> tuple[bytes, dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Health-Copilot-E0/1"})
    try:
        response = urllib.request.urlopen(request, timeout=90)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise FetchError(f"download failed for {url}: {exc}") from exc
    with response:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise FetchError(f"response exceeded review evidence limit: {url}")
            chunks.append(chunk)
        headers = response.headers
        return b"".join(chunks), {
            "status": getattr(response, "status", None),
            "content_type": headers.get("Content-Type"),
            "content_length_header": headers.get("Content-Length"),
            "etag": headers.get("ETag"),
            "last_modified": headers.get("Last-Modified"),
            "final_url": response.geturl(),
        }


def _download_artifact(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None,
) -> RawArtifactIdentity:
    content, metadata = _http_get(url)
    actual_sha256 = __import__("hashlib").sha256(content).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise FetchError(
            f"raw SHA mismatch for {url}: expected {expected_sha256}, got {actual_sha256}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_bytes(content)
    if destination.exists():
        destination.unlink()
    partial.replace(destination)
    return RawArtifactIdentity(
        name=destination.name,
        url=url,
        sha256=actual_sha256,
        size_bytes=len(content),
        downloaded_at=_now(),
        content_type=metadata.get("content_type"),
        upstream_etag=metadata.get("etag"),
        upstream_last_modified=metadata.get("last_modified"),
    )


def _write_raw_identity(
    raw_root: Path,
    manifest: BenchmarkManifest,
    artifacts: list[RawArtifactIdentity],
    *,
    extraction: dict[str, Any] | None = None,
) -> Path:
    value = {
        "schema_version": "raw-identity-v1",
        "fetch_protocol_version": FETCH_PROTOCOL_VERSION,
        "benchmark_id": manifest.benchmark_id,
        "manifest_revision": manifest.upstream_revision,
        "fetched_at": _now(),
        "artifacts": [artifact.to_dict() for artifact in artifacts],
        "extraction": extraction,
    }
    path = raw_root / "raw_identity.json"
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return path


def fetch_benchmark(manifest: BenchmarkManifest, data_root: str | Path) -> Path:
    """Fetch exactly the manifest artifacts; never mutate the source manifest."""

    root = Path(data_root).expanduser().resolve()
    raw_root = root / "raw" / manifest.benchmark_id
    raw_root.mkdir(parents=True, exist_ok=True)
    artifacts: list[RawArtifactIdentity] = []
    for artifact in manifest.raw_artifacts:
        destination = raw_root / artifact.name
        artifacts.append(
            _download_artifact(
                artifact.url,
                destination,
                expected_sha256=artifact.sha256,
            )
        )
    extraction = None
    if manifest.benchmark_id == "nfcorpus-v1":
        archive = raw_root / manifest.raw_artifacts[0].name
        extraction = safe_extract_zip(archive, raw_root / "extracted")
        (raw_root / "extraction_manifest.json").write_text(
            canonical_json(extraction) + "\n", encoding="utf-8"
        )
    _write_raw_identity(raw_root, manifest, artifacts, extraction=extraction)
    return raw_root


def _evidence_urls(manifest: BenchmarkManifest) -> tuple[str, ...]:
    values = [manifest.canonical_url, manifest.paper_url, manifest.license.license_url]
    values.extend(manifest.evidence_urls)
    return tuple(dict.fromkeys(value for value in values if value))


def collect_source_evidence(
    manifest: BenchmarkManifest,
    repository_root: str | Path,
    *,
    max_bytes_per_url: int = 2_000_000,
) -> Path:
    """Collect machine-readable source evidence without making legal decisions."""

    root = Path(repository_root)
    evidence: list[dict[str, Any]] = []
    for url in _evidence_urls(manifest):
        if url.startswith("repo://"):
            evidence.append({"url": url, "status": "LOCAL_REFERENCE", "sha256": None})
            continue
        try:
            content, metadata = _http_get(url, max_bytes=max_bytes_per_url)
            evidence.append(
                {
                    "url": url,
                    "status": metadata.get("status") or "OK",
                    "content_type": metadata.get("content_type"),
                    "size_bytes": len(content),
                    "sha256": __import__("hashlib").sha256(content).hexdigest(),
                    "etag": metadata.get("etag"),
                    "last_modified": metadata.get("last_modified"),
                    "final_url": metadata.get("final_url"),
                    "excerpt": content[:1200].decode("utf-8", errors="replace"),
                }
            )
        except FetchError as exc:
            evidence.append({"url": url, "status": "FETCH_ERROR", "error": str(exc)})
    raw_identity_path = data_root_from_root(root) / "raw" / manifest.benchmark_id / "raw_identity.json"
    raw_identity = None
    if raw_identity_path.exists():
        raw_identity = json.loads(raw_identity_path.read_text(encoding="utf-8"))
    output = root / "runs" / "e0" / f"license_review_{manifest.benchmark_id}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Source and license evidence — {manifest.benchmark_id}",
        "",
        "Status: `REVIEW_REQUIRED` — machine-collected evidence only; no legal conclusion or human approval.",
        "",
        f"Canonical source: {manifest.canonical_url}",
        f"Pinned revision: `{manifest.upstream_revision}`",
        f"Manifest admissibility: `{manifest.status.value}`",
        f"Declared license review: `{manifest.license.license_review_status.value}`",
        "",
        "## Evidence URLs",
        "",
    ]
    for item in evidence:
        lines.append(f"- `{item['status']}` {item['url']}")
        if item.get("sha256"):
            lines.append(f"  - fetched bytes: {item['size_bytes']}; SHA-256: `{item['sha256']}`")
            lines.append(f"  - Content-Type: `{item.get('content_type')}`; ETag: `{item.get('etag')}`")
            lines.append(f"  - Last-Modified: `{item.get('last_modified')}`")
    lines.extend(
        [
            "",
            "## Machine evidence excerpts",
            "",
            "The excerpts below are for reviewer navigation and are not a license interpretation.",
            "",
        ]
    )
    for item in evidence:
        if item.get("excerpt"):
            lines.extend([f"### {item['url']}", "", "```text", item["excerpt"], "```", ""])
    lines.extend(
        [
            "## Human decision fields",
            "",
            "- reviewer: `null`",
            "- review_date: `null`",
            "- final_admissibility: `REVIEW_REQUIRED`",
            "- redistribution_allowed: `null`",
            "- derived_artifact_commit_allowed: `null`",
            "- notes: pending human review of the evidence above",
            "",
        ]
    )
    if raw_identity:
        lines.extend(["## Local raw identity", "", "```json", canonical_json(raw_identity), "```", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    (output.with_suffix(".json")).write_text(
        canonical_json({
            "schema_version": "source-review-v1",
            "benchmark_id": manifest.benchmark_id,
            "review_status": "REVIEW_REQUIRED",
            "human_decision": None,
            "evidence": evidence,
            "raw_identity": raw_identity,
        })
        + "\n",
        encoding="utf-8",
    )
    return output


def data_root_from_root(repository_root: Path) -> Path:
    import os

    configured = os.environ.get("HEALTH_COPILOT_BENCH_DATA")
    return Path(configured).expanduser().resolve() if configured else repository_root / ".health-bench-data"
