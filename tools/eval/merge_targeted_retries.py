"""Create a new full-cohort checkpoint from an immutable run and targeted retries."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

try:
    from .extract_failed_case_ids import RETRYABLE_REASONS, sha256_file
except ImportError:
    from extract_failed_case_ids import RETRYABLE_REASONS, sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no} must contain a JSON object")
                rows.append(value)
    return rows


def _retryable(row: dict[str, Any]) -> bool:
    flags = row.get("response", {}).get("safety_flags", ())
    return any(flag in RETRYABLE_REASONS for flag in flags)


def _unique_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in indexed:
            raise ValueError(f"duplicate {label} case ID: {case_id}")
        indexed[case_id] = row
    return indexed


def merge_targeted_retries(
    *,
    source_dir: Path,
    retry_dir: Path,
    selection_path: Path,
    output_dir: Path,
) -> Path:
    source_dir = source_dir.resolve()
    retry_dir = retry_dir.resolve()
    selection_path = selection_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite recovery artifact: {output_dir}")

    source_manifest_path = source_dir / "run_manifest.json"
    retry_manifest_path = retry_dir / "run_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    retry_manifest = _read_json(retry_manifest_path)
    selection = _read_json(selection_path)
    source_checkpoint = source_dir / "cases.jsonl"
    retry_checkpoint = retry_dir / "cases.jsonl"
    source_invalidation = source_dir / "invalidation.json"

    if source_manifest.get("status") != "INVALID_INFRASTRUCTURE_FAILURE":
        raise ValueError("source run must remain marked INVALID_INFRASTRUCTURE_FAILURE")
    if retry_manifest.get("status") != "COMPLETE":
        raise ValueError("targeted retry run must be COMPLETE")
    if not source_invalidation.is_file():
        raise ValueError("invalidated source run has no invalidation record")
    if sha256_file(source_checkpoint) != selection.get("source_cases_sha256"):
        raise ValueError("selection manifest does not identify the unchanged source checkpoint")
    if sha256_file(source_invalidation) != selection.get("source_invalidation_sha256"):
        raise ValueError("selection manifest does not identify the unchanged invalidation record")
    if selection.get("source_run_dir") != str(source_dir):
        raise ValueError("selection manifest source path does not match the source run")
    selected_ids = selection.get("case_ids")
    if not isinstance(selected_ids, list) or not selected_ids or any(not isinstance(x, str) for x in selected_ids):
        raise ValueError("selection manifest must contain a non-empty string case_ids list")
    selected_hash = sha256(("\n".join(selected_ids) + "\n").encode("utf-8")).hexdigest()
    if selected_hash != selection.get("case_ids_sha256"):
        raise ValueError("selected ID hash is invalid")

    source_dataset = source_manifest.get("dataset", {})
    retry_dataset = retry_manifest.get("dataset", {})
    for field in (
        "dataset_id", "source_revision", "combined_dataset_identity_sha256",
        "ids_manifest_sha256", "ids_sequence_sha256",
    ):
        if source_dataset.get(field) != retry_dataset.get(field):
            raise ValueError(f"source and retry dataset identity differ: {field}")
    if source_manifest.get("system_profile") != retry_manifest.get("system_profile"):
        raise ValueError("source and retry system profiles differ")
    if source_manifest.get("model", {}).get("checkpoint_sha256") != retry_manifest.get("model", {}).get("checkpoint_sha256"):
        raise ValueError("source and retry model checkpoint identities differ")
    if retry_dataset.get("selected_case_ids_sha256") != selected_hash:
        raise ValueError("retry run did not execute the frozen selected ID list")
    if retry_dataset.get("case_ids_file_sha256") != sha256_file(selection_path):
        raise ValueError("retry run manifest points to a different selected ID manifest")

    source_rows = _read_jsonl(source_checkpoint)
    retry_rows = _read_jsonl(retry_checkpoint)
    source_by_id = _unique_by_id(source_rows, "source")
    retry_by_id = _unique_by_id(retry_rows, "retry")
    actual_failed_ids = {case_id for case_id, row in source_by_id.items() if _retryable(row)}
    if actual_failed_ids != set(selected_ids):
        raise ValueError("selected IDs do not exactly match transport failures in the source checkpoint")
    if set(retry_by_id) != set(selected_ids):
        raise ValueError("retry checkpoint IDs do not exactly match the selected failure IDs")
    if any(_retryable(row) for row in retry_rows):
        raise ValueError("retry checkpoint still contains transport-failure rows")

    full_selection_hash = source_dataset.get("ids_sequence_sha256")
    expected_count = int(source_dataset.get("expected_case_count", 0))
    if len(source_rows) != expected_count or len(source_by_id) != expected_count:
        raise ValueError("source checkpoint is not a complete unique frozen cohort")
    if len(selected_ids) != int(selection.get("case_count", -1)):
        raise ValueError("selection manifest case_count is inconsistent")

    merged_rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        case_id = str(source_row["case_id"])
        row = dict(retry_by_id.get(case_id, source_row))
        row["record_origin"] = "targeted_transport_retry" if case_id in retry_by_id else "original_invalidated_run"
        row["record_origin_selection_sha256"] = row.get("dataset_selection_sha256")
        row["dataset_selection_sha256"] = full_selection_hash
        merged_rows.append(row)

    source_summary = _read_json(source_dir / "summary.json")
    retry_summary = _read_json(retry_dir / "summary.json")
    source_calls = sum(int(row.get("response", {}).get("provider_calls", 0)) for row in source_rows)
    retry_calls = sum(int(row.get("response", {}).get("provider_calls", 0)) for row in retry_rows)
    source_tokens = sum(
        int(row.get("response", {}).get("input_tokens") or 0)
        + int(row.get("response", {}).get("output_tokens") or 0)
        for row in source_rows
    )
    retry_tokens = sum(
        int(row.get("response", {}).get("input_tokens") or 0)
        + int(row.get("response", {}).get("output_tokens") or 0)
        for row in retry_rows
    )
    source_manifest_hash = sha256_file(source_manifest_path)
    retry_manifest_hash = sha256_file(retry_manifest_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        with (staging_dir / "cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in merged_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        trace_ids: set[str] = set()
        with (staging_dir / "traces.jsonl").open("w", encoding="utf-8", newline="\n") as out:
            for trace_path in (source_dir / "traces.jsonl", retry_dir / "traces.jsonl"):
                if not trace_path.is_file():
                    continue
                for trace in _read_jsonl(trace_path):
                    trace_id = str(trace.get("trace_id", ""))
                    if not trace_id:
                        raise ValueError(f"trace has no trace_id: {trace_path}")
                    if trace_id in trace_ids:
                        raise ValueError(f"duplicate trace ID across source and retry runs: {trace_id}")
                    trace_ids.add(trace_id)
                    out.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())

        manifest = dict(retry_manifest)
        manifest["status"] = "COMPOSITE_RECOVERY_PENDING_RESCORE"
        manifest["dataset"] = dict(retry_dataset)
        manifest["dataset"].update({
            "expected_case_count": expected_count,
            "selected_case_ids_sha256": None,
            "case_ids_file_path": None,
            "case_ids_file_sha256": None,
        })
        manifest["recovery_composite"] = {
            "schema_version": "harness-v1-targeted-recovery-composite-v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_run_dir": str(source_dir),
            "source_run_manifest_sha256": source_manifest_hash,
            "source_cases_sha256": sha256_file(source_checkpoint),
            "source_invalidation_sha256": sha256_file(source_invalidation),
            "retry_run_dir": str(retry_dir),
            "retry_run_manifest_sha256": retry_manifest_hash,
            "retry_cases_sha256": sha256_file(retry_checkpoint),
            "selection_manifest_path": str(selection_path),
            "selection_manifest_sha256": sha256_file(selection_path),
            "selected_case_count": len(selected_ids),
            "selected_case_ids_sha256": selected_hash,
            "retry_selection_reason": selection.get("selection_reason"),
            "source_run_status_remains_invalid": True,
            "selected_transport_rows_replaced": len(selected_ids),
            "non_transport_source_rows_preserved": expected_count - len(selected_ids),
            "combined_trace_count": len(trace_ids),
            "scoring_status": "PENDING_DETERMINISTIC_RESCORE",
            "source_provider_calls": source_calls,
            "retry_provider_calls": retry_calls,
            "provider_calls_including_retries": source_calls + retry_calls,
            "tokens_including_retries": source_tokens + retry_tokens,
        }
        (staging_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        elapsed = float(source_summary.get("run_wall_seconds", 0)) + float(retry_summary.get("run_wall_seconds", 0))
        (staging_dir / "summary.json").write_text(
            json.dumps({
                "status": "PENDING_DETERMINISTIC_RESCORE",
                "run_wall_seconds": round(elapsed, 3),
                "case_concurrency": max(
                    int(source_manifest.get("concurrency", 1)),
                    int(retry_manifest.get("concurrency", 1)),
                ),
                "source_run_wall_seconds": source_summary.get("run_wall_seconds"),
                "retry_run_wall_seconds": retry_summary.get("run_wall_seconds"),
                "retry_case_count": len(selected_ids),
                "case_count": expected_count,
                "provider_calls_including_retries": source_calls + retry_calls,
                "tokens_including_retries": source_tokens + retry_tokens,
                "provider_calls_per_case_including_retries": round(
                    (source_calls + retry_calls) / expected_count, 6
                ),
            }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging_dir.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--retry-run", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = merge_targeted_retries(
        source_dir=args.source_run,
        retry_dir=args.retry_run,
        selection_path=args.selection_manifest,
        output_dir=args.output,
    )
    print(json.dumps({"status": "COMPOSITE_RECOVERY_PENDING_RESCORE", "output": str(output)}))


if __name__ == "__main__":
    main()
