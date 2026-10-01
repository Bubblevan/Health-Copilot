"""Fail-closed Windows loopback listener snapshot with narrow netstat fallback."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


def snapshot_with_permission_fallback(
    primary: Callable[[], dict[str, Any]],
    fallback: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Fallback only when Get-NetTCPConnection is explicitly denied access."""
    try:
        return primary(), "GET_NET_TCP_CONNECTION"
    except RuntimeError as exc:
        if "Get-NetTCPConnection" not in str(exc) or "PermissionDenied" not in str(exc):
            raise
        return fallback(), "NETSTAT_CIM_PERMISSION_FALLBACK"


def netstat_cim_process_snapshot(
    *, port: int, server_path: str, model_path: str
) -> dict[str, Any]:
    """Resolve one TCP listener through netstat, then verify its CIM process."""
    script = rf"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$listeners = @(netstat -ano -p tcp | ForEach-Object {{
  $parts = @(($_.Trim() -split '\s+') | Where-Object {{ $_ }})
  if ($parts.Count -ge 5 -and $parts[0] -eq 'TCP' -and $parts[3] -eq 'LISTENING' -and $parts[1] -match '^(?<address>.+):{port}$') {{
    [pscustomobject]@{{ LocalAddress = $Matches['address']; OwningProcess = [int]$parts[4] }}
  }}
}})
if ($listeners.Count -ne 1) {{ throw 'expected_exactly_one_listener' }}
$ownerPid = [int]$listeners[0].OwningProcess
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerPid"
if (-not $proc -or $proc.ExecutablePath -ne '{server_path}') {{ throw 'listener_process_path_mismatch' }}
function Get-FileSha256([string]$path) {{
  $algorithm = [System.Security.Cryptography.SHA256]::Create()
  $stream = [System.IO.File]::OpenRead($path)
  try {{
    return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
  }} finally {{ $stream.Dispose(); $algorithm.Dispose() }}
}}
[pscustomobject]@{{
  pid = $ownerPid
  addresses = @($listeners | ForEach-Object {{ $_.LocalAddress }})
  executable_path = $proc.ExecutablePath
  executable_sha256 = Get-FileSha256 $proc.ExecutablePath
  model_path = '{model_path}'
  model_sha256 = Get-FileSha256 '{model_path}'
  command_line = $proc.CommandLine
}} | ConvertTo-Json -Compress
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8-sig",
        timeout=45,
    )
    if result.returncode != 0:
        raise RuntimeError(f"netstat_cim_process_snapshot_failed:{result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("netstat_cim_process_snapshot_invalid_json") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("netstat_cim_process_snapshot_not_object")
    return payload


def sha256_file(path: str | Path) -> str:
    """Small helper for tests and audit tooling."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
