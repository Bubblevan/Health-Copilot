param(
    [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path,
    [string]$ModelPath = 'E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf',
    [string]$ServerPath = 'C:\Users\bubblevan\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe'
)

$ErrorActionPreference = 'Stop'
$expectedModelSha = 'd98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785'
$expectedServerSha = '3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb'
$port = 8092
$runtimeDir = Join-Path $RepositoryRoot 'runs\rag_e6\runtime'
$stdoutPath = Join-Path $runtimeDir 'llama-server.stdout.log'
$stderrPath = Join-Path $runtimeDir 'llama-server.stderr.log'
$pidPath = Join-Path $runtimeDir 'llama-server.pid'
$versionPath = Join-Path $runtimeDir 'llama-server.version.txt'
$devicesPath = Join-Path $runtimeDir 'llama-server.devices.txt'
$gpuSnapshotPath = Join-Path $runtimeDir 'nvidia-smi-startup.txt'

if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) { throw 'Pinned Qwen3-8B GGUF is missing.' }
if (-not (Test-Path -LiteralPath $ServerPath -PathType Leaf)) { throw 'llama-server executable is missing.' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $ModelPath).Hash.ToLowerInvariant() -ne $expectedModelSha) {
    throw 'Qwen3-8B GGUF SHA-256 mismatch.'
}
$devices = (& $ServerPath --list-devices 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or $devices -notmatch 'Vulkan1: NVIDIA GeForce RTX 4090 Laptop GPU') {
    throw 'Verified Vulkan1 RTX 4090 device is unavailable to llama.cpp.'
}
$versionText = (& $ServerPath --version 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or $versionText -notmatch 'version: 10068 \(571d0d540\)') {
    throw 'llama.cpp version differs from the executable pinned in the U3-R manifest.'
}
$listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
if ($listener) { throw "Port $port is already occupied; refusing to attach to an unknown model server." }
if ((Test-Path -LiteralPath $stdoutPath) -or (Test-Path -LiteralPath $stderrPath) -or
    (Test-Path -LiteralPath $pidPath) -or (Test-Path -LiteralPath $versionPath) -or
    (Test-Path -LiteralPath $devicesPath) -or (Test-Path -LiteralPath $gpuSnapshotPath)) {
    throw 'E6A llama-server runtime files already exist; inspect them before starting another process.'
}
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$devices | Set-Content -LiteralPath $devicesPath -Encoding utf8
$versionText | Set-Content -LiteralPath $versionPath -Encoding utf8

$arguments = @(
    '--model', $ModelPath,
    '--alias', 'local-qwen3-8b',
    '--host', '127.0.0.1',
    '--port', "$port",
    '--ctx-size', '65536',
    '--n-gpu-layers', 'all',
    '--device', 'Vulkan1',
    '--flash-attn', 'on',
    '--cache-type-k', 'q4_0',
    '--cache-type-v', 'q4_0',
    '--parallel', '1',
    '--threads', '8',
    '--threads-batch', '8',
    '--batch-size', '512',
    '--ubatch-size', '128'
)
$process = Start-Process -FilePath $ServerPath -ArgumentList $arguments -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -NoNewline -Encoding ascii

$deadline = (Get-Date).AddMinutes(5)
$ready = $false
while ((Get-Date) -lt $deadline) {
    if ($process.HasExited) {
        throw "llama-server exited during startup (code $($process.ExitCode)); inspect stderr log."
    }
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3
        if ($health.status -eq 'ok') { $ready = $true; break }
    } catch {
        Start-Sleep -Seconds 2
    }
}
if (-not $ready) { throw 'llama-server did not become healthy within five minutes.' }
(& nvidia-smi 2>&1 | Out-String) | Set-Content -LiteralPath $gpuSnapshotPath -Encoding utf8
Write-Output "E6A llama-server ready on 127.0.0.1:$port; PID=$($process.Id); device=Vulkan1; ctx=65536; GPU layers=all."
