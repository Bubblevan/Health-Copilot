param(
    [string]$ModelPath = 'E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf',
    [string]$RunRoot = 'D:\MyLab\Jianli\Health-Copilot-rag-e5-a3-20260929\runs\integration\u3r-rag-transfer-v1-e28ea9ef-final',
    [string]$ServerPath = 'C:\Users\bubblevan\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe',
    [int]$Port = 8091
)

$ErrorActionPreference = 'Stop'
$resolvedModel = (Resolve-Path -LiteralPath $ModelPath).Path
$resolvedServer = (Resolve-Path -LiteralPath $ServerPath).Path
$resolvedRunRoot = [System.IO.Path]::GetFullPath($RunRoot)
$manifestPath = Join-Path $resolvedRunRoot 'u3r_cpu_server_manifest.json'
if (Test-Path -LiteralPath $manifestPath) {
    throw 'U3-R CPU server manifest already exists; inspect it before replacing or reusing it.'
}
if (-not (Test-Path -LiteralPath $resolvedRunRoot -PathType Container)) {
    throw 'U3-R run root must be prepared before recording the CPU service.'
}
$listenerLines = @(& netstat.exe -ano -p tcp | Where-Object {
    $_ -match "^\s*TCP\s+127\.0\.0\.1:$Port\s+\S+\s+LISTENING\s+\d+\s*$"
})
if ($listenerLines.Count -ne 1) {
    throw "Expected exactly one listener on 127.0.0.1:$Port."
}
$processId = [int]([regex]::Match($listenerLines[0], '(\d+)\s*$').Groups[1].Value)
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId"
if (-not $process -or [System.IO.Path]::GetFullPath($process.ExecutablePath) -ne $resolvedServer) {
    throw 'The loopback listener is not the expected llama-server executable.'
}
$commandLine = [string]$process.CommandLine
$requiredArgs = @(
    '--host 127.0.0.1',
    "--port $Port",
    '--n-gpu-layers 0',
    '--ctx-size 16384',
    '--parallel 1',
    '--threads ',
    '--threads-batch '
)
foreach ($requiredArg in $requiredArgs) {
    if ($commandLine -notlike "*$requiredArg*") {
        throw "The llama-server process is missing required CPU/binding argument: $requiredArg"
    }
}
if ($commandLine -notlike "*$resolvedModel*") {
    throw 'The llama-server command line does not reference the frozen Qwen3 GGUF.'
}
$modelHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedModel).Hash.ToLowerInvariant()
if ($modelHash -ne 'd98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785') {
    throw 'Qwen3-8B GGUF SHA-256 does not match the frozen artifact.'
}
$health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
$models = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 10
if (-not $models.data -or -not $models.data[0].id) {
    throw 'llama-server did not publish an OpenAI-compatible model ID.'
}
$versionOutput = @(& $resolvedServer --version 2>&1 | ForEach-Object { $_.ToString() })
$version = ($versionOutput -join [Environment]::NewLine).Trim()
$manifest = [ordered]@{
    schema_version = 'u3r-cpu-llama-server-v1'
    created_at_utc = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
    host = '127.0.0.1'
    port = $Port
    cpu_only = $true
    n_gpu_layers = 0
    cpu_threads = [int]([regex]::Match($commandLine, '--threads\s+(\d+)').Groups[1].Value)
    context_size = [int]([regex]::Match($commandLine, '--ctx-size\s+(\d+)').Groups[1].Value)
    parallel_slots = [int]([regex]::Match($commandLine, '--parallel\s+(\d+)').Groups[1].Value)
    model_path = $resolvedModel
    model_sha256 = $modelHash
    model_api_id = [string]$models.data[0].id
    llama_server_path = $resolvedServer
    llama_server_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedServer).Hash.ToLowerInvariant()
    llama_cpp_version = $version
    process_id = $processId
    process_command_line = $commandLine
    health_response = ($health | ConvertTo-Json -Compress -Depth 5)
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8NoBOM
Write-Output "Recorded U3-R CPU llama-server PID=$processId, model=$($models.data[0].id), host=127.0.0.1, GPU layers=0"
