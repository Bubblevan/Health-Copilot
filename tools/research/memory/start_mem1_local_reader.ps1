param(
    [switch] $RestartMem1Server,
    [switch] $CpuKv,
    [ValidateSet("q4_0", "q8_0")]
    [string] $KvCacheType = "q4_0"
)

$ErrorActionPreference = "Stop"

$Model = "E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf"
$ExpectedSha256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
$ExpectedServerVersion = "10068"
$ExpectedServerBuild = "571d0d540"
$ExpectedServerSha256 = "3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb"
$ModelHash = (Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ModelHash -ne $ExpectedSha256) {
    throw "Local Qwen3-8B reader does not match the frozen MEM-1 artifact."
}

$PackageRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
$Candidates = @(Get-ChildItem -LiteralPath $PackageRoot -Directory -Filter "ggml.llamacpp_*" |
    ForEach-Object { Join-Path $_.FullName "llama-server.exe" } |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
if (-not $Candidates) {
    throw "The installed Windows llama.cpp server was not found."
}
$MatchingServers = @()
foreach ($Candidate in $Candidates) {
    $CandidateHash = (Get-FileHash -LiteralPath $Candidate -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($CandidateHash -ne $ExpectedServerSha256) {
        continue
    }
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        # llama.cpp may write its version banner to stderr; treat it as captured text.
        $ErrorActionPreference = "Continue"
        $VersionOutput = (& $Candidate --version 2>&1 | Out-String).Trim()
        $VersionExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($VersionExitCode -ne 0 -or
        $VersionOutput -notmatch "\b$ExpectedServerVersion\b" -or
        $VersionOutput -notmatch [regex]::Escape($ExpectedServerBuild)) {
        throw "Pinned llama-server.exe SHA256 matched, but its runtime version/build did not match $ExpectedServerVersion ($ExpectedServerBuild): $VersionOutput"
    }
    $MatchingServers += $Candidate
}
if ($MatchingServers.Count -ne 1) {
    throw "Expected exactly one llama-server.exe with version $ExpectedServerVersion ($ExpectedServerBuild) and SHA256 $ExpectedServerSha256; found $($MatchingServers.Count)."
}
$Server = $MatchingServers[0]

$Owned = Get-CimInstance Win32_Process -Filter "name='llama-server.exe'" |
    Where-Object {
        $_.CommandLine -like "*$Model*" -and
        $_.CommandLine -like "*--host 127.0.0.1*--port 8081*"
    }
if ($Owned) {
    if (-not $RestartMem1Server) {
        throw "The frozen MEM-1 reader is already running. Use -RestartMem1Server only to replace this exact local reader."
    }
    foreach ($Process in $Owned) {
        Stop-Process -Id $Process.ProcessId -Force
    }
    Start-Sleep -Seconds 2
}

$Other = Get-CimInstance Win32_Process -Filter "name='llama-server.exe'" |
    Where-Object { $_.CommandLine -like "*--port 8081*" }
if ($Other) {
    throw "Port 8081 belongs to a different llama.cpp process; refusing to replace it."
}

$Log = Join-Path $env:TEMP "health-copilot-mem1-reader.log"
$ErrorLog = Join-Path $env:TEMP "health-copilot-mem1-reader-error.log"
$Arguments = @(
    "-m", $Model,
    "--host", "127.0.0.1",
    "--port", "8081",
    "--ctx-size", "131072",
    "--n-predict", "8192",
    "--rope-scaling", "yarn",
    "--rope-scale", "4",
    "--yarn-orig-ctx", "32768",
    "--override-kv", "qwen3.context_length=int:131072",
    "--cache-type-k", $KvCacheType,
    "--cache-type-v", $KvCacheType,
    "--n-gpu-layers", "99",
    "--flash-attn", "on",
    "--parallel", "1"
)
if ($CpuKv) {
    $Arguments += "--no-kv-offload"
}

$Process = Start-Process -FilePath $Server -ArgumentList $Arguments `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $Log -RedirectStandardError $ErrorLog
$KvMode = if ($CpuKv) { "CPU" } else { "GPU" }
Write-Output "Started frozen MEM-1 llama.cpp reader (PID $($Process.Id)); $KvCacheType KV cache: $KvMode; log: $Log"
