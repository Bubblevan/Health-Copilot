param(
    [switch] $RestartOwned
)

$ErrorActionPreference = "Stop"

$Model = "E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf"
$ExpectedModelSha256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
$ExpectedServerSha256 = "3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb"
$ExpectedServerVersion = "10068"
$ExpectedServerBuild = "571d0d540"
$ModelSha256 = (Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ModelSha256 -ne $ExpectedModelSha256) {
    throw "Local Qwen3-8B reader does not match the frozen artifact."
}

$PackageRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
$Servers = @(Get-ChildItem -LiteralPath $PackageRoot -Directory -Filter "ggml.llamacpp_*" |
    ForEach-Object { Join-Path $_.FullName "llama-server.exe" } |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
$MatchingServers = @()
foreach ($Candidate in $Servers) {
    if ((Get-FileHash -LiteralPath $Candidate -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedServerSha256) {
        continue
    }
    $VersionOutput = (& $Candidate --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -eq 0 -and $VersionOutput -match "\b$ExpectedServerVersion\b" -and $VersionOutput -match [regex]::Escape($ExpectedServerBuild)) {
        $MatchingServers += $Candidate
    }
}
if ($MatchingServers.Count -ne 1) {
    throw "Expected one pinned llama-server $ExpectedServerVersion / $ExpectedServerBuild; found $($MatchingServers.Count)."
}
$Server = $MatchingServers[0]

$PortProcesses = @(Get-CimInstance Win32_Process -Filter "name='llama-server.exe'" |
    Where-Object { $_.CommandLine -like "*--port 8081*" })
$Owned = @($PortProcesses | Where-Object {
    $_.CommandLine -like "*$Model*" -and $_.CommandLine -like "*--host 127.0.0.1*"
})
$Other = @($PortProcesses | Where-Object { $_.ProcessId -notin @($Owned | ForEach-Object ProcessId) })
if ($Other) {
    throw "Port 8081 belongs to a different llama.cpp process; refusing to stop it."
}
if ($Owned -and -not $RestartOwned) {
    throw "The owned Qwen reader is already running. Use -RestartOwned to replace only this exact 8081 process."
}
foreach ($Process in $Owned) {
    Stop-Process -Id $Process.ProcessId -Force
}
if ($Owned) {
    Start-Sleep -Seconds 2
}

$Log = Join-Path $env:TEMP "health-copilot-memora-qwen-parallel.log"
$ErrorLog = Join-Path $env:TEMP "health-copilot-memora-qwen-parallel-error.log"
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
    "--cache-type-k", "q4_0",
    "--cache-type-v", "q4_0",
    "--n-gpu-layers", "99",
    "--flash-attn", "on",
    "--parallel", "4"
)
$Process = Start-Process -FilePath $Server -ArgumentList $Arguments `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $Log -RedirectStandardError $ErrorLog
Write-Output "Started pinned local Qwen Memora service (PID $($Process.Id)) on 127.0.0.1:8081; 4 parallel sequences, 32,768 tokens/sequence; logs: $Log"
