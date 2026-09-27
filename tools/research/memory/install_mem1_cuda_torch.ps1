$ErrorActionPreference = "Stop"

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$MemEvalRoot = (Resolve-Path (Join-Path $RepositoryRoot "..\external\memory\MemEval")).Path
$Python = Join-Path $MemEvalRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Pinned MemEval Python environment is missing: $Python"
}

$Uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $Uv) {
    throw "uv is required to install the ignored local CUDA runtime overlay."
}

$Cache = Join-Path $env:TEMP ("health-copilot-mem1-uv-" + [guid]::NewGuid().ToString("N"))
$PreviousCache = $env:UV_CACHE_DIR
try {
    $env:UV_CACHE_DIR = $Cache
    & $Uv.Source pip install --python $Python `
        --index-url https://download.pytorch.org/whl/cu128 `
        "torch==2.10.0+cu128"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA PyTorch overlay installation failed with exit code $LASTEXITCODE"
    }

    & $Python -c "import torch; print(f'torch={torch.__version__}; cuda={torch.version.cuda}; available={torch.cuda.is_available()}'); assert torch.__version__ == '2.10.0+cu128'; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA PyTorch verification failed with exit code $LASTEXITCODE"
    }
}
finally {
    if ($null -eq $PreviousCache) {
        Remove-Item Env:UV_CACHE_DIR -ErrorAction SilentlyContinue
    }
    else {
        $env:UV_CACHE_DIR = $PreviousCache
    }
    if (Test-Path -LiteralPath $Cache -PathType Container) {
        $ResolvedCache = (Resolve-Path -LiteralPath $Cache).Path
        $TempRoot = (Resolve-Path -LiteralPath $env:TEMP).Path
        if (-not $ResolvedCache.StartsWith($TempRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to clean an unexpected uv cache path: $ResolvedCache"
        }
        Remove-Item -LiteralPath $ResolvedCache -Recurse -Force
    }
}
