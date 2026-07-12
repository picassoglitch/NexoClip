# Start the PC worker with worker.env loaded.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_pc_worker.ps1 [-Port 8100]
param([int]$Port = 8100)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "worker.env")) {
    Write-Error "worker.env not found — run scripts\make_worker_env.ps1 first."
}

foreach ($line in Get-Content "worker.env") {
    if ($line -match "^\s*#" -or $line -notmatch "=") { continue }
    $name, $value = $line -split "=", 2
    Set-Item -Path "env:$name" -Value $value
}

& .venv\Scripts\python.exe -m nexoclip.cli worker --port $Port
