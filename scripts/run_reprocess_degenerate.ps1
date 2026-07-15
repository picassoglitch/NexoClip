# Run the degenerate-queue reprocess with worker.env loaded.
# Usage: powershell -File scripts\run_reprocess_degenerate.ps1 -Tenant ten_... [-Live]
param(
    [Parameter(Mandatory = $true)][string]$Tenant,
    [switch]$Live
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

foreach ($line in Get-Content "worker.env") {
    if ($line -match "^\s*#" -or $line -notmatch "=") { continue }
    $n, $v = $line -split "=", 2
    Set-Item -Path "env:$n" -Value $v
}

$flags = @("--tenant", $Tenant)
if ($Live) { $flags += "--live" }
& .venv\Scripts\python.exe scripts\reprocess_degenerate_queue.py @flags
