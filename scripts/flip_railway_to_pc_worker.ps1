# Point Railway prod at the PC worker (jobs + LLM) - the "no paid APIs" flip.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\flip_railway_to_pc_worker.ps1 -FunnelUrl https://<host>.ts.net
#
# Sets on the NexoClip service (triggers a redeploy):
#   NEXOCLIP_JOB_DISPATCHER=modal          (the dispatcher is protocol-generic)
#   NEXOCLIP_MODAL_PIPELINE_ENDPOINT_URL   -> the worker's tunnel URL
#   NEXOCLIP_OPENLLM_BASE_URL              -> <tunnel>/v1 (worker's Ollama proxy)
#   OPENLLM_API_KEY                        -> same shared secret as NEXOCLIP_MODAL_TOKEN
param(
    [Parameter(Mandatory = $true)][string]$FunnelUrl
)

$ErrorActionPreference = "Stop"
$base = $FunnelUrl.TrimEnd("/")

$app = railway variables --service NexoClip --json | ConvertFrom-Json
$token = $app.NEXOCLIP_MODAL_TOKEN
if (-not $token) { Write-Error "NEXOCLIP_MODAL_TOKEN missing on the NexoClip service" }

railway variables --service NexoClip `
    --set "NEXOCLIP_JOB_DISPATCHER=modal" `
    --set "NEXOCLIP_MODAL_PIPELINE_ENDPOINT_URL=$base/" `
    --set "NEXOCLIP_OPENLLM_BASE_URL=$base/v1" `
    --set "OPENLLM_API_KEY=$token" `
    --skip-deploys

Write-Output "Variables set. Trigger the deploy when ready: railway redeploy --service NexoClip"
