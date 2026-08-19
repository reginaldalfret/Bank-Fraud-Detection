# deployment/windows/start_server.ps1
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\..\.."
Set-Location $ProjectRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STARTING BANK FRAUD CLASSIFICATION PRODUCTION SERVER" -ForegroundColor Green
Write-Host "Project Root : $ProjectRoot" -ForegroundColor Gray
Write-Host "Port         : 8050" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Cyan

$PythonExe = "$ProjectRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment python not found at $PythonExe"
    exit 1
}

$env:HOST = "0.0.0.0"
$env:PORT = "8050"
$env:DATABASE_URL = "sqlite:///./api/fraud_api.db"
$env:JWT_SECRET = "production-secret-baf-classification-2026-secure"
$env:MFA_REQUIRED = "false"
$env:ENFORCE_MODEL_CHECKSUM = "false"

& $PythonExe "$ProjectRoot\src\serve.py"
