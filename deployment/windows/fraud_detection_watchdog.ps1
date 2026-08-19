# deployment/windows/fraud_detection_watchdog.ps1
$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\..\.."
Set-Location $ProjectRoot

$HealthUrl = "http://127.0.0.1:8050/api/health"
$StartScript = "$ScriptDir\start_server.ps1"
$PythonExe = "$ProjectRoot\.venv\Scripts\python.exe"

Write-Host "Starting Bank Fraud Classification Watchdog Service..." -ForegroundColor Cyan

while ($true) {
    $healthy = $false
    try {
        $response = Invoke-RestMethod -Uri $HealthUrl -Method Get -TimeoutSec 5 -ErrorAction Stop
        if ($response.status -eq "HEALTHY" -or $response.status -eq "DEGRADED" -or $response.healthy -eq $true) {
            $healthy = $true
        }
    } catch {
        $healthy = $false
    }

    if (-not $healthy) {
        Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [ALERT] Service unhealthy or unreachable. Restarting server process..." -ForegroundColor Red
        
        # Kill any zombie process on port 8050
        try {
            $conn = Get-NetTCPConnection -LocalPort 8050 -ErrorAction SilentlyContinue
            if ($conn) {
                $pidToKill = $conn.OwningProcess | Select-Object -Unique
                Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue
            }
        } catch {}

        # Launch server process in background
        Start-Process -FilePath "powershell.exe" -ArgumentList "-ExecutionPolicy Bypass -NoProfile -File `"$StartScript`"" -WindowStyle Hidden
        Start-Sleep -Seconds 5
    } else {
        Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [HEALTHY] Service is operational." -ForegroundColor Green
    }

    Start-Sleep -Seconds 15
}
