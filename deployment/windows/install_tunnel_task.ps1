# deployment/windows/install_tunnel_task.ps1
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TunnelScript = "$ScriptDir\start_tunnel.ps1"
$TaskName = "BankFraudDetection_CloudflareTunnel"

Write-Host "Registering Windows Scheduled Task '$TaskName' to ensure Cloudflare tunnel auto-starts on boot..." -ForegroundColor Cyan

# Define task action
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -File `"$TunnelScript`""

# Define task trigger: At Windows logon & startup
$Trigger = New-ScheduledTaskTrigger -AtLogOn

# Define task settings: restart on failure, run indefinitely
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit 0 -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

# Register task
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Auto-starts Cloudflare Tunnel for frauddetection.reginaldalfret.tech" -Force

Write-Host "Successfully registered Windows Task: $TaskName" -ForegroundColor Green
Write-Host "Starting tunnel task immediately..." -ForegroundColor Cyan
Start-ScheduledTask -TaskName $TaskName
