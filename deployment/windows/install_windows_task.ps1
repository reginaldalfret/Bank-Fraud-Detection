# deployment/windows/install_windows_task.ps1
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\..\.."
$WatchdogScript = "$ScriptDir\fraud_detection_watchdog.ps1"

$TaskName = "BankFraudDetection_Watchdog"

Write-Host "Registering Windows Scheduled Task '$TaskName' to ensure high-availability on Windows boot..." -ForegroundColor Cyan

# Define task action
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -File `"$WatchdogScript`""

# Define task trigger: At Windows logon & startup
$Trigger = New-ScheduledTaskTrigger -AtLogOn

# Define task settings: restart on failure, run indefinitely
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit 0 -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

# Register task
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Auto-starts and monitors the Supervised Bank Fraud Classification System" -Force

Write-Host "Successfully registered Windows Task: $TaskName" -ForegroundColor Green
Write-Host "Starting task immediately..." -ForegroundColor Cyan
Start-ScheduledTask -TaskName $TaskName
