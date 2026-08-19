# deployment/windows/start_tunnel.ps1
$ErrorActionPreference = "Stop"

Write-Host "Starting Cloudflare Tunnel for frauddetection.reginaldalfret.tech..." -ForegroundColor Cyan

cloudflared tunnel --config "C:\Users\ASUS\.cloudflared\config.yml" run
