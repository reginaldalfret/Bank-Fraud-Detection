# PRODUCTION DEPLOYMENT & OPERATIONS GUIDE

## Architecture Overview
- **Deployment Platform:** Windows 11 Host Laptop.
- **Unified Service Port:** `8050` (FastAPI REST API + Static Dashboard).
- **Public Domain:** `https://frauddetection.reginaldalfret.tech` (via Cloudflare Tunnel).
- **Auto-Start & Recovery:** Managed via Windows Task Scheduler + PowerShell Watchdog.

---

## 1. Quickstart (Manual Launch)

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Start unified server
python src\serve.py
```
Open your browser at `http://localhost:8050/` or `https://frauddetection.reginaldalfret.tech`.

---

## 2. Windows Auto-Start & Crash Recovery Installation

To ensure the server starts automatically when Windows starts and auto-recovers on failure:

```powershell
# Run with Administrator privileges in PowerShell:
powershell.exe -ExecutionPolicy Bypass -File "deployment\windows\install_windows_task.ps1"
powershell.exe -ExecutionPolicy Bypass -File "deployment\windows\install_tunnel_task.ps1"
```

### Registered Tasks:
1. **`BankFraudDetection_Watchdog`**: Pings `http://127.0.0.1:8050/api/health` every 15 seconds. If unresponsive, it terminates stale processes on port 8050 and restarts `src/serve.py`.
2. **`BankFraudDetection_CloudflareTunnel`**: Automatically establishes the secure Cloudflare tunnel routing `frauddetection.reginaldalfret.tech` to `http://127.0.0.1:8050`.

---

## 3. Production Health Check

```powershell
# Local Health Check
Invoke-RestMethod -Uri "http://127.0.0.1:8050/api/health"

# Public Ingress Health Check
Invoke-RestMethod -Uri "https://frauddetection.reginaldalfret.tech/api/health"
```
