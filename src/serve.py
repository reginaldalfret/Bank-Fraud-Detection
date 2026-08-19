# src/serve.py
"""
Unified Production Server for Supervised Bank Fraud Classification System.
Serves:
- REST API at /api/*
- Web Dashboard at /
- Interactive OpenAPI Docs at /docs
Host: 0.0.0.0, Port: 8050
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    host = os.environ.get("HOST", "0.0.0.0")
    print("=" * 70)
    print(f"STARTING BANK FRAUD CLASSIFICATION PRODUCTION SERVER ON http://{host}:{port}")
    print(f"Dashboard UI : http://localhost:{port}/")
    print(f"OpenAPI Docs : http://localhost:{port}/docs")
    print(f"API Health   : http://localhost:{port}/api/health")
    print("=" * 70)
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=1,
        log_level="info",
        access_log=True,
    )
