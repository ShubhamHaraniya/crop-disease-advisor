"""
Local App Entrypoint — app.py
Launches the FastAPI backend and Streamlit UI locally.
"""

import os
import sys
from pathlib import Path

# ── Launch the Phase 2 API Server ──────────────────────────────────────
print("🌾 Crop Disease Advisor — API Server")
print("─" * 50)
print("\n🚀 Starting FastApi Backend on port 8000...")
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from src.api.main_phase2 import app as fastapi_app

if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")

