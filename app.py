"""
App Entrypoint — app.py
Launches the FastAPI backend.
Locally: runs on port 8000.
Cloud Run/Render: reads PORT env variable.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# MUST load environment variables before importing the API so PHASE=1 applies
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from src.api.main_phase2 import app as fastapi_app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🌾 Crop Disease Advisor — starting on port {port} (PHASE={os.environ.get('PHASE')})")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port, log_level="info")
