"""
FastAPI endpoint tests using httpx async client.
Prompt 2.9 — test_api.py
"""

import io
import json
import pytest
import numpy as np
from PIL import Image

# httpx is needed for FastAPI TestClient
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_png_bytes(color=(34, 139, 34), size=(224, 224)) -> bytes:
    """Generate a synthetic PNG image as bytes."""
    arr = np.full((*size, 3), color, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_client():
    """
    Build a FastAPI TestClient with the vision model mocked so we don't
    need actual GPU / checkpoint files during CI.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    # Patch heavy dependencies before importing app
    with patch("src.api.main.EfficientNetB4Classifier") as MockModel, \
         patch("src.api.main.GradCAM") as MockCAM, \
         patch("src.api.main.torch.load", return_value={"state_dict": {}}):

        # Configure mock model to return valid logits
        mock_instance = MagicMock()
        mock_instance.return_value = __import__("torch").zeros(1, 38)
        mock_instance.eval.return_value = mock_instance
        mock_instance.get_target_layer.return_value = MagicMock()
        MockModel.return_value = mock_instance

        # Configure mock CAM
        mock_cam = MagicMock()
        mock_cam.generate_cam.return_value = np.zeros((7, 7), dtype=np.float32)
        mock_cam.overlay_heatmap.return_value = Image.new("RGB", (224, 224), (0, 255, 0))
        MockCAM.return_value = mock_cam

        # Import after patching
        from src.api.main import app
        client = TestClient(app, raise_server_exceptions=True)
        yield client


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAPIEndpoints:

    def test_health_endpoint(self, test_client):
        """GET /health must return 200 with status=healthy."""
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "model"   in data
        assert "classes" in data

    def test_classes_endpoint(self, test_client):
        """GET /classes must return a list."""
        resp = test_client.get("/classes")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_predict_valid_png(self, test_client):
        """POST /predict with valid PNG must return 200 with all required fields."""
        png_bytes = make_png_bytes()
        resp = test_client.post(
            "/predict",
            files={"file": ("leaf.png", png_bytes, "image/png")},
        )
        assert resp.status_code == 200, f"Unexpected: {resp.text}"
        data = resp.json()
        required = {"disease", "crop", "confidence", "top5", "heatmap_base64", "inference_ms"}
        assert required.issubset(data.keys()), f"Missing: {required - data.keys()}"

    def test_predict_valid_jpeg(self, test_client):
        """POST /predict with JPEG must also return 200."""
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        resp = test_client.post(
            "/predict",
            files={"file": ("leaf.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_predict_invalid_text_file(self, test_client):
        """POST /predict with a .txt file must return 422."""
        resp = test_client.post(
            "/predict",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 422

    def test_predict_invalid_pdf(self, test_client):
        """POST /predict with application/pdf must return 422."""
        resp = test_client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert resp.status_code == 422

    def test_process_time_header(self, test_client):
        """Every response must include X-Process-Time header."""
        resp = test_client.get("/health")
        assert "x-process-time" in resp.headers or "X-Process-Time" in resp.headers

    def test_confidence_is_percentage(self, test_client):
        """Confidence must be a float in [0, 100]."""
        png_bytes = make_png_bytes()
        resp = test_client.post(
            "/predict",
            files={"file": ("leaf.png", png_bytes, "image/png")},
        )
        if resp.status_code == 200:
            conf = resp.json()["confidence"]
            assert 0.0 <= conf <= 100.0, f"Confidence out of range: {conf}"
