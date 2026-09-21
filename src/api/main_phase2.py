"""
FastAPI backend — Phase 2 (Vision + LLM Treatment Plans).
Prompt 2.5 — Updated FastAPI Backend (Phase 2)
"""

import io
import base64
import json
import time
import os
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from PIL import Image

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.vision.model    import EfficientNetB4Classifier

from src.vision.preprocess import VAL_TRANSFORM

# Phase env flag — set PHASE=1 to disable LLM (CPU deployment)
PHASE = int(os.getenv("PHASE", "2"))

# ── Allowed values ────────────────────────────────────────────────────────────
VALID_REGIONS = [
    "North India", "South India", "East India", "West India", "Central India",
]
VALID_SEASONS = [
    "Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)",
]

# ── Global State ──────────────────────────────────────────────────────────────
VISION_MODEL = None
ADVISOR      = None
CLASS_NAMES: List[str] = []
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH       = Path(os.getenv("MODEL_PATH", "models/vision/efficientnet_b4_best.pt"))
ADAPTER_PATH     = Path(os.getenv("ADAPTER_PATH", "models/llm/qwen2.5_3b_qlora_adapter"))
CLASS_NAMES_PATH = Path("data/processed/class_names.json")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global VISION_MODEL, ADVISOR, CLASS_NAMES

    # Class names
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH) as f:
            m = json.load(f)
            # Critical Fix: The vision model was trained on folders numbered "0" to "37".
            # PyTorch ImageFolder sorts folders lexicographically, so the node order is:
            # 0, 1, 10, 11, 12 ... 19, 2, 20 ...
            # We must map the logit indices to the correct original folder key.
            sorted_folder_names = sorted(list(m.keys()), key=str)
            CLASS_NAMES = [m[folder] for folder in sorted_folder_names]
    else:
        CLASS_NAMES = [f"class_{i}" for i in range(38)]

    # Vision model — load from disk (committed via git-lfs)
    if not MODEL_PATH.exists():
        print(f"  ⚠  Model not found at {MODEL_PATH}. Using random weights.")


    VISION_MODEL = EfficientNetB4Classifier(num_classes=len(CLASS_NAMES), pretrained=False).to(DEVICE)
    if MODEL_PATH.exists():
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        VISION_MODEL.load_state_dict(ckpt["state_dict"])
    VISION_MODEL.eval()
    print(f"✓ Vision model ready | Device: {DEVICE}")

    # LLM advisor (Phase 2 only — async, non-blocking)
    if PHASE == 2:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _load_advisor)
        except Exception as e:
            print(f"⚠ LLM load failed ({e}). Running in Phase 1 fallback mode.")

    yield


def _load_advisor():
    global ADVISOR
    from src.llm.advisor import CropDiseaseAdvisor
    ADVISOR = CropDiseaseAdvisor(
        base_model   = os.getenv("BASE_LLM", "Qwen/Qwen2.5-3B-Instruct"),
        adapter_path = str(ADAPTER_PATH),
    )


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crop Disease Advisor API — Phase 2",
    description="EfficientNet-B4 vision diagnosis + LLaMA 3 QLoRA treatment planning.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Serve Vite static build in production (when frontend/dist exists) ─────────
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(str(_DIST / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Catch-all: serve index.html for any non-API path (SPA routing)."""
        file_path = _DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_DIST / "index.html"))


# ── Middleware ─────────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    elapsed  = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time"] = f"{elapsed:.2f}ms"
    return response


# ── Pydantic Models ───────────────────────────────────────────────────────────
class Top5Pred(BaseModel):
    label:      str
    confidence: float

class TreatmentEntry(BaseModel):
    method:      Optional[str] = None
    application: Optional[str] = None
    frequency:   Optional[str] = None
    product:     Optional[str] = None
    dosage:      Optional[str] = None
    timing:      Optional[str] = None
    safety_note: Optional[str] = None

class TreatmentPlan(BaseModel):
    organic_treatments:  list
    chemical_treatments: list
    preventive_measures: list
    yield_impact_estimate: str
    action_urgency:      str
    regional_notes:      str
    seasonal_notes:      str

class PredictResponse(BaseModel):
    disease:        str
    crop:           str
    confidence:     float
    severity:       str
    top5:           List[Top5Pred]
    treatment_plan: Optional[TreatmentPlan] = None
    inference_ms:   float


# ── Helpers ───────────────────────────────────────────────────────────────────
VALID_MIME = {"image/jpeg", "image/png", "image/jpg"}

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def extract_crop(label: str) -> str:
    return label.split("___")[0].replace("_", " ") if "___" in label else label




# ── Vision-Only Predict (Phase 1 fallback endpoint) ───────────────────────────
async def _run_vision(file: UploadFile) -> dict:
    if file.content_type not in VALID_MIME:
        raise HTTPException(422, f"Invalid type: {file.content_type}")
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(422, "Cannot decode image.")

    tensor = VAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = VISION_MODEL(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top5_idx  = probs.topk(5).indices.tolist()
    top5_conf = probs.topk(5).values.tolist()
    pred_idx  = top5_idx[0]
    conf      = top5_conf[0]

    return {
        "image":       image,
        "tensor":      tensor,
        "pred_idx":    pred_idx,
        "confidence":  conf,
        "top5_idx":    top5_idx,
        "top5_conf":   top5_conf,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":   "healthy",
        "model":    "efficientnet_b4",
        "classes":  len(CLASS_NAMES),
        "device":   DEVICE,
        "phase":    PHASE if ADVISOR or PHASE == 1 else "1-fallback",
        "llm_ready": ADVISOR is not None,
    }

@app.get("/classes")
def get_classes():
    return CLASS_NAMES

@app.get("/regions")
def get_regions():
    return VALID_REGIONS

@app.get("/seasons")
def get_seasons():
    return VALID_SEASONS


@app.post("/predict/vision-only")
async def predict_vision_only(file: UploadFile = File(...)):
    """Phase 1-compatible endpoint — vision only, no treatment plan."""
    t0  = time.perf_counter()
    vis = await _run_vision(file)
    disease    = CLASS_NAMES[vis["pred_idx"]]
    confidence = vis["confidence"]
    return {
        "disease":        disease,
        "crop":           extract_crop(disease),
        "confidence":     round(confidence * 100, 2),
        "severity":       "Unknown",
        "top5":           [{"label": CLASS_NAMES[i], "confidence": round(c * 100, 2)}
                           for i, c in zip(vis["top5_idx"], vis["top5_conf"])],
        "inference_ms":   round((time.perf_counter() - t0) * 1000, 2),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file:   UploadFile = File(...),
    region: str        = Form(default="North India"),
    season: str        = Form(default="Kharif (Monsoon)"),
    severity: str      = Form(default="Moderate"),
):
    """Phase 2 endpoint — vision + LLM treatment plan."""
    if region not in VALID_REGIONS:
        raise HTTPException(422, f"Invalid region '{region}'. Valid: {VALID_REGIONS}")
    if season not in VALID_SEASONS:
        raise HTTPException(422, f"Invalid season '{season}'. Valid: {VALID_SEASONS}")

    t0  = time.perf_counter()
    vis = await _run_vision(file)

    disease    = CLASS_NAMES[vis["pred_idx"]]
    confidence = vis["confidence"]
    crop       = extract_crop(disease)

    if "healthy" in disease.lower():
        severity = "None (Healthy)"
    
    top5       = [Top5Pred(label=CLASS_NAMES[i], confidence=round(c * 100, 2))
                  for i, c in zip(vis["top5_idx"], vis["top5_conf"])]

    # Treatment plan — LLM (Phase 2) or DISEASE_DB (Phase 1 / CPU fallback)
    treatment_plan = None
    if ADVISOR is not None:
        raw_plan = ADVISOR.generate_treatment_plan(
            disease=disease.replace("___", " — ").replace("_", " "),
            crop=crop, region=region, season=season, severity=severity,
        )
    else:
        # Phase 1 / CPU mode — use curated DISEASE_DB directly (instant, no LLM)
        try:
            from src.llm.generate_dataset import (
                DISEASE_DB, REGIONAL_NOTES, SEASONAL_NOTES, URGENCY_MAP, normalise_disease
            )

            disease_norm = normalise_disease(disease)
            db = DISEASE_DB.get(disease_norm, {})

            severity_long = {
                "Mild":     "Mild (0–30% infection)",
                "Moderate": "Moderate (30–60% infection)",
                "Severe":   "Severe (60–100% infection)",
            }.get(severity.split(" ")[0], "Moderate (30–60% infection)")

            reg_note = next(
                (v for k, v in REGIONAL_NOTES.items() if region.split()[0].lower() in k.lower()),
                "Follow local ICAR recommendations for your region."
            )
            sea_note = next(
                (v for k, v in SEASONAL_NOTES.items() if season.split()[0].lower() in k.lower()),
                "Refer to seasonal crop advisory bulletins."
            )

            raw_plan = {
                "organic_treatments":  [
                    {"method": t["method"], "application": t["application"], "frequency": t["frequency"]}
                    for t in db.get("organic", [])
                ],
                "chemical_treatments": [
                    {"product": t["product"], "dosage": t["dosage"],
                     "safety_note": t.get("safety_note", t.get("timing", ""))}
                    for t in db.get("chemical", [])
                ],
                "preventive_measures":   db.get("preventive", []),
                "yield_impact_estimate": db.get("yield_impact", "Consult local agronomist."),
                "action_urgency":        URGENCY_MAP.get(severity_long, "Within 3 days"),
                "regional_notes":        reg_note,
                "seasonal_notes":        sea_note,
            }
        except Exception as e:
            print(f"[DISEASE_DB] fallback error: {e}")
            raw_plan = None

    if raw_plan:
        treatment_plan = TreatmentPlan(
            organic_treatments=raw_plan.get("organic_treatments", []),
            chemical_treatments=raw_plan.get("chemical_treatments", []),
            preventive_measures=raw_plan.get("preventive_measures", []),
            yield_impact_estimate=raw_plan.get("yield_impact_estimate", ""),
            action_urgency=raw_plan.get("action_urgency", ""),
            regional_notes=raw_plan.get("regional_notes", ""),
            seasonal_notes=raw_plan.get("seasonal_notes", ""),
        )

    return PredictResponse(
        disease=disease,
        crop=crop,
        confidence=round(confidence * 100, 2),
        severity=severity,
        top5=top5,
        treatment_plan=treatment_plan,
        inference_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
