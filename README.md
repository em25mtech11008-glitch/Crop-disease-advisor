<div align="center">

# 🌾 Crop Disease Advisor

### AI-powered plant disease diagnosis & precision treatment planning for Indian farmers

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Render-46E3B7?style=for-the-badge)](https://crop-disease-advisor.onrender.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)

</div>

---

## 📌 Overview

**Crop Disease Advisor** is an end-to-end MLOps project that combines computer vision and large language models to help Indian farmers identify plant diseases from leaf photographs and receive structured, region-aware treatment plans.

Upload a photo → **instant diagnosis** across 38 disease classes → **structured treatment plan** with organic/chemical treatments, yield impact, and region/season advisory.

---

## 🌐 Links

| Resource | URL |
|---|---|
| 🚀 **Production App (Render)** | https://crop-disease-advisor.onrender.com/ |
| 💻 **Source Code (GitHub)** | https://github.com/em25mtech11008-glitch/Crop-disease-advisor |

---

## 📊 Results

| Model | Metric | Score |
|---|---|---|
| EfficientNet-B4 | Test Accuracy | **96.8%** |
| EfficientNet-B4 | F1 Macro | **95.4%** |
| EfficientNet-B4 | AUC (OvR) | **97.1%** |
| Qwen2.5-3B QLoRA | JSON Validity | **100%** |
| Qwen2.5-3B QLoRA | Schema Compliance | **100%** |
| Qwen2.5-3B QLoRA | BERTScore F1 | **81.68%** |
| Qwen2.5-3B QLoRA | Perplexity | **1.42** |

---

## 🏗️ Architecture

```
User (Browser)  →  Vite Frontend (HTML/JS/CSS)
                          │ REST API
                          ▼
              FastAPI Backend  [app.py → src/api/main_phase2.py]
                 │                          │
    ┌────────────▼──────────┐   ┌───────────▼──────────────────┐
    │    Vision Pipeline    │   │    Treatment Pipeline        │
    │  src/vision/model.py  │   │  PHASE=1 → DISEASE_DB        │
    │  EfficientNet-B4      │   │  (src/llm/generate_dataset)  │
    │  38 disease classes   │   │  PHASE=2 → Qwen2.5-3B QLoRA  │
    └───────────────────────┘   │  (src/llm/advisor.py)        │
                                └──────────────────────────────┘
```

---

## 📁 Repository Structure

```
Crop-disease-advisor/
│
├── app.py                              # Uvicorn entrypoint — loads .env, reads PORT
├── app/
│   └── chatbot.py                      # Streamlit LLM chatbot (GPU presentation)
│
├── src/                                # Core library — imported by scripts and API
│   ├── api/
│   │   └── main_phase2.py              # FastAPI routes: /predict /health /classes
│   ├── vision/
│   │   ├── model.py                    # EfficientNetB4Classifier class definition
│   │   └── preprocess.py               # TRAIN_TRANSFORM, VAL_TRANSFORM pipelines
│   ├── llm/
│   │   ├── advisor.py                  # CropDiseaseAdvisor: LLM inference + DISEASE_DB fallback
│   │   ├── generate_dataset.py         # DISEASE_DB knowledge base + dataset generator
│   │   └── evaluate_llm.py             # Two-mode LLM evaluation (JSON + text metrics)
│   └── data/
│       ├── download_plantvillage.py    # Dataset downloader — 4 fallback methods
│       └── convert_parquet.py          # HuggingFace Parquet → ImageFolder converter
│
├── scripts/                            # CLI runners — call src/ library code
│   ├── training/
│   │   ├── train_vision.py             # EfficientNet-B4 training loop with W&B
│   │   ├── evaluate_vision.py          # Full test-set eval → eval/eval_results.json
│   │   ├── train_qlora.py              # Qwen2.5 QLoRA fine-tuning with SFTTrainer
│   │   └── test_qlora.py               # Quick LLM adapter smoke test (10 prompts)
│   ├── data/
│   │   ├── download_plantvillage.py    # CLI wrapper for dataset download
│   │   └── convert_parquet.py          # CLI wrapper for parquet conversion
│   └── ops/
│       ├── upload_models.py            # Upload vision + LLM models to a model hub
│       └── register_model.py           # W&B registration with accuracy gate
│
├── tests/                              # pytest suite (no GPU / checkpoint needed)
│   ├── conftest.py                     # Shared fixtures: leaf image, model, tensors
│   ├── test_api.py                     # FastAPI endpoint tests (mocked model)
│   ├── test_vision_model.py            # Model forward pass, shape, NaN/Inf checks
│   ├── test_llm_advisor.py             # Advisor schema validation, retry, fallback
│   └── test_preprocessing.py           # Transform shape, dtype, normalization range
│
├── configs/
│   └── vision_config.yaml              # EfficientNet-B4 hyperparameters
│
├── eval/
│   ├── eval_results.json               # Vision test-set metrics
│   └── llm_eval_run.log                # LLM evaluation log
│
├── models/
│   ├── vision/
│   │   └── efficientnet_b4_best.pt     # Trained checkpoint (~68 MB)
│   └── llm/
│       └── qwen2.5_3b_qlora_adapter/   # QLoRA adapter weights (safetensors + tokenizer)
│
├── data/
│   └── processed/
│       └── class_names.json            # 38 disease class index → label mapping
│
├── frontend/                           # Vite SPA
│   ├── index.html                      # App shell
│   ├── app.js                          # UI logic: upload, predict, results
│   ├── style.css                       # Glassmorphic dark theme
│   ├── vite.config.js                  # Vite build config
│   └── dist/                           # Pre-built production bundle
│
├── report/
│   └── project_report.html             # Full MLOps project report (open → Print → PDF)
│
├── requirements.txt                    # CPU-only deployment deps (Render / Docker)
├── requirements_training.txt           # Full GPU deps (training + chatbot)
├── Dockerfile                          # Multi-stage build: Python + Node + Vite
├── .dockerignore                       # Excludes node_modules, data/raw, tests, scripts
├── .env                                # Local env vars (not committed)
└── .env.example                        # Template: PHASE, HF_TOKEN, WANDB_API_KEY
```

---

## 🚀 Run the App

### Local (CPU)
```bash
git clone https://github.com/em25mtech11008-glitch/Crop-disease-advisor.git
cd Crop-disease-advisor

pip install -r requirements.txt
# Windows:
set PHASE=1 && python app.py
# Linux/Mac:
PHASE=1 python app.py
# → API: http://localhost:8000
```

### Frontend Dev Server
```bash
cd frontend
npm install
npm run dev       # → http://localhost:5173
```

### Docker
```bash
docker build -t crop-disease-advisor .
docker run -p 8080:8080 -e PHASE=1 crop-disease-advisor
```

### LLM Chatbot (GPU required)
```bash
pip install -r requirements_training.txt
streamlit run app/chatbot.py
```

---

## 📋 Script Reference

### 🔵 Core Library (`src/`)

| File | What it does |
|---|---|
| `src/api/main_phase2.py` | FastAPI app. Loads EfficientNet-B4 at startup via lifespan event. Handles `POST /predict` (image → disease + treatment), `GET /health`, `GET /classes`. Serves `frontend/dist/` as static files. Switches between DISEASE_DB (PHASE=1) and LLM advisor (PHASE=2) at runtime. |
| `src/vision/model.py` | Defines `EfficientNetB4Classifier`: EfficientNet-B4 backbone (timm) + custom 2-layer MLP head (1792→512→38). Exposes `forward_features()` for spatial feature maps and `get_target_layer()` for hook-based visualization. |
| `src/vision/preprocess.py` | Defines `TRAIN_TRANSFORM` (RandomResizedCrop, HFlip, VFlip, Rotation, ColorJitter, Normalize) and `VAL_TRANSFORM` (Resize 256 → CenterCrop 224 → Normalize). Uses ImageNet mean/std. |
| `src/llm/advisor.py` | `CropDiseaseAdvisor`: loads Qwen2.5-3B + QLoRA adapter (4-bit NF4), formats chat prompts, generates structured JSON treatment plans, retries with lower temperature on parse failure, falls back to `DISEASE_DB` if both attempts fail. |
| `src/llm/generate_dataset.py` | Two roles: (1) **Runtime** — `DISEASE_DB` dict with curated organic/chemical/preventive treatments for all 38 diseases, used as production fallback; (2) **Training** — generates the instruction-tuning dataset by cross-producting diseases × regions × seasons × farmer profiles. |
| `src/llm/evaluate_llm.py` | LLM evaluation suite. **Eval 1 (JSON mode):** JSON validity, schema compliance (6 required keys), field completeness, perplexity. **Eval 2 (Text mode):** BLEU-4, ROUGE-L, BERTScore F1 (RoBERTa-large), semantic similarity (MiniLM). Saves to `outputs/llm_eval_results.json`. |
| `src/data/download_plantvillage.py` | Downloads PlantVillage with 4 automatic fallbacks: (1) HuggingFace `datasets`, (2) HF `snapshot_download`, (3) Kaggle API, (4) direct `wget`. Verifies class and image counts. |
| `src/data/convert_parquet.py` | Converts manually-downloaded Parquet files to a PyTorch `ImageFolder` structure. Auto-detects image and label columns; handles binary bytes and dict-format images. |

### 🟢 Training Scripts (`scripts/training/`)

| File | What it does |
|---|---|
| `train_vision.py` | **EfficientNet-B4 training pipeline.** Reads `configs/vision_config.yaml`. 3-phase gradual unfreezing (head → last 2 MBConv → full model). AMP (FP16), CosineAnnealingWarmRestarts, label smoothing, early stopping. Logs every epoch to W&B. Saves best val_acc checkpoint to `models/vision/efficientnet_b4_best.pt`. |
| `evaluate_vision.py` | **Test-set evaluation.** Runs the best checkpoint on the hold-out split. Computes accuracy, F1 macro/weighted, AUC (macro OvR), and per-class accuracy. Saves to `eval/eval_results.json`. |
| `train_qlora.py` | **Qwen2.5-3B QLoRA fine-tuning.** Reads `configs/llm_config.yaml`. 4-bit NF4 base, LoRA rank 16, TRL `SFTTrainer`. `JSONValidityCallback` logs JSON parse success to W&B every 500 steps. Saves adapter to `models/llm/qwen2.5_3b_qlora_adapter/`. |
| `test_qlora.py` | **Adapter smoke test.** Loads base model + adapter and runs 10 standard prompts (e.g., Tomato Late Blight, Apple Scab, Potato Early Blight) to confirm coherent JSON output. |

### 🟡 Data Scripts (`scripts/data/`)

| File | What it does |
|---|---|
| `download_plantvillage.py` | CLI wrapper for dataset download. Usage: `python scripts/data/download_plantvillage.py --method hf` |
| `convert_parquet.py` | CLI wrapper for Parquet conversion. Usage: `python scripts/data/convert_parquet.py --parquet_dir downloads/` |

### 🔴 MLOps / Ops Scripts (`scripts/ops/`)

| File | What it does |
|---|---|
| `upload_models.py` | Uploads the vision checkpoint + `class_names.json` and the QLoRA adapter folder to a model hub repo (created automatically if missing). Requires `HF_TOKEN`. |
| `register_model.py` | **MLOps gating pipeline.** Checks a promotion accuracy gate (default ≥ 85%; use `--min_accuracy 0.98` for stricter). If passed, logs the model as a W&B artifact with eval metadata (accuracy, F1, AUC, epoch), tagged `staging` or `production`, and uploads it with eval metrics in the commit message. |

---

## 🧪 Test Suite (`tests/`)

Runs **without a GPU and without model checkpoints**. `conftest.py` builds a synthetic leaf image and an EfficientNet-B4 with random weights — tests verify architecture and logic, not accuracy.

**`conftest.py` fixtures:** `sample_leaf_image` (synthetic 224×224 green image) · `sample_tensor` (normalized `(1,3,224,224)`) · `class_names` (real JSON or dummy fallback) · `vision_model` (random weights, eval mode)

**`test_vision_model.py` — 8 tests**

| Test | Checks |
|---|---|
| `test_forward_pass_shape` | Logits are exactly `(1, 38)` |
| `test_confidence_sum` | Softmax probabilities sum to 1.0 |
| `test_output_dtype` | Logits are `float32` |
| `test_forward_features_shape` | Feature map is 4-D `(1, C, h, w)` |
| `test_get_target_layer` | Hook target is a valid `nn.Module` |
| `test_gradcam_output_shape` | GradCAM heatmap is 2-D `float32` |
| `test_gradcam_range` | GradCAM values in `[0.0, 1.0]` |
| `test_no_nan_in_output` | No NaN/Inf in logits |

**`test_preprocessing.py` — 6 tests**

| Test | Checks |
|---|---|
| `test_output_shape` | `VAL_TRANSFORM` → `(3, 224, 224)` |
| `test_normalization_range` | Values stay in `[-4, 4]` |
| `test_output_dtype` | Output is `float32` |
| `test_augmentation_determinism` | Same seed → identical output |
| `test_batch_consistency` | Different inputs → different outputs |
| `test_channels_first` | Tensor is `(C, H, W)` |

**`test_llm_advisor.py` — 5 tests** (model mocked with `unittest.mock`)

| Test | Checks |
|---|---|
| `test_output_schema` | Plan dict has all 10 required keys |
| `test_validate_output_passes` | `validate_output()` is `True` for a complete plan |
| `test_validate_output_fails_on_missing_key` | `False` when `action_urgency` is missing |
| `test_json_retry_on_parse_error` | Bad JSON → 3 retries → falls back to DISEASE_DB |
| `test_urgency_is_valid` | `action_urgency` is one of the 4 valid levels |

**`test_api.py` — 7 tests** (`TestClient`, model fully mocked)

| Test | Checks |
|---|---|
| `test_health_endpoint` | `GET /health` → 200, `status: healthy` |
| `test_classes_endpoint` | `GET /classes` returns a list |
| `test_predict_valid_png` | PNG → disease, confidence, top5 |
| `test_predict_valid_jpeg` | JPEG → 200 |
| `test_predict_invalid_text_file` | `.txt` → 422 |
| `test_predict_invalid_pdf` | PDF → 422 |
| `test_confidence_is_percentage` | Confidence in `[0, 100]` |

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## ⚙️ Configuration

### `configs/vision_config.yaml`

```yaml
model: efficientnet_b4
num_classes: 38
image_size: 224
batch_size: 32
epochs: 50
lr_phase1: 1.0e-3    # classifier head only
lr_phase2: 5.0e-4    # + last 2 MBConv blocks
lr_phase3: 1.0e-4    # full model
unfreeze_schedule:
  phase1_end: 15
  phase2_end: 30
early_stop_patience: 10
checkpoint_dir: models/vision
wandb_project: crop-disease-advisor
```

> ⚠️ `configs/llm_config.yaml` is not committed (contains model paths and API keys). Create it from the parameters documented in `scripts/training/train_qlora.py` before running LLM training.

---

## 📥 MLOps Workflow

```
Step 1   Download dataset
         python src/data/download_plantvillage.py --method hf

Step 2   Train vision model
         python scripts/training/train_vision.py --config configs/vision_config.yaml

Step 3   Evaluate vision model
         python scripts/training/evaluate_vision.py --config configs/vision_config.yaml

Step 4   Generate LLM instruction dataset
         python src/llm/generate_dataset.py

Step 5   Fine-tune LLM (GPU)
         python scripts/training/train_qlora.py --config configs/llm_config.yaml

Step 6   Test LLM adapter
         python scripts/training/test_qlora.py

Step 7   Evaluate LLM
         python src/llm/evaluate_llm.py

Step 8   Register model with accuracy gate
         python scripts/ops/register_model.py \
           --checkpoint models/vision/efficientnet_b4_best.pt \
           --eval_path eval/eval_results.json \
           --stage production --min_accuracy 0.98

Step 9   (Optional) Upload models to a hub
         python scripts/ops/upload_models.py --username <your-hub-username>

Step 10  Deploy
         git push origin main   # triggers Render auto-deploy
```

---

## ⚙️ Environment Variables

| Variable | Value | Description |
|---|---|---|
| `PHASE` | `1` (default) | `1` = CPU + DISEASE_DB; `2` = LLM inference |
| `PORT` | auto | Injected by Render / Cloud Run |
| `HF_TOKEN` | your token | Model hub API token (only for upload/download) |
| `WANDB_API_KEY` | your key | Weights & Biases API key |

Copy `.env.example` → `.env` and fill in values for local development.

---

## 🌿 Supported Crops & Diseases (38 Classes)

| Crop | Diseases |
|---|---|
| 🍎 Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🫐 Blueberry | Healthy |
| 🍒 Cherry | Powdery Mildew, Healthy |
| 🌽 Corn | Cercospora/Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍇 Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| 🍊 Orange | Haunglongbing (Citrus Greening) |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🫑 Pepper | Bacterial Spot, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🍓 Raspberry | Healthy |
| 🫘 Soybean | Healthy |
| 🎃 Squash | Powdery Mildew |
| 🍓 Strawberry | Leaf Scorch, Healthy |
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

**Regions:** North · South · East · West · Central India
**Seasons:** Kharif (Monsoon) · Rabi (Winter) · Zaid (Summer)

---

## 👤 Author

**Parth Vekariya** · [GitHub](https://github.com/em25mtech11008-glitch)
