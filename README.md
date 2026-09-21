<div align="center">

# 🌾 Crop Disease Advisor

### AI-powered plant disease diagnosis & precision treatment planning for Indian farmers

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Render-46E3B7?style=for-the-badge)](https://crop-disease-advisor.onrender.com/)
[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/spidey1807/crop-disease-advisor)
[![Vision Model](https://img.shields.io/badge/🤗%20EfficientNet--B4-Model-blue?style=for-the-badge)](https://huggingface.co/spidey1807/crop-disease-efficientnet-b4)
[![LLM Model](https://img.shields.io/badge/🤗%20Qwen2.5--3B%20QLoRA-Model-purple?style=for-the-badge)](https://huggingface.co/spidey1807/crop-disease-qwen2.5-qlora)

[![W&B Vision](https://img.shields.io/badge/W%26B-Vision%20Training-FFBE00?style=flat-square&logo=weightsandbiases)](https://wandb.ai/spharaniya18-intelkit-solutions/crop-disease-advisor/workspace?nw=nwuserspharaniya18)
[![W&B LLM](https://img.shields.io/badge/W%26B-LLM%20Training-FFBE00?style=flat-square&logo=weightsandbiases)](https://wandb.ai/spharaniya18-intelkit-solutions/crop-disease-advisor-llm/workspace?nw=nwuserspharaniya18)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

</div>

---

## 📌 Overview

**Crop Disease Advisor** is a full end-to-end MLOps project combining computer vision and large language models to help Indian farmers identify plant diseases from leaf photographs and receive structured, region-aware treatment plans.

Upload a photo → **instant diagnosis** across 38 disease classes → **structured treatment plan** with organic/chemical treatments, yield impact, and region/season advisory.

---

## 🌐 Live Links

| Resource | URL |
|---|---|
| 🚀 **Production App (Render)** | https://crop-disease-advisor.onrender.com/ |
| 🤗 **HuggingFace Space** | https://huggingface.co/spaces/spidey1807/crop-disease-advisor |
| 🧠 **EfficientNet-B4 Model** | https://huggingface.co/spidey1807/crop-disease-efficientnet-b4 |
| 🦙 **Qwen2.5-3B QLoRA Adapter** | https://huggingface.co/spidey1807/crop-disease-qwen2.5-qlora |
| 📊 **W&B Vision Training** | https://wandb.ai/spharaniya18-intelkit-solutions/crop-disease-advisor |
| 📊 **W&B LLM Training** | https://wandb.ai/spharaniya18-intelkit-solutions/crop-disease-advisor-llm |

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

## 📁 Complete Repository Structure

```
crop-disease-advisor/
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
│   │   └── preprocess.py              # TRAIN_TRANSFORM, VAL_TRANSFORM pipelines
│   ├── llm/
│   │   ├── advisor.py                  # CropDiseaseAdvisor: LLM inference + DISEASE_DB fallback
│   │   ├── generate_dataset.py         # DISEASE_DB knowledge base + dataset generator
│   │   └── evaluate_llm.py            # Two-mode LLM evaluation (JSON + text metrics)
│   └── data/
│       ├── download_plantvillage.py    # Dataset downloader — 4 fallback methods
│       └── convert_parquet.py         # HuggingFace Parquet → ImageFolder converter
│
├── scripts/                            # CLI runners — call src/ library code
│   ├── training/
│   │   ├── train_vision.py             # EfficientNet-B4 training loop with W&B
│   │   ├── evaluate_vision.py          # Full test-set eval → eval/eval_results.json
│   │   ├── train_qlora.py              # Qwen2.5 QLoRA fine-tuning with SFTTrainer
│   │   └── test_qlora.py              # Quick LLM adapter smoke test (10 prompts)
│   ├── data/
│   │   ├── download_plantvillage.py    # CLI wrapper for dataset download
│   │   └── convert_parquet.py         # CLI wrapper for parquet conversion
│   └── ops/
│       ├── upload_models.py            # Upload vision + LLM models to HuggingFace Hub
│       └── register_model.py          # W&B + HF registration with accuracy gate
│
├── tests/                              # pytest test suite (no GPU / checkpoint needed)
│   ├── conftest.py                     # Shared fixtures: leaf image, model, tensors
│   ├── test_api.py                     # FastAPI endpoint tests (mocked model)
│   ├── test_vision_model.py           # Model forward pass, shape, NaN/Inf checks
│   ├── test_llm_advisor.py            # Advisor schema validation, retry, fallback
│   └── test_preprocessing.py          # Transform shape, dtype, normalization range
│
├── configs/
│   └── vision_config.yaml             # EfficientNet-B4 hyperparameters
│
├── eval/
│   ├── eval_results.json              # Vision: 99.69% acc, 99.58% F1, AUC 1.0
│   └── llm_eval_run.log               # LLM: 100% JSON validity, 81.68% BERTScore
│
├── models/
│   ├── vision/
│   │   └── efficientnet_b4_best.pt    # Trained checkpoint (~68 MB)
│   └── llm/
│       └── qwen2.5_3b_qlora_adapter/  # QLoRA adapter weights (safetensors + tokenizer)
│
├── data/
│   └── processed/
│       └── class_names.json           # 38 disease class index → label mapping
│
├── frontend/                          # Vite SPA
│   ├── index.html                     # App shell
│   ├── app.js                         # UI logic: upload, predict, results
│   ├── style.css                      # Glassmorphic dark theme
│   ├── vite.config.js                 # Vite build config
│   └── dist/                          # Pre-built production bundle
│
├── report/
│   └── project_report.html           # Full MLOps project report (open → Print → PDF)
│
├── requirements.txt                   # CPU-only deployment deps (Render / Docker)
├── requirements_training.txt          # Full GPU deps (training + chatbot)
├── Dockerfile                         # Multi-stage build: Python + Node + Vite
├── .dockerignore                      # Excludes node_modules, data/raw, tests, scripts
├── .env                               # Local env vars (not committed)
└── .env.example                       # Template: PHASE, HF_TOKEN, WANDB_API_KEY
```

---

## 🚀 Run the App

### Local (CPU)
```bash
git clone https://github.com/ShubhamHaraniya/crop-disease-advisor.git
cd crop-disease-advisor

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

## 📋 Script Reference — Every File Explained

### 🔵 Core Library (`src/`)

| File | What it does |
|---|---|
| `src/api/main_phase2.py` | FastAPI app. Loads EfficientNet-B4 at startup via lifespan event. Handles `POST /predict` (image → disease + treatment), `GET /health`, `GET /classes`. Serves `frontend/dist/` as static files. Switches between DISEASE_DB (PHASE=1) and LLM advisor (PHASE=2) at runtime. |
| `src/vision/model.py` | Defines `EfficientNetB4Classifier`: EfficientNet-B4 backbone (timm) + custom 2-layer MLP head (1792→512→38). Exposes `forward_features()` for spatial feature maps and `get_target_layer()` for hook-based visualization. |
| `src/vision/preprocess.py` | Defines `TRAIN_TRANSFORM` (RandomResizedCrop, HFlip, VFlip, Rotation, ColorJitter, Normalize) and `VAL_TRANSFORM` (Resize 256 → CenterCrop 224 → Normalize). Uses ImageNet mean/std. |
| `src/llm/advisor.py` | `CropDiseaseAdvisor` class: loads Qwen2.5-3B + QLoRA adapter (4-bit NF4), formats LLaMA-3 chat prompts, generates structured JSON treatment plans, retries with lower temperature on parse failure, falls back to `DISEASE_DB` if both attempts fail. |
| `src/llm/generate_dataset.py` | Two roles: (1) **Runtime** — `DISEASE_DB` dict with expert-curated organic/chemical/preventive treatments for all 38 diseases, used as production fallback; (2) **Training** — generates instruction-tuning dataset by cross-producting diseases × regions × seasons × farmer profiles. |
| `src/llm/evaluate_llm.py` | Full LLM evaluation suite. **Eval 1 (JSON mode):** JSON validity, schema compliance (6 required keys), field completeness, perplexity. **Eval 2 (Text mode):** BLEU-4, ROUGE-L, BERTScore F1 (RoBERTa-large), semantic similarity (MiniLM). Saves to `outputs/llm_eval_results.json`. |
| `src/data/download_plantvillage.py` | Downloads PlantVillage dataset with 4 automatic fallback methods: (1) HuggingFace `datasets` library, (2) HF `snapshot_download`, (3) Kaggle API, (4) direct `wget`. Verifies downloaded class count and image count. |
| `src/data/convert_parquet.py` | Converts manually-downloaded HuggingFace Parquet files to PyTorch `ImageFolder` directory structure. Auto-detects image and label columns. Handles both binary bytes and dict-format image data. |

---

### 🟢 Training Scripts (`scripts/training/`)

| File | What it does |
|---|---|
| `scripts/training/train_vision.py` | **Full EfficientNet-B4 training pipeline.** Reads `configs/vision_config.yaml`. Implements 3-phase gradual unfreezing (head → last 2 MBConv → full model). Uses AMP (FP16), CosineAnnealingWarmRestarts, label smoothing, early stopping. Logs every epoch to W&B. Saves best val_acc checkpoint to `models/vision/efficientnet_b4_best.pt`. |
| `scripts/training/evaluate_vision.py` | **Test-set evaluation.** Loads the best checkpoint, runs inference on the hold-out test split. Computes accuracy, F1 macro, F1 weighted, AUC (macro OvR), and per-class accuracy. Saves metrics to `eval/eval_results.json`. |
| `scripts/training/train_qlora.py` | **Qwen2.5-3B QLoRA fine-tuning.** Reads `configs/llm_config.yaml`. Loads model in 4-bit NF4, applies LoRA adapters (rank=16), trains with TRL `SFTTrainer` on the instruction dataset. Includes `JSONValidityCallback` that logs JSON parse success rate to W&B every 500 steps. Saves adapter to `models/llm/qwen2.5_3b_qlora_adapter/`. |
| `scripts/training/test_qlora.py` | **Quick adapter smoke test.** Loads base model + QLoRA adapter, runs inference on 10 standard agricultural prompts (e.g., Tomato Late Blight, Apple Scab, Potato Early Blight). Confirms adapter loads correctly and generates coherent JSON responses. Run after fine-tuning to sanity-check. |

---

### 🟡 Data Scripts (`scripts/data/`)

| File | What it does |
|---|---|
| `scripts/data/download_plantvillage.py` | CLI entry point for dataset download. Calls the same logic as `src/data/download_plantvillage.py`. Use: `python scripts/data/download_plantvillage.py --method hf` |
| `scripts/data/convert_parquet.py` | CLI entry point for Parquet conversion. Calls the same logic as `src/data/convert_parquet.py`. Use: `python scripts/data/convert_parquet.py --parquet_dir downloads/` |

---

### 🔴 MLOps / Ops Scripts (`scripts/ops/`)

| File | What it does |
|---|---|
| `scripts/ops/upload_models.py` | Uploads both models to HuggingFace Hub. Creates repos automatically if not present. Uploads EfficientNet-B4 `.pt` + `class_names.json` to `spidey1807/crop-disease-efficientnet-b4`, and QLoRA adapter folder to `spidey1807/crop-disease-qwen2.5-qlora`. |
| `scripts/ops/register_model.py` | **MLOps gating pipeline.** First checks a promotion accuracy gate (default ≥ 85% — set `--min_accuracy 0.98` for stricter gate). If passed: (1) logs model as W&B artifact with all eval metadata (accuracy, F1, AUC, epoch), tagged as `staging` or `production`; (2) uploads to HuggingFace Hub with eval metrics in commit message. |

---

### 🧪 Test Suite (`tests/`)

The test suite is designed to run **without a GPU and without downloading any model checkpoints**. The `conftest.py` fixtures create a synthetic leaf image and an EfficientNet-B4 with random weights — tests verify architecture and logic, not accuracy.

#### `tests/conftest.py` — Shared Fixtures
Provides session-scoped pytest fixtures reused across all test files:
- `sample_leaf_image` — synthetic 224×224 green PIL image (simulates a leaf scan)
- `sample_tensor` — normalized `(1, 3, 224, 224)` torch tensor from the above image
- `class_names` — loads real `data/processed/class_names.json` or falls back to dummy names
- `vision_model` — `EfficientNetB4Classifier` with random weights, set to eval mode

#### `tests/test_vision_model.py` — EfficientNet-B4 Architecture Tests
8 tests verifying the model's output contract (no checkpoint needed):

| Test | Checks |
|---|---|
| `test_forward_pass_shape` | Output logits are exactly `(1, 38)` |
| `test_confidence_sum` | Softmax probabilities sum to exactly 1.0 |
| `test_output_dtype` | Logits are `float32` |
| `test_forward_features_shape` | Spatial feature map is 4-D `(1, C, h, w)` |
| `test_get_target_layer` | Hook target returns a valid `nn.Module` |
| `test_gradcam_output_shape` | GradCAM heatmap is 2-D `float32` |
| `test_gradcam_range` | GradCAM values are in `[0.0, 1.0]` |
| `test_no_nan_in_output` | No NaN or Inf values in logits |

#### `tests/test_preprocessing.py` — Image Transform Tests
6 tests verifying the preprocessing pipeline:

| Test | Checks |
|---|---|
| `test_output_shape` | `VAL_TRANSFORM` produces `(3, 224, 224)` tensor |
| `test_normalization_range` | Normalized values stay in `[-4, 4]` range |
| `test_output_dtype` | Output tensor is `float32` |
| `test_augmentation_determinism` | Same seed → identical augmented output |
| `test_batch_consistency` | Different pixel inputs → different tensor outputs |
| `test_channels_first` | Final tensor is `(C, H, W)` not `(H, W, C)` |

#### `tests/test_llm_advisor.py` — LLM Advisor Logic Tests
5 tests verifying schema validation, retry logic, and fallback (model mocked with `unittest.mock`):

| Test | Checks |
|---|---|
| `test_output_schema` | Treatment plan dict has all 10 required keys |
| `test_validate_output_passes` | `validate_output()` returns `True` for complete plan |
| `test_validate_output_fails_on_missing_key` | Returns `False` when `action_urgency` is missing |
| `test_json_retry_on_parse_error` | When LLM returns bad JSON: retries 3 times, falls back to DISEASE_DB |
| `test_urgency_is_valid` | `action_urgency` is one of the 4 valid urgency levels |

#### `tests/test_api.py` — FastAPI Endpoint Tests
7 tests using `TestClient` with the model fully mocked (no actual inference):

| Test | Checks |
|---|---|
| `test_health_endpoint` | `GET /health` returns 200 with `status: healthy` |
| `test_classes_endpoint` | `GET /classes` returns a list |
| `test_predict_valid_png` | `POST /predict` with PNG returns disease, confidence, top5 |
| `test_predict_valid_jpeg` | `POST /predict` with JPEG returns 200 |
| `test_predict_invalid_text_file` | `.txt` file returns 422 Unprocessable Entity |
| `test_predict_invalid_pdf` | PDF file returns 422 |
| `test_confidence_is_percentage` | Confidence float is in `[0, 100]` |

#### Run all tests:
```bash
pip install pytest httpx
pytest tests/ -v
```

---

## ⚙️ Configuration

### `configs/vision_config.yaml`
All EfficientNet-B4 training hyperparameters in one place:

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

> ⚠️ **Note:** `configs/llm_config.yaml` is not committed (contains model paths and API keys). Create it from the parameters documented in `scripts/training/train_qlora.py` before running LLM training.

---

## 📥 Complete MLOps Workflow

```
Step 1  Download dataset
        python src/data/download_plantvillage.py --method hf

Step 2  Train vision model
        python scripts/training/train_vision.py --config configs/vision_config.yaml

Step 3  Evaluate vision model
        python scripts/training/evaluate_vision.py --config configs/vision_config.yaml

Step 4  Generate LLM instruction dataset
        python src/llm/generate_dataset.py

Step 5  Fine-tune LLM (needs GPU)
        python scripts/training/train_qlora.py --config configs/llm_config.yaml

Step 6  Test LLM adapter
        python scripts/training/test_qlora.py

Step 7  Evaluate LLM
        python src/llm/evaluate_llm.py

Step 8  Register model with accuracy gate
        python scripts/ops/register_model.py \
          --checkpoint models/vision/efficientnet_b4_best.pt \
          --eval_path eval/eval_results.json \
          --stage production --min_accuracy 0.98

Step 9  Upload to HuggingFace Hub
        python scripts/ops/upload_models.py --username spidey1807

Step 10 Deploy
        git push origin main   # triggers Render auto-deploy
```

---

## ⚙️ Environment Variables

| Variable | Value | Description |
|---|---|---|
| `PHASE` | `1` (default) | `1` = CPU + DISEASE_DB; `2` = LLM inference |
| `PORT` | auto | Injected by Render/Cloud Run |
| `HF_TOKEN` | your token | HuggingFace API token |
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

## 👥 Authors

**Shubham Haraniya** · **Vidhan Savaliya**

---

## 📄 License

MIT License
