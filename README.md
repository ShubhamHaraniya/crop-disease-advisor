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

**Crop Disease Advisor** is an end-to-end MLOps project that combines computer vision and large language models to help Indian farmers identify plant diseases from leaf photographs and receive detailed, contextual treatment plans — all in a single, production-deployed web application.

Upload a photo of a diseased leaf → get an **instant diagnosis** across 38 disease classes → receive a **structured treatment plan** with organic treatments, chemical treatments, preventive measures, yield impact, and region/season-aware advisory.

---

## 🌐 Live Links

| Resource | URL |
|---|---|
| 🚀 **Production App (Render)** | https://crop-disease-advisor.onrender.com/ |
| 🤗 **HuggingFace Space** | https://huggingface.co/spaces/spidey1807/crop-disease-advisor |
| 🧠 **EfficientNet-B4 Model** | https://huggingface.co/spidey1807/crop-disease-efficientnet-b4 |
| 🦙 **Qwen2.5-3B QLoRA Model** | https://huggingface.co/spidey1807/crop-disease-qwen2.5-qlora |
| 📊 **W&B Vision Training** | https://wandb.ai/spharaniya18-intelkit-solutions/crop-disease-advisor |
| 📊 **W&B LLM Training** | https://wandb.ai/spharaniya18-intelkit-solutions/crop-disease-advisor-llm |

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🖼️ **Drag-and-drop image upload** | Instant preview with file validation |
| 🦠 **38 disease classes, 14 crops** | EfficientNet-B4 fine-tuned on PlantVillage |
| 🎚️ **Dual severity mode** | Auto-detection or manual slider (Mild / Moderate / Severe) |
| 💊 **Tabbed treatment plans** | Organic · Chemical · Preventive · Advisory |
| 🌍 **Region & Season aware** | Tailored for 5 Indian regions × 3 crop seasons (Kharif/Rabi/Zaid) |
| 📄 **Professional PDF report** | One-click clinical report download |
| ⚡ **CPU-only deployment** | No GPU needed — works on Render free tier |

---

## 🏗️ System Architecture

```
                        ┌─────────────────────────────┐
                        │       User (Browser)         │
                        │    Vite + Vanilla JS UI      │
                        └────────────┬────────────────┘
                                     │ REST API
                        ┌────────────▼────────────────┐
                        │     FastAPI Backend          │
                        │       (app.py)               │
                        └──────┬─────────────┬────────┘
                               │             │
               ┌───────────────▼──┐   ┌─────▼──────────────────┐
               │  Vision Pipeline  │   │   Treatment Pipeline    │
               │                   │   │                         │
               │  EfficientNet-B4  │   │   DISEASE_DB Fallback   │
               │  (38 classes)     │   │   (curated knowledge)   │
               │  timm + PyTorch   │   │   ── OR ──              │
               └───────────────────┘   │   Qwen2.5-3B + QLoRA   │
                                       │   (fine-tuned LLM)      │
                                       └─────────────────────────┘
```

**Phase 1 (Production/CPU):** Vision model → DISEASE\_DB curated treatment plans
**Phase 2 (GPU/Chatbot):** Vision model → Qwen2.5-3B QLoRA fine-tuned LLM advisory

---

## 📂 Repository Structure

```
crop-disease-advisor/
│
├── app.py                          # FastAPI entrypoint (reads PORT env)
├── app/
│   └── chatbot.py                  # Streamlit LLM chatbot (presentation)
│
├── src/
│   ├── api/main_phase2.py          # FastAPI routes & model loading
│   ├── vision/
│   │   ├── model.py                # EfficientNet-B4 classifier
│   │   └── preprocess.py           # Image transforms
│   └── llm/
│       ├── advisor.py              # LLM inference + DISEASE_DB fallback
│       └── generate_dataset.py     # Curated treatment knowledge base
│
├── scripts/
│   ├── training/
│   │   ├── train_vision.py         # Vision model training (W&B, gradual unfreezing)
│   │   ├── evaluate_vision.py      # Vision model evaluation
│   │   ├── train_qlora.py          # QLoRA fine-tuning for Qwen2.5-3B
│   │   └── test_qlora.py           # LLM inference test
│   ├── data/
│   │   └── download_plantvillage.py
│   └── ops/
│       └── upload_models.py
│
├── configs/
│   ├── vision_config.yaml          # Training hyperparameters
│   └── llm_config.yaml             # QLoRA config
│
├── eval/
│   ├── eval_results.json           # Vision model test metrics
│   └── llm_eval_run.log            # LLM evaluation results
│
├── models/
│   └── vision/efficientnet_b4_best.pt  # Trained model weights (~68 MB)
│
├── data/processed/class_names.json # 38 disease class labels
├── frontend/                       # Vite UI (HTML + JS + CSS)
│
├── requirements.txt                # Deployment (CPU-only)
├── requirements_training.txt       # Training + chatbot (GPU)
├── Dockerfile                      # Multi-platform Docker build
└── .dockerignore
```

---

## 📊 Model Evaluation Results

### 🔬 Vision Model — EfficientNet-B4

> Evaluated on **5,431 held-out test samples** across **38 disease classes**

| Metric | Score |
|---|---|
| **Accuracy** | **99.69%** |
| **F1 Score (Macro)** | **99.58%** |
| **F1 Score (Weighted)** | **99.69%** |
| **AUC (Macro OvR)** | **100.00%** |
| Test Samples | 5,431 |
| Classes | 38 |

**Per-class highlights** (lowest performers — all others are 100%):

| Class | Accuracy |
|---|---|
| Corn — Cercospora (Gray leaf spot) | 96.08% |
| Tomato — Yellow Leaf Curl Virus | 96.43% |
| Corn — healthy | 97.98% |
| Corn — Common rust | 98.32% |
| Tomato — Target Spot | 98.81% |

---

### 🤖 LLM Advisor — Qwen2.5-3B + QLoRA

> Evaluated on 10 agricultural advisory prompts across diverse disease/crop/region combinations

#### Eval 1: JSON Structured Output Quality

| Metric | Score |
|---|---|
| **JSON Validity Rate** | **100%** |
| **Schema Compliance** | **100%** |
| **Field Completeness** | **95%** |
| **Perplexity** (avg, lower = better) | **1.42** |
| Avg Inference Latency | 30.0s (GPU) |

#### Eval 2: Advisory Text Quality

| Metric | Score |
|---|---|
| **BERTScore F1** | **81.68%** |
| Semantic Similarity | 44.21% |
| ROUGE-L | 9.23% |
| BLEU-4 | 0.89% |

> **Note:** Low BLEU/ROUGE scores are expected — the model generates contextual, structured advisory narratives that don't match reference sentences word-for-word. The high BERTScore (81.68%) reflects strong semantic alignment with reference answers.

---

## 🏋️ Training Details

### Vision Model

| Setting | Value |
|---|---|
| Architecture | EfficientNet-B4 (timm) |
| Dataset | PlantVillage (38 classes, 14 crops) |
| Strategy | Gradual unfreezing (3 phases) |
| Optimizer | AdamW + CosineAnnealingWarmRestarts |
| Mixed Precision | FP16 (AMP) |
| Label Smoothing | 0.1 |
| Experiment Tracking | Weights & Biases |

📊 **Training Dashboard:** [W&B Vision](https://wandb.ai/spharaniya18-intelkit-solutions/crop-disease-advisor/workspace?nw=nwuserspharaniya18)

### LLM Fine-tuning

| Setting | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-3B-Instruct |
| Method | QLoRA (4-bit NF4 quantization) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Trainer | TRL SFTTrainer |
| Task | Instruction-following (structured JSON advisory) |
| Dataset | Curated crop disease instruction dataset |
| Experiment Tracking | Weights & Biases |

📊 **Training Dashboard:** [W&B LLM](https://wandb.ai/spharaniya18-intelkit-solutions/crop-disease-advisor-llm/workspace?nw=nwuserspharaniya18)

---

## 🚀 Quick Start

### Run the API Locally

```bash
# Clone the repo
git clone https://github.com/ShubhamHaraniya/crop-disease-advisor.git
cd crop-disease-advisor

# Install CPU dependencies
pip install -r requirements.txt

# Set environment
set PHASE=1   # Windows
# export PHASE=1  # Linux/Mac

# Start API
python app.py
# → http://localhost:8000
```

### Run the Frontend (dev)

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Run with Docker

```bash
docker build -t crop-disease-advisor .
docker run -p 8080:8080 -e PHASE=1 crop-disease-advisor
```

---

## 🤖 Run the LLM Chatbot (GPU required)

```bash
# Install GPU training dependencies
pip install -r requirements_training.txt

# Run chatbot
streamlit run app/chatbot.py
```

---

## 🔁 Retrain the Models

### Retrain Vision Model

```bash
pip install -r requirements_training.txt
python scripts/training/train_vision.py --config configs/vision_config.yaml
```

### Retrain LLM (QLoRA)

```bash
# Generate instruction dataset first
python src/llm/generate_dataset.py

# Fine-tune
python scripts/training/train_qlora.py --config configs/llm_config.yaml
```

---

## 🌿 Supported Crops & Diseases

| Crop | Diseases |
|---|---|
| 🍎 Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🫐 Blueberry | Healthy |
| 🍒 Cherry | Powdery Mildew, Healthy |
| 🌽 Corn (Maize) | Cercospora/Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍇 Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| 🍊 Orange | Haunglongbing (Citrus Greening) |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🫑 Pepper (Bell) | Bacterial Spot, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🍓 Raspberry | Healthy |
| 🫘 Soybean | Healthy |
| 🎃 Squash | Powdery Mildew |
| 🍓 Strawberry | Leaf Scorch, Healthy |
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🌍 Region & Season Support

**Indian Regions:** North India · South India · West India · East India · North-East India

**Crop Seasons:** Kharif (Monsoon, Jun–Oct) · Rabi (Winter, Nov–Mar) · Zaid (Summer, Apr–Jun)

---

## ⚙️ Environment Variables

| Variable | Values | Description |
|---|---|---|
| `PHASE` | `1` (default) | `1` = CPU mode, DISEASE_DB fallback. `2` = GPU + LLM inference |
| `PORT` | auto | Injected by Render/Cloud Run at runtime |
| `HF_TOKEN` | your token | Required only for downloading gated HF models |

---

## 👥 Authors

**Shubham Haraniya** · **Vidhan Savaliya**

---

## 📄 License

This project is licensed under the **MIT License**.
