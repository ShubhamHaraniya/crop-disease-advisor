---
title: Crop Disease Advisor
emoji: 🌾
colorFrom: green
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: MLOps-ready plant disease diagnosis & treatment planning
---

# 🌾 Crop Disease Advisor — MLOps Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" />
</p>

An AI-powered agricultural diagnostic tool designed for Indian farmers. This project demonstrates a complete **End-to-End MLOps lifecycle**—from robust dual-model training (Vision + LLM) and experiment tracking to interactive UI development and containerized cloud deployment.

## 🚀 Live Deployments
* 🌐 **Production Web App (Render):** [crop-disease-advisor.onrender.com](https://crop-disease-advisor.onrender.com/)
* 🤗 **HuggingFace Space (Demo):** [hf.co/spaces/spidey1807/crop-disease-advisor](https://huggingface.co/spaces/spidey1807/crop-disease-advisor)

## 📊 MLOps & Model Artifacts
All experiments and trained model weights are publicly accessible and strictly tracked:

| Component | HuggingFace Model Weights | Weights & Biases Logging |
|---|---|---|
| **Vision (EfficientNet-B4)** | [HF Repo: crop-disease-efficientnet-b4](https://huggingface.co/spidey1807/crop-disease-efficientnet-b4) | [W&B: Vision Training Runs](https://wandb.ai/spharaniya18-intelkit-solutions/crop-disease-advisor/workspace?nw=nwuserspharaniya18) |
| **LLM (Qwen 2.5 3B QLoRA)** | [HF Repo: crop-disease-qwen2.5-qlora](https://huggingface.co/spidey1807/crop-disease-qwen2.5-qlora) | [W&B: LLM Fine-tuning Runs](https://wandb.ai/spharaniya18-intelkit-solutions/crop-disease-advisor-llm/workspace?nw=nwuserspharaniya18) |

## ✨ Core Features
- **🖼️ Vision Diagnostics:** Drag-and-drop capability for instant plant disease classification across 38 disease classes and 14 different crops.
- **💊 Comprehensive Treatment Plans:** Context-aware actionable advice separated into Organic, Chemical, Preventive, and Regional/Seasonal advisories.
- **📱 Responsive UI/UX:** Built with Vite and modern JS principles featuring glassmorphic designs and dark mode aesthetics.
- **📄 Clinical PDF Reports:** Auto-generates downloadable, professional disease diagnostic reports.
- **🧠 Hybrid Inference Engine:** Dynamic routing between LLM-generated treatment plans (GPU inference) and an optimized, highly-curated Knowledge Base fallback (CPU-only cloud production).

## 🏗️ Technical Architecture
- **Frontend:** Vanilla JS, HTML/CSS packaged dynamically with `Vite`.
- **Backend:** `FastAPI` bridging async image processing with model inference.
- **Vision Subsystem:** `EfficientNet-B4` trained via `PyTorch` with mixed-precision, progressive unfreezing, and extensive augmentations.
- **LLM/Chatbot Subsystem:** Instruction-tuned `Qwen 2.5 3B` using `QLoRA` (`peft`, `trl`, `transformers`) for agricultural advisory semantic generation.
- **Infrastructure:** Fully containerized via `Docker` ensuring portable footprint across Render and huggingFace architectures.

## 💻 Local Setup & Development

### 1. Production Mode (CPU-only / Web Deployment)
Optimized for memory-constrained cloud environments (e.g. Render Free Tier). Uses `requirements.txt`.
```bash
git clone https://github.com/ShubhamHaraniya/crop-disease-advisor.git
cd crop-disease-advisor

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server (PHASE=1 by default)
python app.py
```

### 2. Training / Local Presentation Mode (GPU Required)
Retains heavy DL pipelines, LLM inference packages, and training endpoints.
```bash
# Install heavy-weight AI dependencies
pip install -r requirements_training.txt

# Run the experimental Streamlit UI / Chatbot integration
streamlit run app/chatbot.py
```

## 👨‍💻 Authors
- **Shubham Haraniya**
- **Vidhan Savaliya**
