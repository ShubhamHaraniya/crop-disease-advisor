---
title: Crop Disease Advisor
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.12.0
app_file: app.py
pinned: false
hardware: t4-medium
license: mit
tags:
  - computer-vision
  - agriculture
  - plant-disease
  - efficientnet
  - llama
  - qlora
  - gradcam
  - mlops
---

# 🌾 Crop Disease Advisor

**AI-powered plant disease diagnosis + treatment recommendations for Indian farmers**

- **Shubham Haraniya**
- **Vidhan Savaliya**

---

## 🔬 How It Works

This system uses a **two-stage AI pipeline**:

1. **Stage 1 — Vision Diagnosis** (EfficientNet-B4):
   - Classifies leaf images into 38 disease categories across 14 crop types
   - Generates Grad-CAM heatmaps highlighting the affected leaf regions
   - Trained on PlantVillage dataset (87,900 images)

2. **Stage 2 — Treatment Advisory** (LLaMA 3 8B + QLoRA):
   - Fine-tuned on 12,000 agronomic instruction pairs
   - Generates region-aware, season-specific treatment plans
   - Returns structured JSON with organic, chemical, and preventive options

---

## 🌿 Supported Crops & Diseases

| Crop       | Diseases Covered |
|------------|-----------------|
| Tomato     | Late blight, Early blight, Bacterial spot, Leaf Mold, +5 more |
| Apple      | Apple scab, Black rot, Cedar rust |
| Corn       | Common rust, Northern Leaf Blight, Cercospora |
| Grape      | Black rot, Esca, Leaf blight |
| Potato     | Early blight, Late blight |
| Pepper     | Bacterial spot |
| Strawberry | Leaf scorch |
| Peach      | Bacterial spot |
| + more...  | |

---

## 📊 Model Performance

| Model | Dataset | Accuracy |
|-------|---------|----------|
| EfficientNet-B4 | PlantVillage (test) | **≥ 95%** |
| LLaMA 3 QLoRA | Instruction eval | JSON validity ≥ 90% |

---

## 🛠 Tech Stack

`PyTorch` · `timm` · `EfficientNet-B4` · `Grad-CAM` · `LLaMA 3 8B` · `QLoRA` · `PEFT` · `FastAPI` · `Gradio` · `Weights & Biases` · `HuggingFace Hub`
