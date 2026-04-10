---
title: Crop Disease Advisor
emoji: 🌾
colorFrom: green
colorTo: emerald
sdk: docker
pinned: false
license: mit
short_description: AI-powered plant disease diagnosis & treatment planning
---

# 🌾 Crop Disease Advisor

**AI-powered plant disease detection with region & season-aware treatment plans.**

- 🖼 Upload a leaf photo → EfficientNet-B4 diagnoses the disease
- 📋 Get organic, chemical & preventive treatment recommendations
- 📄 Download a professional PDF diagnostic report
- 🇮🇳 Tailored for Indian farmers across 5 regions & 3 seasons

> **Note:** Running in CPU mode on HuggingFace free tier.
> Treatment plans are generated from a curated agricultural knowledge base (DISEASE_DB).
> Full LLM (Qwen2.5-3B) mode is available when running locally with GPU.
