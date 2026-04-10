import sys
import os

from src.llm.advisor import CropDiseaseAdvisor

base_model = "Qwen/Qwen2.5-3B-Instruct"
adapter_path = "models/llm/qwen2.5_3b_qlora_adapter"

print("Initializing Advisor...")
# Reduce VRAM usage for the test by disabling auto
advisor = CropDiseaseAdvisor(base_model, adapter_path, device="cuda")

disease = "Tomato___Spider_mites"
crop = "Tomato"
region = "North India"
season = "Kharif"
severity = "Severe (60–100%)"

print("\nGenerating...")
plan = advisor.generate_treatment_plan(
    disease=disease,
    crop=crop,
    region=region,
    season=season,
    severity=severity,
    max_new_tokens=500
)

print("\nPLAN RESULT:")
import json
print(json.dumps(plan, indent=2))
