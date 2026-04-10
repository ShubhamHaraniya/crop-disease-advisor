"""
Unified two-stage pipeline: EfficientNet-B4 (Vision) + LLaMA 3 QLoRA (LLM).
Prompt 2.4 — Unified Two-Stage Pipeline
"""

import time
import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.vision.model   import EfficientNetB4Classifier
from src.vision.gradcam import GradCAM
from src.vision.preprocess import VAL_TRANSFORM
from src.llm.advisor    import CropDiseaseAdvisor


# ── Severity Mapping ──────────────────────────────────────────────────────────

def confidence_to_severity(confidence: float) -> str:
    """Map model confidence [0, 1] to severity label."""
    if confidence >= 0.80:
        return "Severe (60–100%)"
    elif confidence >= 0.50:
        return "Moderate (30–60%)"
    else:
        return "Mild (0–30%)"


# ── Pipeline ──────────────────────────────────────────────────────────────────

class CropDiseasePipeline:
    """
    End-to-end pipeline combining vision diagnosis and LLM treatment planning.

    Usage:
        pipeline = CropDiseasePipeline(
            vision_model_path="models/vision/efficientnet_b4_best.pt",
            llm_adapter_path="models/llm/llama3_qlora_adapter",
        )
        result = pipeline.predict(image, region="North India", season="Kharif (Monsoon)")
    """

    def __init__(
        self,
        vision_model_path: str,
        llm_adapter_path:  str,
        class_names_path:  str = "data/processed/class_names.json",
        base_llm_model:    str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device:            str = "cuda",
        num_classes:       int = 38,
    ):
        self.device = device

        # ── Load class names ─────────────────────────────────────────────────
        with open(class_names_path) as f:
            mapping          = json.load(f)
            self.class_names = [mapping[str(i)] for i in range(len(mapping))]

        # ── Load vision model ─────────────────────────────────────────────────
        print("[Pipeline] Loading EfficientNet-B4...")
        self.vision_model = EfficientNetB4Classifier(num_classes=num_classes).to(device)
        ckpt = torch.load(vision_model_path, map_location=device)
        self.vision_model.load_state_dict(ckpt["state_dict"])
        self.vision_model.eval()
        self.cam = GradCAM(self.vision_model, self.vision_model.get_target_layer())

        # ── Load LLM advisor ──────────────────────────────────────────────────
        print("[Pipeline] Loading LLaMA 3 + LoRA adapter...")
        self.advisor = CropDiseaseAdvisor(base_llm_model, llm_adapter_path, device=device)

        print("[Pipeline] Ready ✓")

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, image: Image.Image, region: str, season: str) -> dict:
        """
        Full two-stage prediction.

        Args:
            image  : PIL Image (leaf photo)
            region : e.g. "North India"
            season : e.g. "Kharif (Monsoon)"

        Returns:
            Unified result dict.
        """
        t_start = time.perf_counter()

        # ── Stage 1: Vision ──────────────────────────────────────────────────
        t_vis = time.perf_counter()
        tensor = VAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.vision_model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        top5_idx  = probs.topk(5).indices.tolist()
        top5_conf = probs.topk(5).values.tolist()
        pred_idx  = top5_idx[0]
        confidence = top5_conf[0]
        disease    = self.class_names[pred_idx]
        crop       = disease.split("___")[0].replace("_", " ") if "___" in disease else disease
        severity   = confidence_to_severity(confidence)

        # Grad-CAM
        cam_array = self.cam.generate_cam(tensor, target_class=pred_idx)
        heatmap   = self.cam.overlay_heatmap(image, cam_array, alpha=0.5)

        top5_predictions = [
            {"label": self.class_names[i], "confidence": round(c * 100, 2)}
            for i, c in zip(top5_idx, top5_conf)
        ]
        t_vis_ms = (time.perf_counter() - t_vis) * 1000

        # ── Stage 2: LLM Advisory ─────────────────────────────────────────────
        t_llm = time.perf_counter()
        treatment_plan = self.advisor.generate_treatment_plan(
            disease=disease.replace("___", " — ").replace("_", " "),
            crop=crop,
            region=region,
            season=season,
            severity=severity,
        )
        t_llm_ms = (time.perf_counter() - t_llm) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000

        return {
            "disease":           disease,
            "crop":              crop,
            "confidence":        round(confidence * 100, 2),
            "severity":          severity,
            "top5_predictions":  top5_predictions,
            "heatmap":           heatmap,
            "treatment_plan":    treatment_plan,
            "inference_time_ms": {
                "vision_ms": round(t_vis_ms, 2),
                "llm_ms":    round(t_llm_ms, 2),
                "total_ms":  round(total_ms, 2),
            },
        }

    # ── Dry Run ───────────────────────────────────────────────────────────────

    def dry_run(self, image: Optional[Image.Image] = None):
        """Test the full pipeline with a synthetic green image and print timing."""
        if image is None:
            image = Image.new("RGB", (224, 224), color=(34, 139, 34))  # forest green

        print("\n[DryRun] Running full pipeline on synthetic image...")
        result = self.predict(image, region="North India", season="Kharif (Monsoon)")

        print(f"\n── Dry Run Results ─────────────────────────────────────")
        print(f"  Disease    : {result['disease']}")
        print(f"  Crop       : {result['crop']}")
        print(f"  Confidence : {result['confidence']:.1f}%")
        print(f"  Severity   : {result['severity']}")
        print(f"\n  Timing:")
        for k, v in result["inference_time_ms"].items():
            print(f"    {k:<12} : {v:.2f} ms")
        print(f"\n  Treatment plan keys: {list(result['treatment_plan'].keys())}")
        print("✓ Dry run complete!")
        return result


if __name__ == "__main__":
    pipeline = CropDiseasePipeline(
        vision_model_path="models/vision/efficientnet_b4_best.pt",
        llm_adapter_path="models/llm/llama3_qlora_adapter",
    )
    pipeline.dry_run()
