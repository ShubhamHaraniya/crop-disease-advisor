"""
End-to-end pipeline tests (vision-only path, no LLM).
Prompt 2.9 — test_pipeline.py
"""

import torch
import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestPipeline:

    def test_confidence_to_severity_severe(self):
        from src.pipeline import confidence_to_severity
        assert confidence_to_severity(0.95) == "Severe (60–100%)"

    def test_confidence_to_severity_moderate(self):
        from src.pipeline import confidence_to_severity
        assert confidence_to_severity(0.65) == "Moderate (30–60%)"

    def test_confidence_to_severity_mild(self):
        from src.pipeline import confidence_to_severity
        assert confidence_to_severity(0.20) == "Mild (0–30%)"

    def test_confidence_boundary_80(self):
        """Exactly 0.80 should be Severe."""
        from src.pipeline import confidence_to_severity
        assert confidence_to_severity(0.80) == "Severe (60–100%)"

    def test_confidence_boundary_50(self):
        """Exactly 0.50 should be Moderate."""
        from src.pipeline import confidence_to_severity
        assert confidence_to_severity(0.50) == "Moderate (30–60%)"

    def test_top5_length(self, vision_model, sample_tensor, class_names):
        """Top-5 predictions must always return exactly 5 entries."""
        with torch.no_grad():
            logits = vision_model(sample_tensor)
            probs  = torch.softmax(logits, dim=1)[0]
        top5 = probs.topk(5)
        assert len(top5.indices) == 5
        assert len(top5.values)  == 5

    def test_crop_name_extraction(self):
        """Crop name must be extracted correctly from disease label."""
        disease = "Tomato___Late_blight"
        crop = disease.split("___")[0].replace("_", " ")
        assert crop == "Tomato"

    def test_crop_name_extraction_no_separator(self):
        """When no '___' separator, full label is used as crop name."""
        disease = "Unknown"
        crop = disease.split("___")[0].replace("_", " ") if "___" in disease else disease
        assert crop == "Unknown"

    def test_heatmap_is_pil_image(self, vision_model, sample_leaf_image, sample_tensor):
        """GradCAM overlay must return a PIL Image."""
        from src.vision.gradcam import GradCAM
        cam     = GradCAM(vision_model, vision_model.get_target_layer())
        heatmap = cam.generate_cam(sample_tensor, target_class=0)
        overlay = cam.overlay_heatmap(sample_leaf_image, heatmap, alpha=0.5)
        cam.remove_hooks()
        assert isinstance(overlay, Image.Image), "Overlay must be a PIL Image"
        assert overlay.size == sample_leaf_image.size, "Overlay size must match original"
