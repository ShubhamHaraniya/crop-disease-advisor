"""
Gradio Phase 1 UI — Vision-only crop disease detector.
Prompt 1.8 — Phase 1 Gradio Frontend
"""

import io
import base64
import os
import requests
from pathlib import Path
from PIL import Image

import gradio as gr


API_URL = os.getenv("API_URL", "http://localhost:8000")


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(image: Image.Image):
    if image is None:
        return None, None, "⚠ Please upload an image."

    # Convert PIL image to bytes for the API
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    try:
        resp = requests.post(
            f"{API_URL}/predict",
            files={"file": ("image.png", buf, "image/png")},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return None, None, "❌ API not reachable. Make sure the server is running."
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

    # Top-5 predictions for gr.Label
    label_dict = {p["label"].replace("___", " — ").replace("_", " "): p["confidence"] / 100
                  for p in data["top5"]}

    # Decode heatmap
    heatmap_bytes = base64.b64decode(data["heatmap_base64"])
    heatmap_img   = Image.open(io.BytesIO(heatmap_bytes)).convert("RGB")

    # Confidence info text
    confidence = data["confidence"]
    disease    = data["disease"].replace("___", " — ").replace("_", " ")
    crop       = data["crop"]
    warning    = "\n\n⚠ **Low confidence** — please upload a clearer image." if confidence < 60 else ""

    info_md = (
        f"**🌾 Crop:** {crop}  \n"
        f"**🦠 Disease:** {disease}  \n"
        f"**📊 Confidence:** {confidence:.1f}%"
        f"{warning}"
    )

    return label_dict, heatmap_img, info_md


# ── Examples ──────────────────────────────────────────────────────────────────
# Using placeholder paths — replace with actual example images
EXAMPLE_IMAGES = [
    ["notebooks/sample_augmentations/sample_aug_00.png"],
    ["notebooks/sample_augmentations/sample_aug_01.png"],
    ["notebooks/sample_augmentations/sample_aug_02.png"],
    ["notebooks/sample_augmentations/sample_aug_03.png"],
    ["notebooks/sample_augmentations/sample_aug_04.png"],
]
# Filter to only existing files
EXAMPLE_IMAGES = [e for e in EXAMPLE_IMAGES if Path(e[0]).exists()]


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
.confidence-high { color: #22c55e; font-weight: bold; }
.confidence-low  { color: #ef4444; font-weight: bold; }
.header-title    { text-align: center; font-size: 2rem; }
"""

with gr.Blocks(css=CSS, title="🌿 Crop Disease Detector") as demo:

    gr.Markdown(
        """
        # 🌿 Crop Disease Detector (Vision AI)
        **Upload a leaf photo to identify plant diseases using EfficientNet-B4 + Grad-CAM**
        """,
        elem_classes=["header-title"],
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="📷 Upload Leaf Photo",
                sources=["upload", "webcam"],
                height=320,
            )
            submit_btn  = gr.Button("🔍 Analyze Leaf", variant="primary", size="lg")

        with gr.Column():
            label_out   = gr.Label(
                num_top_classes=5,
                label="🦠 Top 5 Disease Predictions",
            )
            heatmap_out = gr.Image(
                label="🔥 Affected Region Heatmap (Grad-CAM)",
                height=320,
            )
            info_out    = gr.Markdown(label="📋 Diagnosis Summary")

    if EXAMPLE_IMAGES:
        gr.Examples(
            examples=EXAMPLE_IMAGES,
            inputs=[input_image],
            label="📂 Example Leaf Images",
        )

    gr.Markdown(
        "---\n*Powered by EfficientNet-B4 trained on PlantVillage (38 classes, 87K images)*"
    )

    submit_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[label_out, heatmap_out, info_out],
        show_progress="full",
        api_name="predict",
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
