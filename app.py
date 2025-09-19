import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import gradio as gr
import numpy as np
import cv2
import os

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load trained SegFormer model from Hugging Face Hub ---
NUM_CLASSES = 5
MODEL_REPO = "your-username/segformer-flood-damage"  # ⚠️ REPLACE THIS WITH YOUR ACTUAL HF REPO

model = None
feature_extractor = None

print("📥 Attempting to load model and feature extractor from Hugging Face Hub...")
try:
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_REPO)
    feature_extractor = SegformerImageProcessor.from_pretrained(MODEL_REPO)
    model.to(device)
    model.eval()
    print("✅ Model and processor loaded successfully.")
except Exception as e:
    print(f"❌ Model or processor failed to load: {e}")
    # DO NOT RAISE OR EXIT — let app start and show error in UI

# --- Define colors for classes ---
COLORS = np.array([
    [0, 0, 0],       # class 0: Background
    [0, 255, 0],     # class 1: No Damage
    [255, 255, 0],   # class 2: Minor Damage
    [255, 165, 0],   # class 3: Major Damage
    [255, 0, 0],     # class 4: Destroyed
], dtype=np.uint8)

# --- Prediction and Overlay function ---
def segment_and_overlay(image: Image.Image, opacity: float = 0.5):
    # Check if model loaded
    if model is None or feature_extractor is None:
        return None, "⚠️ Model failed to load. Check server logs or contact administrator.", None

    if image is None:
        return None, "Please upload an image.", None

    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred_mask = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()

        orig_w, orig_h = image.size
        mask_resized = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        damaged_mask = np.zeros_like(mask_resized, dtype=bool)
        for damage_class_id in [2, 3, 4]:
            damaged_mask = np.logical_or(damaged_mask, (mask_resized == damage_class_id))

        total_pixels = mask_resized.size
        damaged_pixels = np.sum(damaged_mask)

        if total_pixels > 0:
            damaged_percent = (damaged_pixels / total_pixels) * 100
            damaged_text = f"Damaged Area (Minor, Major, Destroyed): {damaged_percent:.2f}%"
        else:
            damaged_text = "Image has no pixels."

        color_mask = COLORS[mask_resized]
        orig_np = np.array(image).astype(np.uint8)

        if color_mask.shape[:2] != orig_np.shape[:2]:
            color_mask = cv2.resize(color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        if orig_np.shape != color_mask.shape:
            return Image.fromarray(orig_np), "Error: Shape mismatch in overlay generation.", None

        overlay = cv2.addWeighted(orig_np, 1 - opacity, color_mask, opacity, 0)
        overlay_img_pil = Image.fromarray(overlay)
        output_filename = "segmentation_overlay.png"
        overlay_img_pil.save(output_filename)

        return overlay_img_pil, damaged_text, output_filename

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, f"⚠️ Error during prediction: {str(e)}", None

# --- Gradio interface ---
demo = gr.Interface(
    fn=segment_and_overlay,
    inputs=[
        gr.Image(type="pil", label="Upload Satellite/Drone Image"),
        gr.Slider(0.0, 1.0, value=0.5, label="Overlay Opacity")
    ],
    outputs=[
        gr.Image(type="pil", label="Segmentation Overlay"),
        gr.Textbox(label="Damaged Area Percentage"),
        gr.File(label="Download Overlay")
    ],
    title="🌊 Flood Damage Segmentation and Area Calculation",
    description="""
    Upload a satellite or drone image. The model predicts flood damage levels and calculates % damaged area.
    Model: `segformer-flood-damage` fine-tuned on xBD dataset.
    """
)

# --- Launch ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Gradio server on 0.0.0.0:{port}...")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
