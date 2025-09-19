

import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from PIL import Image
import gradio as gr
import numpy as np
import cv2
import os

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load trained SegFormer model ---
# Define model configuration (must match training)
NUM_CLASSES = 5
MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
# Change this line in your script
CHECKPOINT_PATH = r"D:\Technova\models\segformer_epoch_10.pth" # Path to your best checkpoint

# Load base model
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_ID,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

# Load trained weights
try:
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Successfully loaded model weights from {CHECKPOINT_PATH}")
except FileNotFoundError:
    print(f"❌ Error: Model checkpoint not found at {CHECKPOINT_PATH}. Please ensure the path is correct.")
    # You might want to exit or handle this error more gracefully
    exit() # Exit if checkpoint not found


model.to(device)
model.eval()

# --- Feature extractor ---
# Using SegformerImageProcessor is recommended in newer versions of transformers
try:
    from transformers import SegformerImageProcessor
    feature_extractor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    print("✅ Using SegformerImageProcessor")
except ImportError:
    print("⚠️ SegformerImageProcessor not found, falling back to SegformerFeatureExtractor.")
    feature_extractor = SegformerFeatureExtractor(do_resize=True, size=512)


# --- Define colors for classes (adjust based on your dataset's class mapping) ---
# Example mapping: 0=background, 1=no-damage, 2=minor, 3=major, 4=destroyed
COLORS = np.array([
    [0, 0, 0],       # class 0: Background (e.g., not a building)
    [0, 255, 0],     # class 1: No Damage
    [255, 255, 0],   # class 2: Minor Damage
    [255, 165, 0],   # class 3: Major Damage
    [255, 0, 0],     # class 4: Destroyed
], dtype=np.uint8)


# --- Prediction and Overlay function ---
def segment_and_overlay(image: Image.Image, opacity: float = 0.5):
    """
    Takes a PIL image, predicts segmentation mask, creates a color overlay,
    and calculates the percentage of damaged area (classes 2, 3, and 4).
    """
    if image is None:
        return None, "Please upload an image.", None

    # --- Preprocess ---
    # Convert image to RGB if it's not (Gradio provides PIL images)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # The feature extractor handles resizing
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # --- Model prediction ---
    with torch.no_grad():
        outputs = model(**inputs)
        # Get predicted mask and move to CPU, convert to numpy
        pred_mask = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()

    # --- Resize mask to original image size ---
    # Use the original PIL image size
    orig_w, orig_h = image.size
    # cv2.resize expects (width, height)
    mask_resized = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # --- Calculate Damaged Area Percentage (Classes 2, 3, 4) ---
    # Assuming your classes 2, 3, and 4 represent different levels of damage
    damaged_mask = np.zeros_like(mask_resized, dtype=bool)
    for damage_class_id in [2, 3, 4]: # Adapt these class IDs if your mapping is different
        damaged_mask = np.logical_or(damaged_mask, (mask_resized == damage_class_id))

    total_pixels = mask_resized.size
    damaged_pixels = np.sum(damaged_mask)

    if total_pixels > 0:
         damaged_percent = (damaged_pixels / total_pixels) * 100
         damaged_text = f"Damaged Area (Minor, Major, Destroyed): {damaged_percent:.2f}%"
    else:
         damaged_text = "Image has no pixels." # Should not happen with valid image input


    # --- Create color mask for all classes ---
    color_mask = COLORS[mask_resized]


    # --- Overlay ---
    # Convert original PIL image to numpy array for OpenCV
    orig_np = np.array(image).astype(np.uint8)
    # Ensure color_mask is in the correct format (H, W, C)
    if color_mask.ndim == 2: # If it's grayscale, convert to BGR/RGB
         color_mask = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2RGB)
    elif color_mask.shape[-1] == 4: # If it has an alpha channel, remove it
         color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGBA2RGB)


    # Resize color_mask if necessary (should already be done with mask_resized)
    if color_mask.shape[:2] != orig_np.shape[:2]:
        color_mask = cv2.resize(color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


    # Check if shapes match before blending
    if orig_np.shape != color_mask.shape:
        print(f"Shape mismatch: Original image shape {orig_np.shape}, Color mask shape {color_mask.shape}")
        # Fallback or error handling if shapes still don't match
        return Image.fromarray(orig_np), "Error processing image overlay due to shape mismatch.", None


    # Blend the original image and the color mask
    overlay = cv2.addWeighted(orig_np, 1 - opacity, color_mask, opacity, 0)

    # --- Convert to PIL and save for download ---
    overlay_img_pil = Image.fromarray(overlay)
    output_filename = "segmentation_overlay.png"
    overlay_img_pil.save(output_filename)


    return overlay_img_pil, damaged_text, output_filename

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
    title="Flood Damage Segmentation and Area Calculation",
    description="Upload a satellite or drone image. The model predicts flood damage (No Damage, Minor, Major, Destroyed) and calculates the percentage of the damaged area. The output is a color-coded overlay."
)

# --- Launch ---
# Note: In a Colab environment, share=True is needed to get a public URL
if __name__ == "__main__":
    demo.launch(share=True)