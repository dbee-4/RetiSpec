import os
import io
import sys
import cv2
import base64
import torch
import pickle
from tta import TTAPredictor
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from preprocessing import preprocess_for_model, preprocess_for_tta


# Ensure the app can find the 'src' directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from model import build_model


app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
SEVERITY_DESC = [
    "No visible diabetic retinopathy.",
    "Small areas of swelling in retinal blood vessels (microaneurysms).",
    "Leakage in blood vessels impacting the retina.",
    "Many blocked blood vessels, depriving the retina of blood flow.",
    "Advanced stage with growth of new, fragile blood vessels."
]

# ── Model & XAI Setup ─────────────────────────────────────────────────────────
device = cfg.DEVICE
model = build_model()
checkpoint = torch.load(cfg.BEST_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Hook into the target layer for real-time Grad-CAM++
# Targeting the entire block ensures we get the [B, C, H, W] output format instead of the [B, H, W, C] norm output.
target_layers = [model.backbone.stages[-1].blocks[-1]]
cam_engine = GradCAMPlusPlus(model=model, target_layers=target_layers)


# Load optimized thresholds if available
threshold_path = os.path.join(cfg.MODEL_DIR, "optimized_thresholds.pkl")
if os.path.exists(threshold_path):
    with open(threshold_path, 'rb') as f:
        OPTIMIZED_THRESHOLDS = pickle.load(f)
    print(f"ℹ️ Loaded optimized thresholds (not used in Flask softmax inference): {OPTIMIZED_THRESHOLDS}")
else:
    OPTIMIZED_THRESHOLDS = None
    print("⚠️  No optimized thresholds found. Using default (argmax).")

# Initialize TTA predictor
tta_predictor = TTAPredictor(model)

# SHAP explainer removed due to Numba DLL Application Control block
# We will use native PyTorch Gradient * Input to calculate pixel-level feature importance instead.
# ── Helpers ───────────────────────────────────────────────────────────────────



def tensor_to_base64(img_array):
    """Converts a numpy image array to a base64 string for the browser."""
    img_array = (img_array * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_array)
    buff = io.BytesIO()
    img_pil.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    use_tta = request.form.get("use_tta", "false").lower() == "true"
    
    pil_img = Image.open(io.BytesIO(file.read()))
    img_np = np.array(pil_img.convert("RGB"))
    input_tensor, vis_img = preprocess_for_model(pil_img)
     
     
    if use_tta:
        # TTA prediction (requires preprocessed numpy image)
        img_preprocessed = preprocess_for_tta(pil_img)
        avg_probs, _, _ = tta_predictor.predict_with_tta(img_preprocessed, num_augments=6)
        pred_idx = int(np.argmax(avg_probs))
        conf = float(avg_probs[pred_idx])

    else:
        # Standard prediction
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            
            avg_probs = probs.cpu().numpy()
            pred_idx = int(np.argmax(avg_probs))
            conf = float(avg_probs[pred_idx])
    
    # Generate Grad-CAM (input_tensor and vis_img are already preprocessed by preprocess_for_model)
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam_engine(input_tensor=input_tensor, targets=targets)[0, :]
    cam_image = show_cam_on_image(vis_img, grayscale_cam, use_rgb=True)
    
    # Generate Pixel-Level Feature Importance (Native PyTorch Gradient * Input)
    # This mathematically approximates SHAP without requiring the blocked 'numba' DLL
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    score = output[0, pred_idx]
    
    # Clear any previous gradients and compute new ones
    model.zero_grad()
    score.backward()
    
    gradients = input_tensor.grad.data.cpu().numpy()[0]
    inputs_np = input_tensor.data.cpu().numpy()[0]
    
    # Gradient * Input (Sensitivity magnitude)
    grad_times_input = np.abs(gradients * inputs_np)
    shap_for_pred = np.transpose(grad_times_input, (1, 2, 0))
    shap_magnitude = shap_for_pred.sum(axis=2)
    
    # Smooth the SHAP heatmap to reduce pixel noise and make it more understandable
    shap_magnitude = cv2.GaussianBlur(shap_magnitude, (21, 21), 0)
    
    shap_norm = (shap_magnitude - shap_magnitude.min()) / (shap_magnitude.max() - shap_magnitude.min() + 1e-8)
    shap_heatmap = cv2.applyColorMap((shap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    shap_heatmap = cv2.cvtColor(shap_heatmap, cv2.COLOR_BGR2RGB) / 255.0
    shap_overlay = 0.6 * vis_img + 0.4 * shap_heatmap
    shap_overlay = np.clip(shap_overlay, 0, 1)
    
    # Mask out the background (black pixels in the preprocessed image)
    # This prevents the heatmap from coloring the padding outside the retina
    mask = (vis_img.sum(axis=2) > 0.05).astype(np.float32)[..., np.newaxis]
    cam_image = cam_image * mask
    shap_overlay = shap_overlay * mask
    
    # English Explanation
    if pred_idx == 0:
        explanation = "This person is not affected by diabetic retinopathy."
    else:
        explanation = f"This person is affected by {LABEL_NAMES[pred_idx].lower()} diabetic retinopathy. The highlighted regions in the heatmaps indicate the specific areas affecting them."

    return jsonify({
        "prediction": LABEL_NAMES[pred_idx],
        "description": SEVERITY_DESC[pred_idx],
        "explanation": explanation,
        "confidence": round(conf * 100, 2),
        "gradcam_img": tensor_to_base64(cam_image),
        "shap_img": tensor_to_base64(shap_overlay),
        "original_img": tensor_to_base64(vis_img),
        "all_probs": {LABEL_NAMES[i]: round(avg_probs[i] * 100, 2) for i in range(5)},
        "used_tta": use_tta
    })






if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
