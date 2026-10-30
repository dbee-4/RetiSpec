"""
RetiSpec - FastAPI Web Application
High-performance, async-ready API for Diabetic Retinopathy classification
and Explainable AI (Grad-CAM++ and Gradient x Input).
"""
import os
import io
import cv2
import base64
import pickle
import numpy as np
from PIL import Image
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import config as cfg
from model import build_model
from tta import TTAPredictor
from preprocessing import preprocess_for_model, preprocess_for_tta
from xai_gradcam import create_gradcam_engine, compute_gradcam
from xai_gradientxinput import compute_gradient_x_input

# ── App Initialization ────────────────────────────────────────────────────────
app = FastAPI(
    title="RetiSpec API",
    description="Explainable AI-Based Diabetic Retinopathy Detection System",
    version="2.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files & Templates
BASE_DIR = cfg.BASE_DIR
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=templates_dir)

# ── Clinical Metadata ─────────────────────────────────────────────────────────
LABEL_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
SEVERITY_DESC = [
    "No visible diabetic retinopathy.",
    "Small areas of swelling in retinal blood vessels (microaneurysms).",
    "Leakage in blood vessels impacting the retina.",
    "Many blocked blood vessels, depriving the retina of blood flow.",
    "Advanced stage with growth of new, fragile blood vessels."
]

# ── Model & XAI State ─────────────────────────────────────────────────────────
device = cfg.DEVICE
print(f"[RetiSpec] Initializing on compute device: {device}")

# 1. PyTorch Model Initialization
pytorch_model = build_model(pretrained=False, device=device)
if os.path.exists(cfg.BEST_MODEL_PATH):
    checkpoint = torch.load(cfg.BEST_MODEL_PATH, map_location=device)
    state_dict = checkpoint.get("ema_state", checkpoint.get("model_state", checkpoint))
    pytorch_model.load_state_dict(state_dict)
    print(f"[RetiSpec] Loaded PyTorch model from {cfg.BEST_MODEL_PATH}")
else:
    print(f"[RetiSpec] Warning: Checkpoint not found at {cfg.BEST_MODEL_PATH}")
pytorch_model.eval()

# 2. XAI Engines
cam_engine = create_gradcam_engine(pytorch_model)
tta_predictor = TTAPredictor(pytorch_model, device=device)

# 3. ONNX Runtime Engine (Optional Acceleration)
onnx_session = None
if os.path.exists(cfg.ONNX_MODEL_PATH):
    try:
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_session = ort.InferenceSession(cfg.ONNX_MODEL_PATH, sess_opts, providers=["CPUExecutionProvider"])
        print(f"[RetiSpec] Loaded ONNX Runtime engine from {cfg.ONNX_MODEL_PATH}")
    except Exception as e:
        print(f"[RetiSpec] ONNX Runtime could not be initialized: {e}")

# 4. Optimized Thresholds
OPTIMIZED_THRESHOLDS = None
if os.path.exists(cfg.THRESHOLDS_PATH):
    try:
        with open(cfg.THRESHOLDS_PATH, "rb") as f:
            OPTIMIZED_THRESHOLDS = pickle.load(f)
        print(f"[RetiSpec] Loaded optimized decision thresholds: {OPTIMIZED_THRESHOLDS}")
    except Exception as e:
        print(f"[RetiSpec] Failed loading thresholds: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def numpy_to_base64_png(img_array: np.ndarray) -> str:
    """Converts a numpy image array (float 0-1 or uint8 0-255) to a base64 string."""
    if img_array.dtype != np.uint8:
        if img_array.max() > 1.0:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        else:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_array)
    buff = io.BytesIO()
    img_pil.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the web UI frontend."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health_check():
    """System health and runtime status check."""
    return {
        "status": "healthy",
        "framework": "FastAPI",
        "device": str(device),
        "pytorch_model_loaded": pytorch_model is not None,
        "onnx_model_available": onnx_session is not None,
        "xai_methods": ["Grad-CAM++", "Gradient x Input"],
        "num_classes": cfg.NUM_CLASSES
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    use_tta: bool = Form(True),
    prefer_onnx: bool = Form(False)
):
    """
    Diagnostic inference endpoint:
    - Preprocesses uploaded fundus image (Ben Graham, CLAHE, Circular Crop)
    - Performs DR classification (PyTorch or ONNX Runtime, with optional TTA)
    - Computes dual Explainable AI visualizations: Grad-CAM++ and Gradient x Input
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image")

    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

    # 1. Preprocessing for model inference & visualization
    input_tensor, vis_img = preprocess_for_model(pil_img, device=device)

    # 2. Inference
    engine_used = "PyTorch CPU" if not torch.cuda.is_available() else "PyTorch CUDA"

    if use_tta:
        # Test-Time Augmentation (6 variants)
        img_preprocessed = preprocess_for_tta(pil_img)
        avg_probs, pred_idx, _ = tta_predictor.predict_with_tta(img_preprocessed, num_augments=6)
        conf = float(avg_probs[pred_idx])
        engine_used += " + TTA"
    elif prefer_onnx and onnx_session is not None:
        # Fast ONNX Runtime forward pass
        ort_inputs = {"input": input_tensor.cpu().numpy()}
        logits = onnx_session.run(None, ort_inputs)[0]
        probs = torch.softmax(torch.from_numpy(logits), dim=1)[0].numpy()
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        avg_probs = probs
        engine_used = "ONNX Runtime CPU"
    else:
        # Standard PyTorch forward pass
        with torch.no_grad():
            logits = pytorch_model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            avg_probs = probs

    # 3. Explainable AI Visualizations (Two forms only: Grad-CAM++ & Gradient x Input)
    # Form 1: Grad-CAM++ (Attention on final ConvNeXt feature map)
    cam_overlay = compute_gradcam(cam_engine, input_tensor, pred_idx, vis_img)

    # Form 2: Gradient x Input (Pixel-level sensitivity attribution)
    grad_x_input_overlay = compute_gradient_x_input(pytorch_model, input_tensor, pred_idx, vis_img)

    # 4. Clinical Natural-Language Explanation
    if pred_idx == 0:
        explanation = "This person is not affected by diabetic retinopathy."
    else:
        explanation = (
            f"This person is affected by {LABEL_NAMES[pred_idx].lower()} diabetic retinopathy. "
            f"The highlighted regions in the heatmaps indicate the specific microvascular lesions driving the diagnosis."
        )

    return JSONResponse({
        "prediction": LABEL_NAMES[pred_idx],
        "description": SEVERITY_DESC[pred_idx],
        "explanation": explanation,
        "confidence": round(conf * 100, 2),
        "gradcam_img": numpy_to_base64_png(cam_overlay),
        "gradient_input_img": numpy_to_base64_png(grad_x_input_overlay),
        "shap_img": numpy_to_base64_png(grad_x_input_overlay),  # Backwards compatibility alias
        "original_img": numpy_to_base64_png(vis_img),
        "all_probs": {LABEL_NAMES[i]: round(float(avg_probs[i]) * 100, 2) for i in range(5)},
        "used_tta": use_tta,
        "engine": engine_used
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
