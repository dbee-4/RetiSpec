"""
Explainable AI - Grad-CAM++ (Generalized Gradient-Weighted Class Activation Mapping)
Visualizes the macro regional attention of ConvNeXt-Tiny on the retinal fundus.
"""
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config as cfg
from model import build_model
from preprocessing import preprocess_for_model

LABEL_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def get_target_layers(model):
    """
    For ConvNeXt-Tiny, target the final block of the last stage.
    This yields a [B, C, H, W] activation map with rich semantic features.
    """
    return [model.backbone.stages[-1].blocks[-1]]


def create_gradcam_engine(model):
    """Instantiate GradCAMPlusPlus engine on model"""
    target_layers = get_target_layers(model)
    return GradCAMPlusPlus(model=model, target_layers=target_layers)


def compute_gradcam(cam_engine, input_tensor: torch.Tensor, pred_idx: int, vis_img: np.ndarray) -> np.ndarray:
    """
    Generates vibrant Grad-CAM++ regional attention heatmap overlay.
    
    Args:
        cam_engine: GradCAMPlusPlus instance
        input_tensor: [1, 3, H, W] normalized tensor
        pred_idx: Target class index
        vis_img: [H, W, 3] float RGB image in range [0, 1]
        
    Returns:
        overlay: [H, W, 3] float RGB overlay in range [0, 1]
    """
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam_engine(input_tensor=input_tensor, targets=targets)[0, :]
    
    # Normalize CAM to range [0, 1]
    cam_min, cam_max = grayscale_cam.min(), grayscale_cam.max()
    cam_norm = (grayscale_cam - cam_min) / (cam_max - cam_min + 1e-8)
    
    # Smooth with subtle Gaussian kernel for organic contouring
    cam_smooth = cv2.GaussianBlur(cam_norm, (11, 11), 0)
    cam_smooth = (cam_smooth - cam_smooth.min()) / (cam_smooth.max() - cam_smooth.min() + 1e-8)
    
    # Map to JET colormap (Red = high attention, Yellow/Cyan = mid, Blue = low)
    heatmap = cv2.applyColorMap((cam_smooth * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Dynamic alpha: high-attention zones have strong vivid color (up to 70%),
    # while low-attention background remains clear so the retina is not muddy
    alpha = np.clip((cam_smooth[..., None] - 0.15) / 0.85, 0.0, 0.68)
    overlay = (1.0 - alpha) * vis_img + alpha * heatmap_rgb
    
    # Mask out black background padding outside the circular retina
    mask = (vis_img.sum(axis=2) > 0.05).astype(np.float32)[..., None]
    overlay = np.clip(overlay * mask, 0.0, 1.0)
    
    return overlay


def run_gradcam_demo(image_path=None):
    """Runs Grad-CAM++ demo and saves visualization"""
    model = build_model(pretrained=False)
    if os.path.exists(cfg.BEST_MODEL_PATH):
        ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt.get('ema_state', ckpt['model_state']))
        print(f"Loaded checkpoint from: {cfg.BEST_MODEL_PATH}")
    model.eval()

    cam_engine = create_gradcam_engine(model)

    if image_path and os.path.exists(image_path):
        from PIL import Image
        pil_img = Image.open(image_path)
        tensor, vis_img = preprocess_for_model(pil_img, device=cfg.DEVICE)
    else:
        print("Using synthetic sample for Grad-CAM++ demo run...")
        tensor = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(cfg.DEVICE)
        vis_img = np.ones((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), dtype=np.float32) * 0.5

    with torch.no_grad():
        logits = model(tensor)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    print(f"Prediction: {LABEL_NAMES[pred_idx]} ({probs[pred_idx]*100:.1f}%)")
    cam_image = compute_gradcam(cam_engine, tensor, pred_idx, vis_img)

    out_dir = cfg.GRADCAM_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_gradcam_plusplus.png")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(vis_img)
    plt.title("Preprocessed Fundus")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title(f"Grad-CAM++ ({LABEL_NAMES[pred_idx]})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Grad-CAM++ visualization saved to: {out_path}")


if __name__ == "__main__":
    run_gradcam_demo()
