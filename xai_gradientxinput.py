"""
Explainable AI - Gradient x Input (Fine-Grained Pixel-Level Attribution)
Highlights specific microaneurysms, hemorrhages, and vascular lesions
at pixel-level resolution by calculating S(x) = | x * (d y_c / d x) |.
"""
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import config as cfg
from model import build_model
from preprocessing import preprocess_for_model


def compute_gradient_x_input(model, input_tensor: torch.Tensor, pred_idx: int, vis_img: np.ndarray) -> np.ndarray:
    """
    Computes fine-grained, high-contrast Gradient * Input pixel attribution.
    
    Args:
        model: PyTorch ConvNeXtDR model (in eval mode)
        input_tensor: Tensor of shape [1, 3, H, W]
        pred_idx: Target class index to explain
        vis_img: Preprocessed RGB image in range [0, 1] of shape [H, W, 3]
        
    Returns:
        overlay: RGB visualization of shape [H, W, 3] in range [0, 1]
    """
    inp = input_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass
    output = model(inp)
    target_score = output[0, pred_idx]
    
    # Backward pass for the predicted class logit
    model.zero_grad()
    target_score.backward()
    
    # Extract gradients and input tensor values
    gradients = inp.grad.data.cpu().numpy()[0]   # [3, H, W]
    inputs_np = inp.data.cpu().numpy()[0]       # [3, H, W]
    
    # Element-wise Gradient * Input magnitude
    grad_times_input = np.abs(gradients * inputs_np)
    
    # Channel-wise max or sum sensitivity -> [H, W]
    saliency = np.sum(grad_times_input, axis=0)
    
    # Retinal mask for valid pixels
    mask = (vis_img.sum(axis=2) > 0.05).astype(np.float32)
    saliency = saliency * mask
    
    # Percentile clipping (99th percentile) ensures multiple micro-lesions glow vividly
    # instead of being dominated by a single outlier pixel
    valid_saliency = saliency[mask > 0]
    if len(valid_saliency) > 0 and valid_saliency.max() > 0:
        p_cutoff = np.percentile(valid_saliency, 98.5)
        saliency_scaled = np.clip(saliency / (p_cutoff + 1e-8), 0.0, 1.0)
    else:
        saliency_scaled = np.zeros_like(saliency)
    
    # Gentle 5x5 smoothing to eliminate single-pixel sensor noise while preserving sharp vessel/lesion edges
    saliency_smooth = cv2.GaussianBlur(saliency_scaled, (5, 5), 0)
    s_min, s_max = saliency_smooth.min(), saliency_smooth.max()
    saliency_norm = (saliency_smooth - s_min) / (s_max - s_min + 1e-8)
    
    # COLORMAP_HOT: Renders pixels with high sensitivity in glowing fiery colors (Yellow -> Red -> Purple)
    # This provides stark, beautiful contrast against the cooler Grad-CAM++ JET rainbow
    heatmap = cv2.applyColorMap((saliency_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Selective alpha blending: only pixels with positive sensitivity glow over the fundus;
    # non-contributing background pixels stay clean without muddy dark tint
    alpha = np.clip((saliency_norm[..., None] - 0.10) / 0.90, 0.0, 0.75)
    overlay = (1.0 - alpha) * vis_img + alpha * heatmap_rgb
    
    # Mask background border
    overlay = np.clip(overlay * mask[..., None], 0.0, 1.0)
    
    return overlay


def run_gradient_x_input_demo(image_path=None):
    """Demo runner to generate and save Gradient x Input attribution"""
    LABEL_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    
    model = build_model(pretrained=False)
    if os.path.exists(cfg.BEST_MODEL_PATH):
        ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt.get('ema_state', ckpt['model_state']))
        print(f"Loaded checkpoint from: {cfg.BEST_MODEL_PATH}")
    model.eval()
    
    if image_path and os.path.exists(image_path):
        from PIL import Image
        pil_img = Image.open(image_path)
        tensor, vis_img = preprocess_for_model(pil_img, device=cfg.DEVICE)
    else:
        print("Using synthetic sample for demo run...")
        tensor = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(cfg.DEVICE)
        vis_img = np.ones((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), dtype=np.float32) * 0.5
    
    with torch.no_grad():
        logits = model(tensor)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    print(f"Prediction: {LABEL_NAMES[pred_idx]} ({probs[pred_idx]*100:.1f}%)")
    overlay = compute_gradient_x_input(model, tensor, pred_idx, vis_img)
    
    out_dir = cfg.GRADXINPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_gradient_x_input.png")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(vis_img)
    plt.title("Preprocessed Fundus")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Gradient x Input ({LABEL_NAMES[pred_idx]})")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Attribution heatmap saved to: {out_path}")


if __name__ == "__main__":
    run_gradient_x_input_demo()
