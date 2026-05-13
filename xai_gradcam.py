import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import config as cfg
from model import build_model
from dataset import DRDataset, get_val_transforms

# Labels for clinical context
LABEL_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

def get_target_layers(model):
    """
    For ConvNeXt-Tiny, the most effective layer for Grad-CAM is the 
    final normalization layer of the last stage.
    """
    # backbone.stages[-1].blocks[-1].norm is the standard target for ConvNeXt
    return [model.backbone.stages[-1].blocks[-1].norm]

def run_gradcam_on_samples(n_samples=10):
    """
    Generates Grad-CAM++ heatmaps for validation samples.
    """
    # 1. Setup
    model = build_model()
    checkpoint = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    target_layers = get_target_layers(model)
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    # 2. Dataset
    ds = DRDataset(cfg.VAL_CSV, transform=get_val_transforms())
    indices = np.random.choice(len(ds), n_samples, replace=False)

    print(f"Generating Grad-CAM++ for {n_samples} samples...")

    for i, idx in enumerate(indices):
        input_tensor, label = ds[idx]
        input_batch = input_tensor.unsqueeze(0).to(cfg.DEVICE)

        # 3. Model Prediction
        with torch.no_grad():
            output = model(input_batch)
            pred_idx = torch.argmax(output, dim=1).item()
        
        # 4. Generate CAM
        # We target the actual prediction to see what the model "saw"
        targets = [ClassifierOutputTarget(pred_idx)]
        
        # cam() returns a list of maps, we take the first (and only) one
        grayscale_cam = cam(input_tensor=input_batch, targets=targets)[0, :]

        # 5. Prepare Visualization
        # Reverse normalization for visualization
        rgb_img = input_tensor.permute(1, 2, 0).cpu().numpy()
        rgb_img = (rgb_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        rgb_img = np.clip(rgb_img, 0, 1)

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # 6. Plotting
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_img)
        plt.title(f"Original (Idx: {idx})\nTrue Label: {LABEL_NAMES[label]}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title(f"Grad-CAM++\nPredicted: {LABEL_NAMES[pred_idx]}")
        plt.axis('off')

        save_path = os.path.join(cfg.GRADCAM_DIR, f"cam_sample_{idx}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"✅ Grad-CAM++ visualizations saved to {cfg.GRADCAM_DIR}")

if __name__ == "__main__":
    run_gradcam_on_samples()