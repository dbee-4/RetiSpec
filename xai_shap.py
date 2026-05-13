"""
Efficient SHAP analysis for DR model
Generates pixel-level explanations
"""
import os
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

import config as cfg
from model import build_model
from dataset import DRDataset, get_val_transforms


LABEL_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def prepare_shap_data(dataset, n_background=30, n_test=5, balance_classes=True):
    """
    Prepare data for SHAP analysis
    
    Args:
        n_background: Background samples (baseline)
        n_test: Test samples to explain
        balance_classes: Sample evenly across classes
    """
    if balance_classes:
        # Get indices per class
        labels = dataset.get_labels()
        class_indices = {i: [] for i in range(cfg.NUM_CLASSES)}
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        # Sample background (mostly No DR for baseline)
        bg_indices = []
        bg_indices.extend(np.random.choice(class_indices[0], min(20, len(class_indices[0])), replace=False))
        for c in range(1, cfg.NUM_CLASSES):
            if len(class_indices[c]) > 0:
                bg_indices.extend(np.random.choice(class_indices[c], min(2, len(class_indices[c])), replace=False))
        
        bg_indices = bg_indices[:n_background]
        
        # Sample test (1 per class if possible)
        test_indices = []
        per_class = max(1, n_test // cfg.NUM_CLASSES)
        for c in range(cfg.NUM_CLASSES):
            if len(class_indices[c]) > 0:
                test_indices.extend(np.random.choice(
                    class_indices[c], 
                    min(per_class, len(class_indices[c])), 
                    replace=False
                ))
        test_indices = test_indices[:n_test]
    else:
        # Random sampling
        bg_indices = np.random.choice(len(dataset), n_background, replace=False)
        test_indices = np.random.choice(len(dataset), n_test, replace=False)
    
    # Load images
    bg_images = torch.stack([dataset[i][0] for i in bg_indices])
    test_images = torch.stack([dataset[i][0] for i in test_indices])
    test_labels = [dataset[i][1] for i in test_indices]
    
    return bg_images, test_images, test_labels, test_indices


def denormalize_image(img_tensor):
    """Convert normalized tensor back to displayable image"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_np = img_tensor.detach().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0, 1)
    
    return img_np


def run_shap_analysis(n_background=10, n_test=3):
    """
    Run SHAP analysis on test samples
    """
    print("="*80)
    print("SHAP ANALYSIS - Pixel-Level Feature Importance")
    print("="*80)
    
    # Load model
    model = build_model()
    ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
    
    if 'ema_state' in ckpt:
        ema_state = ckpt['ema_state']
        model_state = model.state_dict()
        for name in model_state.keys():
            if name in ema_state:
                model_state[name] = ema_state[name]
        model.load_state_dict(model_state)
    else:
        model.load_state_dict(ckpt['model_state'])
    
    model.eval()
    print(f"✅ Loaded model from epoch {ckpt['epoch']}\n")
    
    # Prepare data
    dataset = DRDataset(cfg.TEST_CSV, transform=get_val_transforms())
    print(f"Preparing SHAP data...")
    print(f"  Background samples: {n_background}")
    print(f"  Test samples: {n_test}\n")
    
    bg_images, test_images, test_labels, test_indices = prepare_shap_data(
        dataset, n_background, n_test, balance_classes=True
    )
    
    bg_images = bg_images.to(cfg.DEVICE)
    test_images = test_images.to(cfg.DEVICE)
    
    # Initialize SHAP
    print("Initializing SHAP GradientExplainer...")
    explainer = shap.GradientExplainer(model, bg_images)
    
    # Compute SHAP values
    print(f"Computing SHAP values (this takes ~2-3 minutes)...")
    test_images.requires_grad_(True)
    shap_values = explainer.shap_values(test_images)
    
    print("✅ SHAP computation complete!\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    for i in tqdm(range(len(test_images)), desc="Saving SHAP plots"):
        # Get prediction
        with torch.no_grad():
            output = model(test_images[i:i+1])
            pred_class = torch.argmax(output, dim=1).item()
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        
        # Denormalize image
        img_display = denormalize_image(test_images[i])
        
        # SHAP values for predicted class
        shap_for_pred = np.transpose(shap_values[pred_class][i], (1, 2, 0))
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original image
        axes[0].imshow(img_display)
        axes[0].set_title(f"Original Image\nTrue: {LABEL_NAMES[test_labels[i]]}", fontsize=12)
        axes[0].axis('off')
        
        # SHAP heatmap
        shap_magnitude = np.abs(shap_for_pred).sum(axis=2)
        im = axes[1].imshow(shap_magnitude, cmap='hot')
        axes[1].set_title(f"SHAP Importance\nPredicted: {LABEL_NAMES[pred_class]}", fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay
        overlay = img_display.copy()
        shap_norm = (shap_magnitude - shap_magnitude.min()) / (shap_magnitude.max() - shap_magnitude.min() + 1e-8)
        heatmap = cv2.applyColorMap((shap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.6 * img_display + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"SHAP Overlay\nConfidence: {probs[pred_class]*100:.1f}%", fontsize=12)
        axes[2].axis('off')
        
        # Add probability distribution
        prob_text = "\n".join([f"{LABEL_NAMES[c]}: {probs[c]*100:.1f}%" for c in range(cfg.NUM_CLASSES)])
        fig.text(0.98, 0.5, prob_text, fontsize=10, ha='left', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = os.path.join(
            cfg.SHAP_DIR, 
            f"shap_idx{test_indices[i]}_true{test_labels[i]}_pred{pred_class}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n✅ SHAP analysis complete!")
    print(f"   Saved {n_test} visualizations to: {cfg.SHAP_DIR}")
    print(f"\n💡 Interpretation Guide:")
    print(f"   - Red/Yellow regions = High importance (model focuses here)")
    print(f"   - Blue/Dark regions = Low importance (ignored by model)")
    print(f"   - Check if model focuses on lesions vs. artifacts")


if __name__ == "__main__":
    run_shap_analysis(n_background=10, n_test=3)