import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, cohen_kappa_score
)
import config as cfg


def seed_everything(seed=cfg.SEED):
    """FIXED: Enable benchmark for better performance (disable for debugging only)"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # FIXED: Enable benchmark for faster training
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def compute_class_weights(labels, method='sqrt'):
    """
    FIXED: Default to 'sqrt' not 'effective'
    When using weighted sampler + focal loss, 'effective' is too aggressive
    """
    counts = np.bincount(labels, minlength=cfg.NUM_CLASSES).astype(float)
    
    if method == 'inverse':
        weights = 1.0 / (counts + 1e-6)
    elif method == 'sqrt':
        total = len(labels)
        weights = np.sqrt(total / (cfg.NUM_CLASSES * counts + 1e-6))
    else:  # effective
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
    
    weights = weights / np.sum(weights) * cfg.NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Enhanced confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_perc = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    labels = ["No DR", "Mild", "Moderate", "Severe", "Prolif."]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Counts
    im1 = ax1.imshow(cm, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_yticklabels(labels)
    ax1.set_title("Confusion Matrix (Counts)")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax1.text(j, i, f'{cm[i, j]}', ha='center', va='center')
    
    # Percentages
    im2 = ax2.imshow(cm_perc, cmap='Greens', aspect='auto')
    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_yticklabels(labels)
    ax2.set_title("Normalized (Recall)")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax2.text(j, i, f'{cm_perc[i, j]:.2%}', ha='center', va='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_losses, val_losses, val_kappas, save_path=None):
    """FIXED: Removed fake benchmark line"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(epochs, train_losses, label="Train Loss", color='#3b82f6', linewidth=2)
    axes[0].plot(epochs, val_losses, label="Val Loss", color='#f97316', linewidth=2)
    axes[0].set_title("Loss Convergence", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Kappa
    axes[1].plot(epochs, val_kappas, label="Val Kappa", color='#f97316', linewidth=2)
    # FIXED: Removed misleading benchmark line
    axes[1].set_title("Quadratic Weighted Kappa", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Kappa")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_metrics(y_true, y_pred, y_prob=None):
    """Print detailed metrics"""
    labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(
        y_true, y_pred, 
        target_names=labels, 
        digits=4,
        zero_division=0
    ))
    
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    print(f"\nQuadratic Weighted Kappa: {qwk:.4f}")

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
            print(f"Mean ROC-AUC: {auc:.4f}")
        except Exception:
            print("ROC-AUC: Could not calculate")
    
    print("="*80 + "\n")


def get_lr(optimizer):
    """FIXED: Return all LRs for multi-group optimizer"""
    return {
        f"group_{i}": pg["lr"]
        for i, pg in enumerate(optimizer.param_groups)
    }