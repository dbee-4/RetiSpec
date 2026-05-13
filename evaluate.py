"""
Enhanced evaluation with TTA and Threshold Optimization
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, classification_report
import pickle

import config as cfg
from dataset import get_loaders
from model import build_model
from utils import print_metrics, plot_confusion_matrix, compute_class_weights
from train import FocalLoss
from tta import TTAPredictor, evaluate_with_tta
from threshold_optimization import ThresholdOptimizer


def validate_baseline(model, loader, criterion):
    """Baseline validation (no TTA, no threshold optimization)"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Baseline Eval", leave=False):
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            running_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return avg_loss, accuracy, kappa, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def evaluate_model():
    print("="*80)
    print("ENHANCED EVALUATION: Baseline → TTA → Threshold Optimization")
    print("="*80)
    print(f"Device: {cfg.DEVICE}\n")
    
    # Load data
    _, val_loader, test_loader = get_loaders()

    # Load model
    model = build_model()
    try:
        ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
        
        # Handle EMA weights if present
        if 'ema_state' in ckpt:
            print("Loading EMA weights...")
            ema_state = ckpt['ema_state']
            model_state = model.state_dict()
            for name in model_state.keys():
                if name in ema_state:
                    model_state[name] = ema_state[name]
            model.load_state_dict(model_state)
        else:
            model.load_state_dict(ckpt['model_state'])
        
        print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']}")
        print(f"   Saved Val Kappa: {ckpt.get('val_kappa', 'N/A')}")
    except FileNotFoundError:
        print("❌ Best model not found. Run training first.")
        return

    # Setup loss
    all_train_labels = pd.read_csv(cfg.TRAIN_CSV)["label"].values
    class_weights = compute_class_weights(all_train_labels, method='sqrt').to(cfg.DEVICE)
    criterion = FocalLoss(alpha=class_weights)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: BASELINE EVALUATION (No TTA, No Threshold Opt)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─"*80)
    print("STEP 1: BASELINE EVALUATION (Standard Inference)")
    print("─"*80)
    
    loss, acc, kappa, y_true, y_pred_base, y_prob_base = validate_baseline(
        model, test_loader, criterion
    )
    
    print(f"\n📊 Baseline Results:")
    print(f"   Accuracy: {acc*100:.2f}%")
    print(f"   Quadratic Kappa: {kappa:.4f}")
    
    print("\n   Per-Class Performance:")
    print(classification_report(
        y_true, y_pred_base,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Prolif."],
        digits=3, zero_division=0
    ))
    
    # Save baseline confusion matrix
    plot_confusion_matrix(
        y_true, y_pred_base,
        save_path=os.path.join(cfg.LOG_DIR, "baseline_confusion_matrix.png")
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: TEST-TIME AUGMENTATION (TTA)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─"*80)
    print("STEP 2: TEST-TIME AUGMENTATION (6 augmentations)")
    print("─"*80)
    print("Augmentations: Original + HFlip + Rotations(±5°) + Brightness(±5%)")
    print("This will take 6x longer than baseline...\n")
    
    y_pred_tta, y_true_tta, y_prob_tta = evaluate_with_tta(
        model, test_loader, num_augments=6
    )
    
    acc_tta = np.mean(y_pred_tta == y_true_tta)
    kappa_tta = cohen_kappa_score(y_true_tta, y_pred_tta, weights='quadratic')
    
    print(f"\n📊 TTA Results:")
    print(f"   Accuracy: {acc_tta*100:.2f}% (Δ: {(acc_tta-acc)*100:+.2f}%)")
    print(f"   Quadratic Kappa: {kappa_tta:.4f} (Δ: {(kappa_tta-kappa):+.4f})")
    
    print("\n   Per-Class Performance:")
    print(classification_report(
        y_true_tta, y_pred_tta,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Prolif."],
        digits=3, zero_division=0
    ))
    
    # Save TTA confusion matrix
    plot_confusion_matrix(
        y_true_tta, y_pred_tta,
        save_path=os.path.join(cfg.LOG_DIR, "tta_confusion_matrix.png")
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: THRESHOLD OPTIMIZATION (on Validation Set)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─"*80)
    print("STEP 3: THRESHOLD OPTIMIZATION (Using Validation Set)")
    print("─"*80)
    
    # Get validation predictions for threshold tuning
    print("Collecting validation predictions...")
    _, _, _, y_val_true, _, y_val_prob = validate_baseline(
        model, val_loader, criterion
    )
    
    # Optimize thresholds
    optimizer = ThresholdOptimizer(y_val_true, y_val_prob)
    
    print("\nMethod 1: Class-Specific Thresholds")
    best_thresholds_cs, kappa_cs = optimizer.optimize_thresholds_class_specific()
    
    print("\nMethod 2: Grid Search Thresholds")
    best_thresholds_gs, kappa_gs = optimizer.optimize_thresholds_grid_search(n_steps=15)
    
    # Choose best method
    if kappa_cs > kappa_gs:
        print(f"\n✅ Using Class-Specific Thresholds (Kappa: {kappa_cs:.4f})")
        best_thresholds = best_thresholds_cs
    else:
        print(f"\n✅ Using Grid Search Thresholds (Kappa: {kappa_gs:.4f})")
        best_thresholds = best_thresholds_gs
    
    # Save thresholds
    threshold_save_path = os.path.join(cfg.MODEL_DIR, "optimized_thresholds.pkl")
    with open(threshold_save_path, 'wb') as f:
        pickle.dump(best_thresholds, f)
    print(f"Saved thresholds to: {threshold_save_path}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: APPLY THRESHOLDS TO TEST SET
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─"*80)
    print("STEP 4: APPLYING OPTIMIZED THRESHOLDS TO TEST SET")
    print("─"*80)
    
    # Apply to baseline predictions
    test_optimizer = ThresholdOptimizer(y_true, y_prob_base)
    test_optimizer.best_thresholds = best_thresholds
    y_pred_thresh = test_optimizer.predict_with_thresholds(y_prob_base)
    
    acc_thresh = np.mean(y_pred_thresh == y_true)
    kappa_thresh = cohen_kappa_score(y_true, y_pred_thresh, weights='quadratic')
    
    print(f"\n📊 Threshold Optimized Results:")
    print(f"   Accuracy: {acc_thresh*100:.2f}% (Δ: {(acc_thresh-acc)*100:+.2f}%)")
    print(f"   Quadratic Kappa: {kappa_thresh:.4f} (Δ: {(kappa_thresh-kappa):+.4f})")
    
    # Save threshold-optimized confusion matrix
    plot_confusion_matrix(
        y_true, y_pred_thresh,
        save_path=os.path.join(cfg.LOG_DIR, "threshold_optimized_confusion_matrix.png")
    )
    
    # Plot threshold impact
    test_optimizer.plot_threshold_impact(
        save_path=os.path.join(cfg.LOG_DIR, "threshold_impact.png")
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: COMBINE TTA + THRESHOLD OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─"*80)
    print("STEP 5: ULTIMATE COMBO (TTA + Threshold Optimization)")
    print("─"*80)
    
    y_pred_final = test_optimizer.predict_with_thresholds(y_prob_tta)
    
    acc_final = np.mean(y_pred_final == y_true_tta)
    kappa_final = cohen_kappa_score(y_true_tta, y_pred_final, weights='quadratic')
    
    print(f"\n📊 Final Results (TTA + Thresholds):")
    print(f"   Accuracy: {acc_final*100:.2f}% (Δ: {(acc_final-acc)*100:+.2f}%)")
    print(f"   Quadratic Kappa: {kappa_final:.4f} (Δ: {(kappa_final-kappa):+.4f})")
    
    print("\n   Per-Class Performance:")
    print(classification_report(
        y_true_tta, y_pred_final,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Prolif."],
        digits=3, zero_division=0
    ))
    
    # Save final confusion matrix
    plot_confusion_matrix(
        y_true_tta, y_pred_final,
        save_path=os.path.join(cfg.LOG_DIR, "final_tta_threshold_confusion_matrix.png")
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("FINAL SUMMARY - ACCURACY PROGRESSION")
    print("="*80)
    print(f"{'Method':<30} {'Accuracy':<15} {'Kappa':<15} {'Improvement'}")
    print("-"*80)
    print(f"{'1. Baseline':<30} {acc*100:>6.2f}%        {kappa:>6.4f}        --")
    print(f"{'2. + TTA (6 augments)':<30} {acc_tta*100:>6.2f}%        {kappa_tta:>6.4f}        {(kappa_tta-kappa)*100:>+5.2f}%")
    print(f"{'3. + Threshold Opt':<30} {acc_thresh*100:>6.2f}%        {kappa_thresh:>6.4f}        {(kappa_thresh-kappa)*100:>+5.2f}%")
    print(f"{'4. TTA + Threshold (BEST)':<30} {acc_final*100:>6.2f}%        {kappa_final:>6.4f}        {(kappa_final-kappa)*100:>+5.2f}%")
    print("="*80)
    
    # Save summary
    summary = {
        'baseline': {'acc': acc, 'kappa': kappa},
        'tta': {'acc': acc_tta, 'kappa': kappa_tta},
        'threshold': {'acc': acc_thresh, 'kappa': kappa_thresh},
        'final': {'acc': acc_final, 'kappa': kappa_final},
        'thresholds': best_thresholds
    }
    
    summary_path = os.path.join(cfg.LOG_DIR, "evaluation_summary.pkl")
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\n✅ Evaluation complete! Results saved to {cfg.LOG_DIR}")
    print(f"\nGenerated files:")
    print(f"  - baseline_confusion_matrix.png")
    print(f"  - tta_confusion_matrix.png")
    print(f"  - threshold_optimized_confusion_matrix.png")
    print(f"  - final_tta_threshold_confusion_matrix.png")
    print(f"  - threshold_impact.png")
    print(f"  - evaluation_summary.pkl")
    print(f"  - optimized_thresholds.pkl (in models/ dir)")


if __name__ == "__main__":
    evaluate_model()