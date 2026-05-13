import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, classification_report

import config as cfg
from utils import seed_everything, compute_class_weights, plot_training_curves
from dataset import get_loaders
from model import build_model


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, 
            weight=self.alpha,
            reduction='none',
            label_smoothing=cfg.LABEL_SMOOTHING
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class EMAModel:
    """Exponential Moving Average of model weights"""
    def __init__(self, model, decay=0.9997):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, scaler, ema, epoch):
    model.train()
    running_loss = 0.0
    
    # FIXED: Faster mixup decay (kills mixup at 60% of training)
    mixup_prob = cfg.MIXUP_PROB * max(0, 1 - epoch / (cfg.EPOCHS * 0.6))

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        
        # Apply mixup probabilistically
        if np.random.rand() < mixup_prob:
            images, labels_a, labels_b, lam = mixup_data(images, labels, cfg.MIXUP_ALPHA)
            use_mixup = True
        else:
            use_mixup = False

        optimizer.zero_grad()
        
        with autocast():
            output = model(images)
            if use_mixup:
                loss = mixup_criterion(criterion, output, labels_a, labels_b, lam)
            else:
                loss = criterion(output, labels)
        
        scaler.scale(loss).backward()
        
        if cfg.GRAD_CLIP:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        
        scaler.step(optimizer)
        scaler.update()
        
        # FIXED: Update EMA after optimizer step
        if cfg.USE_EMA:
            ema.update()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mixup_p': f'{mixup_prob:.2f}'})
    
    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion):
    """Validate on EMA model"""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            with autocast():
                output = model(images)
                loss = criterion(output, labels)
            
            running_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(loader)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return avg_loss, acc, kappa, all_preds, all_labels


def train():
    seed_everything()
    
    print("="*80)
    print("DIABETIC RETINOPATHY CLASSIFICATION - TRAINING")
    print("="*80)
    print(f"Device: {cfg.DEVICE}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Backbone LR: {cfg.BACKBONE_LR:.2e}")
    print(f"Head LR: {cfg.HEAD_LR:.2e}")
    print(f"Epochs: {cfg.EPOCHS}")
    print(f"EMA Enabled: {cfg.USE_EMA}")
    print("="*80)
    
    # Load data
    train_loader, val_loader, _ = get_loaders()
    
    # Check class distribution
    train_labels = np.array(train_loader.dataset.get_labels())
    unique, counts = np.unique(train_labels, return_counts=True)
    print("\nClass Distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt:5d} ({cnt/len(train_labels)*100:5.2f}%)")
    
    # Build model
    model = build_model()
    
    # FIXED: Calculate class weights using 'sqrt' method (not 'effective')
    weights = compute_class_weights(train_labels, method='sqrt').to(cfg.DEVICE)
    print(f"\nClass Weights: {weights.cpu().numpy()}")
    
    criterion = FocalLoss(alpha=weights, gamma=cfg.FOCAL_GAMMA)
    
    # FIXED: Discriminative learning rates (backbone vs head)
    head_params = []
    backbone_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name or "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    print(f"\nBackbone params: {len(backbone_params)}")
    print(f"Head params: {len(head_params)}")
    
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.BACKBONE_LR},
            {"params": head_params, "lr": cfg.HEAD_LR},
        ],
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    # FIXED: Proper sequential LR scheduler (warmup -> cosine)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.2,
        total_iters=cfg.WARMUP_EPOCHS
    )
    
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.EPOCHS - cfg.WARMUP_EPOCHS,
        eta_min=cfg.MIN_LR
    )
    
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.WARMUP_EPOCHS]
    )
    
    scaler = GradScaler()
    
    # FIXED: Initialize EMA
    ema = EMAModel(model, decay=cfg.EMA_DECAY) if cfg.USE_EMA else None

    # Training tracking
    best_kappa = -1.0
    best_acc = 0.0
    patience_counter = 0
    
    train_losses, val_losses = [], []
    val_kappas, val_accs = [], []
    
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80 + "\n")
    
    for epoch in range(1, cfg.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, ema, epoch
        )
        
        # FIXED: Validate on EMA model
        if cfg.USE_EMA:
            ema.apply_shadow()
        
        val_loss, val_acc, val_kappa, val_preds, val_labels = validate(
            model, val_loader, criterion
        )
        
        if cfg.USE_EMA:
            ema.restore()
        
        # Step scheduler
        scheduler.step()
        
        # Get current LRs
        backbone_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[1]['lr']
        
        # Log
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_kappas.append(val_kappa)
        val_accs.append(val_acc)
        
        print(f"\nEpoch [{epoch}/{cfg.EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Kappa: {val_kappa:.4f}")
        print(f"  LR (backbone/head): {backbone_lr:.2e} / {head_lr:.2e}")
        
        # Save best model based on Kappa
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            best_acc = val_acc
            patience_counter = 0
            
            # Save EMA weights if enabled
            save_dict = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_kappa': val_kappa,
                'val_loss': val_loss,
            }
            
            if cfg.USE_EMA:
                save_dict['ema_state'] = ema.shadow
            
            torch.save(save_dict, cfg.BEST_MODEL_PATH)
            
            print(f"  ⭐ NEW BEST MODEL! Kappa: {val_kappa:.4f}, Acc: {val_acc:.4f}")
            
            # Per-class report
            print("\n  Per-Class Performance:")
            report = classification_report(
                val_labels, val_preds,
                target_names=["No DR", "Mild", "Moderate", "Severe", "Prolif."],
                digits=3,
                zero_division=0
            )
            print(report)
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{cfg.EARLY_STOP_PAT}")
        
        # Early stopping
        if patience_counter >= cfg.EARLY_STOP_PAT:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break
        
        # Save last
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
        }, cfg.LAST_MODEL_PATH)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Validation Kappa: {best_kappa:.4f}")
    print(f"Best Validation Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("="*80)
    
    # FIXED: Plot without fake benchmark line
    plot_training_curves(
        train_losses, val_losses, val_kappas,
        save_path=os.path.join(cfg.LOG_DIR, "training_curves.png")
    )


if __name__ == "__main__":
    train()