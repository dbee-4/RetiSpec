import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config as cfg


def get_train_transforms():
    """Conservative augmentation - safe for retinal fundus images"""
    return A.Compose([
        A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        
        # FIXED: Only horizontal flip (vertical flip removed - anatomically unsafe)
        A.HorizontalFlip(p=0.5),
        
        # FIXED: RandomRotate90 removed - too aggressive for fundus orientation
        # Small rotations only via ShiftScaleRotate
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.08,
            rotate_limit=12,  # Reduced from 15
            border_mode=cv2.BORDER_CONSTANT,
            p=0.4
        ),
        
        # FIXED: Reduced color augmentation intensity
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.12,  # Reduced from 0.15
                contrast_limit=0.12,
                p=1.0
            ),
            A.CLAHE(clip_limit=2.0, p=1.0),
        ], p=0.35),  # Reduced probability
        
        # FIXED: Reduced blur/sharpen probability
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.Sharpen(alpha=(0.1, 0.25), p=1.0),
        ], p=0.15),  # Reduced from 0.2
        
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


class DRDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.base_dir = cfg.BASE_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        label = int(row["label"])

        if not os.path.isabs(path):
            path = os.path.join(self.base_dir, path)

        img = cv2.imread(path)
        
        # FIXED: Fail fast instead of silent gray image
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label

    def get_labels(self):
        return self.df["label"].tolist()


def build_weighted_sampler(dataset):
    """FIXED: Tempered reweighting (power=0.35 instead of sqrt=0.5)"""
    labels = np.array(dataset.get_labels())
    class_counts = np.bincount(labels, minlength=cfg.NUM_CLASSES).astype(float)
    
    total = len(labels)
    
    # FIXED: Use configurable power (0.35 is sweet spot for DR)
    class_weights = (total / (cfg.NUM_CLASSES * class_counts + 1e-6)) ** cfg.SAMPLER_POWER
    
    sample_weights = torch.tensor(
        [class_weights[l] for l in labels], 
        dtype=torch.float32
    )
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def get_loaders():
    train_ds = DRDataset(cfg.TRAIN_CSV, transform=get_train_transforms())
    val_ds = DRDataset(cfg.VAL_CSV, transform=get_val_transforms())
    test_ds = DRDataset(cfg.TEST_CSV, transform=get_val_transforms())

    sampler = build_weighted_sampler(train_ds) if cfg.USE_BALANCED_SAMPLER else None
    shuffle = sampler is None

    # FIXED: Added persistent_workers and prefetch_factor
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
        prefetch_factor=2 if cfg.NUM_WORKERS > 0 else None,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
        prefetch_factor=2 if cfg.NUM_WORKERS > 0 else None,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
        prefetch_factor=2 if cfg.NUM_WORKERS > 0 else None,
    )
    
    return train_loader, val_loader, test_loader