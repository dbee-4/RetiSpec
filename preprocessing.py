# preprocessing.py

import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from typing import Tuple
import config as cfg


# ============================================================
# FUNDUS PREPROCESSING (Match training pipeline)
# ============================================================

def ben_graham_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Ben Graham's preprocessing technique for retinal fundus images.
    Improves image quality and contrast. Expects BGR image.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    height, width = image.shape[:2]
    if height > 1000 or width > 1000:
        scale = 1000 / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab[:, :, 0] = l_channel
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    img_float = image.astype(np.float32) / 255.0
    background = cv2.medianBlur(image, 51)
    background = background.astype(np.float32) / 255.0
    image = cv2.addWeighted(img_float, 4, background, -4, 128 / 255.0)
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    
    return image

def circular_crop(image: np.ndarray, radius_ratio: float = 0.95) -> np.ndarray:
    """Crop retinal image to circular region (remove black borders)."""
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    radius = int(min(height, width) * radius_ratio / 2)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    if len(image.shape) == 3:
        result = cv2.bitwise_and(image, image, mask=mask)
    else:
        result = cv2.bitwise_and(image, mask)
    
    return result

def resize_image_padded(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image while preserving aspect ratio and padding."""
    h, w = size
    img_h, img_w = image.shape[:2]
    
    scale = min(h / img_h, w / img_w)
    new_h, new_w = int(img_h * scale), int(img_w * scale)
    
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    if len(image.shape) == 3:
        canvas = np.zeros((h, w, image.shape[2]), dtype=image.dtype)
    else:
        canvas = np.zeros((h, w), dtype=image.dtype)
    
    pad_y = (h - new_h) // 2
    pad_x = (w - new_w) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = image
    
    return canvas

def apply_fundus_preprocessing(image_rgb: np.ndarray) -> np.ndarray:
    """
    Applies the full fundus preprocessing pipeline on an RGB image.
    Returns a preprocessed RGB image.
    """
    # Convert RGB to BGR for Ben Graham preprocessing
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    image_bgr = ben_graham_preprocessing(image_bgr)
    image_bgr = circular_crop(image_bgr)
    image_bgr = resize_image_padded(image_bgr, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
    
    # Convert back to RGB
    image_rgb_processed = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb_processed


# ============================================================
# BASE TRANSFORMS (TRAIN / INFERENCE CONSISTENCY)
# ============================================================

def get_base_transform():
    """
    Shared transform for training + inference (CRITICAL for consistency)
    """
    return A.Compose([
        A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),

        # Mild enhancement (safe for medical images)
        A.CLAHE(clip_limit=2.0, p=0.3),

        # Normalize for ImageNet pretrained ConvNeXt
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        ToTensorV2(),
    ])


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_input):
    """
    Accepts:
    - PIL Image
    - file path
    - numpy array

    Returns: RGB numpy image
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))

    elif isinstance(image_input, np.ndarray):
        img = image_input

        # if BGR accidentally passed
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise TypeError("Unsupported image input type")

    return img


# ============================================================
# MODEL INPUT PREPROCESSING (FLASK / INFERENCE)
# ============================================================

def preprocess_for_model(image_input, device=cfg.DEVICE):
    """
    Main function used in app.py

    Returns:
    - tensor [1, 3, H, W]
    - visualization image (0–1 float RGB)
    """
    img = load_image(image_input)
    
    # Apply standard fundus preprocessing FIRST
    img = apply_fundus_preprocessing(img)

    # Visualization version (for Grad-CAM)
    # The image is already resized by apply_fundus_preprocessing, but we ensure it here
    vis_img = cv2.resize(img, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
    vis_img = vis_img.astype(np.float32) / 255.0

    # Model input
    transform = get_base_transform()
    tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    return tensor, vis_img


# ============================================================
# BATCH PREPROCESSING (TRAINING PIPELINE)
# ============================================================

def preprocess_batch(images):
    """
    For dataset loader usage if needed
    """
    transform = get_base_transform()
    processed = []

    for img in images:
        img = load_image(img)
        tensor = transform(image=img)["image"]
        processed.append(tensor)

    return torch.stack(processed)


# ============================================================
# TTA SAFE INPUT CONVERSION
# ============================================================

def preprocess_for_tta(image_input):
    """
    TTA expects raw numpy RGB image (uint8), but it MUST be preprocessed
    first so augmentations are applied on the correct image type.
    """
    img = load_image(image_input)
    img = apply_fundus_preprocessing(img)
    return img


# ============================================================
# DEBUG VISUALIZATION HELPER
# ============================================================

def denormalize(img_tensor):
    """
    Convert normalized tensor back to image (for debugging only)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * std) + mean
    img = np.clip(img, 0, 1)

    return img