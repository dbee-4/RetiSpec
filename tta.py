import torch
import numpy as np
import cv2
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config as cfg


class TTAPredictor:
    """Test-Time Augmentation for improved accuracy"""
    
    def __init__(self, model, device=cfg.DEVICE):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Base transform (no augmentation)
        self.base_transform = A.Compose([
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def get_tta_transforms(self):
        """Generate all TTA variants"""
        transforms = []
        
        # Original
        transforms.append(lambda img: img)
        
        # Horizontal flip
        transforms.append(lambda img: cv2.flip(img, 1))
        
        # Small rotations
        for angle in [-5, 5]:
            transforms.append(lambda img, a=angle: self._rotate(img, a))
        
        # Brightness adjustments
        for factor in [0.95, 1.05]:
            transforms.append(lambda img, f=factor: self._adjust_brightness(img, f))
        
        return transforms
    
    def _rotate(self, img, angle):
        """Rotate image by angle"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def _adjust_brightness(self, img, factor):
        """Adjust brightness"""
        return np.clip(img * factor, 0, 255).astype(np.uint8)
    
    def predict_with_tta(self, img_np, num_augments=6):
        """
        Predict with TTA
        
        Args:
            img_np: numpy array (H, W, 3) in RGB
            num_augments: how many TTA variants to use (max 6)
        
        Returns:
            avg_probs: averaged probabilities [num_classes]
            pred_class: predicted class index
            all_probs: list of individual predictions
        """
        tta_transforms = self.get_tta_transforms()[:num_augments]
        all_probs = []
        
        with torch.no_grad():
            for transform_fn in tta_transforms:
                # Apply TTA transform
                aug_img = transform_fn(img_np.copy())
                
                # Apply base preprocessing
                transformed = self.base_transform(image=aug_img)["image"]
                input_tensor = transformed.unsqueeze(0).to(self.device)
                
                # Predict
                with autocast():
                    logits = self.model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                
                all_probs.append(probs)
        
        # Average predictions
        avg_probs = np.mean(all_probs, axis=0)
        pred_class = np.argmax(avg_probs)
        
        return avg_probs, pred_class, all_probs


def evaluate_with_tta(model, loader, num_augments=6):
    """
    Evaluate entire dataset with TTA
    
    Returns:
        all_preds: predicted classes
        all_labels: true labels
        all_probs: averaged probabilities
    """
    predictor = TTAPredictor(model)
    all_preds = []
    all_labels = []
    all_probs = []
    
    from tqdm import tqdm
    
    for images, labels in tqdm(loader, desc="TTA Evaluation"):
        for i in range(images.size(0)):
            # Convert tensor back to numpy
            img_tensor = images[i]
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            
            # TTA prediction
            avg_probs, pred_class, _ = predictor.predict_with_tta(img_np, num_augments)
            
            all_probs.append(avg_probs)
            all_preds.append(pred_class)
            all_labels.append(labels[i].item())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)