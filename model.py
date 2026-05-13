import torch
import torch.nn as nn
import timm
import config as cfg


class ConvNeXtDR(nn.Module):
    """Optimized ConvNeXt-Tiny for Diabetic Retinopathy"""

    def __init__(
        self,
        num_classes=cfg.NUM_CLASSES,
        pretrained=cfg.PRETRAINED,
        dropout=cfg.DROPOUT
    ):
        super().__init__()

        # FIXED: Use global_pool='avg' for stable pretrained behavior
        self.backbone = timm.create_model(
            cfg.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',  # Let timm handle pooling
            drop_path_rate=cfg.STOCHASTIC_DEPTH
        )
        
        feat_dim = 768  # ConvNeXt-Tiny

        # FIXED: Simplified head (768 -> 256 -> 5)
        # Removed manual pooling since backbone already pools
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),  # Reduced from 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.backbone(x)  # Already pooled: [B, 768]
        out = self.head(features)
        return out


def build_model():
    """Factory function"""
    model = ConvNeXtDR()
    model = model.to(cfg.DEVICE)
    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model()
    dummy_input = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(cfg.DEVICE)
    output = m(dummy_input)
    print(f"Model Output Shape: {output.shape}")
    print(f"Trainable Parameters: {count_parameters(m):,}")