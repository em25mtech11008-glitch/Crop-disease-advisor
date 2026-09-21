"""
EfficientNet-B4 Classifier for 38-class plant disease detection.
Prompt 1.4 — Model Definition
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple


class EfficientNetB4Classifier(nn.Module):
    """
    EfficientNet-B4 with a custom classification head, Grad-CAM hook support,
    and intermediate feature extraction.
    """

    def __init__(self, num_classes: int = 38, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        # Load pretrained backbone (timm)
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,          # Remove default head
            global_pool="avg",
        )
        in_features = self.backbone.num_features  # 1792 for EfficientNet-B4

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (batch, num_classes)."""
        features = self.backbone(x)          # (B, 1792) after global pool
        logits   = self.classifier(features) # (B, 38)
        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns intermediate spatial feature maps BEFORE global pooling.
        Shape: (B, C, H, W) — useful for Grad-CAM.
        """
        # Bypass global pool by calling forward_features directly on backbone
        return self.backbone.forward_features(x)

    def get_target_layer(self) -> nn.Module:
        """
        Returns the last MBConv block of EfficientNet-B4 for Grad-CAM.
        timm exposes blocks via backbone.blocks[-1][-1].
        """
        return self.backbone.blocks[-1][-1]


# ── Utility ───────────────────────────────────────────────────────────────────

def model_summary(model: EfficientNetB4Classifier, device: str = "cpu") -> None:
    """Prints total/trainable params, model size, and I/O shapes."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb   = total * 4 / (1024 ** 2)   # float32: 4 bytes/param

    print("=" * 50)
    print(f"  Model         : EfficientNet-B4")
    print(f"  Num classes   : {model.num_classes}")
    print(f"  Total params  : {total:,}")
    print(f"  Trainable     : {trainable:,}")
    print(f"  Model size    : {size_mb:.2f} MB")

    # I/O shape check
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    model_dev = model.to(device).eval()
    with torch.no_grad():
        out = model_dev(dummy)
    print(f"  Input shape   : {tuple(dummy.shape)}")
    print(f"  Output shape  : {tuple(out.shape)}")
    print("=" * 50)


if __name__ == "__main__":
    model = EfficientNetB4Classifier(num_classes=38, dropout=0.3)
    model_summary(model)
    print(f"\nTarget layer  : {model.get_target_layer()}")
