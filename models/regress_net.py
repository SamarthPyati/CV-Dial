"""Model C (S3): ResNet-18 regressing f directly from the 384 px dial crop (PLAN §4).

Input frame: the crop from geometry.crop, RGB uint8 (H, W, 3). Output: f, unbounded (no sigmoid).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class RegressNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)


def to_tensor(crop_rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 RGB -> (3, H, W) float32, ImageNet-normalised."""
    x = (crop_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def save_checkpoint(path: Path, model: RegressNet, cfg: dict, **extra) -> None:
    torch.save({"state_dict": model.state_dict(), "config": cfg, **extra}, path)


def load_regressor(path: str | Path, device: str = "cpu") -> RegressNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = RegressNet(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def predict(model: RegressNet, crops_rgb: list[np.ndarray], device: str = "cpu") -> np.ndarray:
    x = torch.stack([to_tensor(c) for c in crops_rgb]).to(device)
    return model(x).float().cpu().numpy().astype(np.float64)
