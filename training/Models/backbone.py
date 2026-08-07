"""
backbone.py — Shared CNN backbone for multi-task model
=======================================================
Provides feature extraction using EfficientNet-B0 (default) or
ResNet-18 for extreme VRAM efficiency on 4GB GPUs.

The backbone produces multi-scale features used by all task heads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BackboneOutput:
    """Multi-scale feature maps from the backbone."""
    features: List[torch.Tensor]   # Multi-scale feature maps
    pooled: torch.Tensor           # Global average pooled features (B, C)


class EfficientNetB0Backbone(nn.Module):
    """
    EfficientNet-B0 backbone with multi-scale feature extraction.
    ~5.3M params, ~0.39 GFLOPs — ideal for 4GB VRAM.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = efficientnet_b0(weights=weights)
        except ImportError:
            from torchvision.models import efficientnet_b0
            base = efficientnet_b0(pretrained=pretrained)

        # Extract feature stages from EfficientNet
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Feature dimensions at each scale for EfficientNet-B0
        # After stage 1: 16ch, stage 2: 24ch, stage 3: 40ch,
        # stage 5: 112ch, stage 8: 1280ch
        self.feature_dims = [16, 24, 40, 112, 1280]
        self.out_channels = 1280

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """
        Extract multi-scale features.

        Parameters
        ----------
        x : (B, 3, H, W) input image tensor

        Returns
        -------
        BackboneOutput with features at multiple scales
        """
        multi_scale = []

        for i, layer in enumerate(self.features):
            x = layer(x)
            # Capture features at key scales
            if i in (1, 2, 3, 5, 8):
                multi_scale.append(x)

        # Capture final features if we didn't get the last one
        if len(multi_scale) < 5:
            multi_scale.append(x)

        pooled = self.pool(x).flatten(1)  # (B, 1280)

        return BackboneOutput(
            features=multi_scale,
            pooled=pooled,
        )


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 backbone — even lighter alternative.
    ~11.7M params but very fast inference.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
        except ImportError:
            from torchvision.models import resnet18
            base = resnet18(pretrained=pretrained)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1  # 64ch
        self.layer2 = base.layer2  # 128ch
        self.layer3 = base.layer3  # 256ch
        self.layer4 = base.layer4  # 512ch
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.feature_dims = [64, 64, 128, 256, 512]
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        multi_scale = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        multi_scale.append(x)  # Scale 0: stride 2

        x = self.maxpool(x)
        x = self.layer1(x)
        multi_scale.append(x)  # Scale 1: stride 4

        x = self.layer2(x)
        multi_scale.append(x)  # Scale 2: stride 8

        x = self.layer3(x)
        multi_scale.append(x)  # Scale 3: stride 16

        x = self.layer4(x)
        multi_scale.append(x)  # Scale 4: stride 32

        pooled = self.pool(x).flatten(1)  # (B, 512)

        return BackboneOutput(
            features=multi_scale,
            pooled=pooled,
        )


# ── Factory ───────────────────────────────────────────────────

_BACKBONES = {
    "efficientnet_b0": EfficientNetB0Backbone,
    "resnet18": ResNet18Backbone,
}


def create_backbone(
    name: str = "efficientnet_b0",
    pretrained: bool = True,
) -> nn.Module:
    """
    Create a backbone network by name.

    Parameters
    ----------
    name       : "efficientnet_b0" or "resnet18"
    pretrained : Use ImageNet pretrained weights

    Returns
    -------
    Backbone module with .out_channels and .feature_dims attributes
    """
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone: {name}. Choose from: {list(_BACKBONES)}")

    backbone = _BACKBONES[name](pretrained=pretrained)
    n_params = sum(p.numel() for p in backbone.parameters())
    logger.info(
        "Created backbone '%s' — %.2fM params, out_channels=%d",
        name, n_params / 1e6, backbone.out_channels,
    )
    return backbone
