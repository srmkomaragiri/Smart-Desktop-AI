"""
text_detector.py — Text detection head
========================================
Predicts text region bounding boxes and text/no-text classification
from backbone features. Used for ICDAR 2015, COCO-Text, and
Custom Desktop text detection tasks.

Architecture: Simple feature pyramid + prediction head, designed
for efficiency on 4GB VRAM GPUs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TextDetectionHead(nn.Module):
    """
    Lightweight text detection head.

    Takes pooled backbone features and predicts:
    - Region confidence scores (text vs. background)
    - Bounding box regressions (normalized [x1, y1, x2, y2])

    Uses a simple convolutional approach rather than a full anchor-based
    detector to stay within VRAM budget.
    """

    def __init__(
        self,
        in_channels: int = 1280,
        hidden_dim: int = 256,
        max_detections: int = 100,
        num_classes: int = 2,  # text / no-text
    ) -> None:
        super().__init__()
        self.max_detections = max_detections
        self.num_classes = num_classes

        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Confidence prediction (per-pixel text probability)
        self.conf_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_classes, 1),
        )

        # Bbox regression (per-pixel offset prediction)
        self.bbox_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 4, 1),  # [dx1, dy1, dx2, dy2]
            nn.Sigmoid(),  # Ensure outputs are in [0, 1]
        )

    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        features : (B, C, H, W) — last feature map from backbone
        targets  : Optional dict with 'bboxes', 'labels', 'mask' for training

        Returns
        -------
        Dict with 'conf_logits', 'bbox_pred', and optionally 'loss'
        """
        refined = self.refine(features)

        # Predictions
        conf_logits = self.conf_head(refined)   # (B, 2, H, W)
        bbox_pred = self.bbox_head(refined)     # (B, 4, H, W)

        output = {
            "conf_logits": conf_logits,
            "bbox_pred": bbox_pred,
        }

        # Compute loss if targets provided
        if targets is not None:
            loss = self._compute_loss(conf_logits, bbox_pred, targets)
            output["loss"] = loss

        return output

    def _compute_loss(
        self,
        conf_logits: torch.Tensor,
        bbox_pred: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute combined confidence + bbox regression loss.

        Uses:
        - Cross-entropy for text/no-text classification
        - Smooth L1 for bbox regression on positive samples
        """
        B = conf_logits.shape[0]
        device = conf_logits.device

        target_bboxes = targets["bboxes"]   # (B, N, 4)
        target_labels = targets["labels"]   # (B, N)
        target_mask = targets["mask"]       # (B, N)

        # Global average pool the spatial predictions to match target shape
        conf_pooled = F.adaptive_avg_pool2d(conf_logits, 1).squeeze(-1).squeeze(-1)  # (B, 2)
        bbox_pooled = F.adaptive_avg_pool2d(bbox_pred, 1).squeeze(-1).squeeze(-1)    # (B, 4)

        # Binary classification: does this image contain text?
        has_text = (target_mask.any(dim=1)).long()  # (B,)
        conf_loss = F.cross_entropy(conf_pooled, has_text)

        # Bbox regression: average target bbox for simple regression
        # (For full detection, use anchor-based matching — this is a lightweight version)
        bbox_loss = torch.tensor(0.0, device=device)
        for b in range(B):
            valid = target_mask[b]
            if valid.any():
                valid_bboxes = target_bboxes[b][valid]
                # Regress toward the mean bbox as a proxy
                mean_bbox = valid_bboxes.mean(dim=0)
                bbox_loss = bbox_loss + F.smooth_l1_loss(bbox_pooled[b], mean_bbox)

        bbox_loss = bbox_loss / max(1, B)

        return conf_loss + bbox_loss
