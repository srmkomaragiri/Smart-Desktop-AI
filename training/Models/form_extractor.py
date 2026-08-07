"""
form_extractor.py — Form understanding / entity extraction head
================================================================
Classifies form entities (question, answer, header, other) and
predicts entity bounding boxes from backbone features.
Used for the FUNSD dataset task.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FormExtractionHead(nn.Module):
    """
    Form entity extraction head for FUNSD-style form understanding.

    Tasks:
    1. Entity classification: question / answer / header / other
    2. Entity bbox regression
    """

    def __init__(
        self,
        in_channels: int = 1280,
        hidden_dim: int = 256,
        num_labels: int = 4,  # question, answer, header, other
        max_entities: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.max_entities = max_entities

        # Feature refinement from spatial features
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Entity label prediction from spatial maps
        self.label_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_labels, 1),
        )

        # Entity bbox regression
        self.bbox_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 4, 1),
            nn.Sigmoid(),
        )

        # Global entity count predictor (from pooled features)
        self.count_head = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, max_entities),
        )

    def forward(
        self,
        features: torch.Tensor,
        pooled: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        features : (B, C, H, W) — spatial features from backbone
        pooled   : (B, C) — globally pooled features
        targets  : Optional dict with 'bboxes', 'labels', 'mask'

        Returns
        -------
        Dict with 'label_logits', 'bbox_pred', and optionally 'loss'
        """
        refined = self.spatial_refine(features)

        label_logits = self.label_head(refined)   # (B, num_labels, H, W)
        bbox_pred = self.bbox_head(refined)       # (B, 4, H, W)

        output = {
            "label_logits": label_logits,
            "bbox_pred": bbox_pred,
        }

        if targets is not None:
            loss = self._compute_loss(label_logits, bbox_pred, targets)
            output["loss"] = loss

        return output

    def _compute_loss(
        self,
        label_logits: torch.Tensor,
        bbox_pred: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute entity classification + bbox regression loss.
        """
        B = label_logits.shape[0]
        device = label_logits.device

        target_bboxes = targets["bboxes"]   # (B, N, 4)
        target_labels = targets["labels"]   # (B, N)
        target_mask = targets["mask"]       # (B, N)

        # Pool spatial predictions
        label_pooled = F.adaptive_avg_pool2d(label_logits, 1).squeeze(-1).squeeze(-1)  # (B, 4)
        bbox_pooled = F.adaptive_avg_pool2d(bbox_pred, 1).squeeze(-1).squeeze(-1)      # (B, 4)

        # Classification loss: most common entity label in the form
        cls_loss = torch.tensor(0.0, device=device)
        bbox_loss = torch.tensor(0.0, device=device)
        n_valid = 0

        for b in range(B):
            valid = target_mask[b]
            if not valid.any():
                continue

            n_valid += 1
            valid_labels = target_labels[b][valid]
            valid_bboxes = target_bboxes[b][valid]

            # Use the most frequent label as the target
            mode_label = valid_labels.mode().values
            cls_loss = cls_loss + F.cross_entropy(
                label_pooled[b].unsqueeze(0), mode_label.unsqueeze(0)
            )

            # Bbox loss: average entity bbox
            mean_bbox = valid_bboxes.mean(dim=0)
            bbox_loss = bbox_loss + F.smooth_l1_loss(bbox_pooled[b], mean_bbox)

        if n_valid > 0:
            cls_loss = cls_loss / n_valid
            bbox_loss = bbox_loss / n_valid

        return cls_loss + 0.5 * bbox_loss
