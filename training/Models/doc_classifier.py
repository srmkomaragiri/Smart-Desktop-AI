"""
doc_classifier.py — Document classification head
==================================================
Classifies document images into 16 RVL-CDIP categories using
the pooled backbone features. Simple FC head for VRAM efficiency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DocClassificationHead(nn.Module):
    """
    Document classification head for RVL-CDIP categories.

    Takes globally pooled backbone features and predicts one of
    16 document classes (letter, form, email, handwritten, etc.).
    """

    def __init__(
        self,
        in_channels: int = 1280,
        num_classes: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        pooled_features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        pooled_features : (B, C) — globally pooled backbone output
        targets         : (B,) — class indices (0-15) for training

        Returns
        -------
        Dict with 'logits', 'probs', and optionally 'loss'
        """
        logits = self.classifier(pooled_features)  # (B, num_classes)
        probs = F.softmax(logits, dim=-1)

        output = {
            "logits": logits,
            "probs": probs,
        }

        if targets is not None:
            loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
            output["loss"] = loss

        return output

    def predict(self, pooled_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference mode — returns predicted class and confidence."""
        with torch.no_grad():
            logits = self.classifier(pooled_features)
            probs = F.softmax(logits, dim=-1)
            confidence, predicted = probs.max(dim=-1)

        return {
            "predicted_class": predicted,
            "confidence": confidence,
            "probs": probs,
        }
