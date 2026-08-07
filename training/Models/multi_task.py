"""
multi_task.py — Multi-task model combining all heads
=====================================================
Shared EfficientNet-B0 backbone with task-specific heads:
  - Text Detection (ICDAR 2015, COCO-Text, Custom Desktop)
  - Document Classification (RVLCDIP)
  - Form Entity Extraction (FUNSD)

The model routes each sample to the appropriate head based on
its task type, computing task-weighted loss for balanced training.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from training.schema import TaskType
from training.models.backbone import create_backbone, BackboneOutput
from training.models.text_detector import TextDetectionHead
from training.models.doc_classifier import DocClassificationHead
from training.models.form_extractor import FormExtractionHead

logger = logging.getLogger(__name__)


class MultiTaskModel(nn.Module):
    """
    Multi-task model with shared backbone and task-specific heads.

    Architecture:
    ┌──────────────────────┐
    │   Input Image (B,3,H,W)   │
    └──────────┬───────────┘
               │
    ┌──────────▼───────────┐
    │   Shared Backbone       │  (EfficientNet-B0 / ResNet-18)
    │   → multi-scale feats   │
    │   → pooled features     │
    └──────────┬───────────┘
               │
       ┌───────┼───────┐
       │       │       │
    ┌──▼──┐ ┌──▼──┐ ┌──▼──┐
    │ Text │ │ Doc │ │Form │
    │ Det  │ │ Cls │ │ Ext │
    └──────┘ └─────┘ └─────┘
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_doc_classes: int = 16,
        num_form_labels: int = 4,
        max_detections: int = 100,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()

        # Shared backbone
        self.backbone = create_backbone(backbone_name, pretrained=pretrained)
        feat_dim = self.backbone.out_channels

        # Task-specific heads
        self.text_detector = TextDetectionHead(
            in_channels=feat_dim,
            hidden_dim=min(256, feat_dim),
            max_detections=max_detections,
        )
        self.doc_classifier = DocClassificationHead(
            in_channels=feat_dim,
            num_classes=num_doc_classes,
        )
        self.form_extractor = FormExtractionHead(
            in_channels=feat_dim,
            hidden_dim=min(256, feat_dim),
            num_labels=num_form_labels,
        )

        # Task loss weights
        self.task_weights = task_weights or {
            "text_detection": 1.0,
            "doc_classification": 1.0,
            "form_understanding": 1.0,
            "desktop_text": 1.5,
        }

        n_params = sum(p.numel() for p in self.parameters())
        logger.info("MultiTaskModel created — %.2fM total params", n_params / 1e6)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass — routes each sample to its task head.

        Parameters
        ----------
        batch : Dict from multi_task_collate containing:
            - 'image': (B, 3, H, W)
            - 'task': List[str] of task names
            - 'doc_class_id': (B,) for classification samples
            - 'text_regions': Dict for detection samples
            - 'form_entities': Dict for form samples

        Returns
        -------
        Dict with 'loss' (scalar), per-task losses, and predictions
        """
        images = batch["image"]
        tasks = batch["task"]

        # ── Backbone forward pass (shared) ────────────────────
        backbone_out: BackboneOutput = self.backbone(images)

        # ── Route to task heads ───────────────────────────────
        output: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=images.device)
        loss_count = 0

        # Group indices by task
        task_indices: Dict[str, List[int]] = {}
        for i, task in enumerate(tasks):
            task_indices.setdefault(task, []).append(i)

        # ── Text Detection (ICDAR, COCO-Text) ────────────────
        det_idx = task_indices.get(TaskType.TEXT_DETECTION.value, [])
        if det_idx:
            idx_t = torch.tensor(det_idx, device=images.device)
            det_features = backbone_out.features[-1][idx_t]
            det_targets = None

            if "text_regions" in batch:
                det_targets = {
                    "bboxes": batch["text_regions"]["bboxes"][: len(det_idx)],
                    "labels": batch["text_regions"]["labels"][: len(det_idx)],
                    "mask": batch["text_regions"]["mask"][: len(det_idx)],
                }

            det_out = self.text_detector(det_features, det_targets)
            output["text_detection"] = det_out

            if "loss" in det_out:
                w = self.task_weights.get("text_detection", 1.0)
                total_loss = total_loss + w * det_out["loss"]
                loss_count += 1

        # ── Desktop Text (Custom Dataset — uses text detector) ─
        desk_idx = task_indices.get(TaskType.DESKTOP_TEXT.value, [])
        if desk_idx:
            idx_t = torch.tensor(desk_idx, device=images.device)
            desk_features = backbone_out.features[-1][idx_t]
            desk_targets = None

            if "text_regions" in batch:
                # desktop text regions come from the same collation
                n_det = len(det_idx)
                n_desk = len(desk_idx)
                if batch["text_regions"]["bboxes"].shape[0] >= n_det + n_desk:
                    desk_targets = {
                        "bboxes": batch["text_regions"]["bboxes"][n_det: n_det + n_desk],
                        "labels": batch["text_regions"]["labels"][n_det: n_det + n_desk],
                        "mask": batch["text_regions"]["mask"][n_det: n_det + n_desk],
                    }

            desk_out = self.text_detector(desk_features, desk_targets)
            output["desktop_text"] = desk_out

            if "loss" in desk_out:
                w = self.task_weights.get("desktop_text", 1.5)
                total_loss = total_loss + w * desk_out["loss"]
                loss_count += 1

        # ── Document Classification (RVLCDIP) ────────────────
        cls_idx = task_indices.get(TaskType.DOC_CLASSIFICATION.value, [])
        if cls_idx:
            idx_t = torch.tensor(cls_idx, device=images.device)
            cls_pooled = backbone_out.pooled[idx_t]
            cls_targets = None

            if "doc_class_id" in batch:
                cls_targets = batch["doc_class_id"][idx_t]

            cls_out = self.doc_classifier(cls_pooled, cls_targets)
            output["doc_classification"] = cls_out

            if "loss" in cls_out:
                w = self.task_weights.get("doc_classification", 1.0)
                total_loss = total_loss + w * cls_out["loss"]
                loss_count += 1

        # ── Form Understanding (FUNSD) ────────────────────────
        form_idx = task_indices.get(TaskType.FORM_UNDERSTANDING.value, [])
        if form_idx:
            idx_t = torch.tensor(form_idx, device=images.device)
            form_features = backbone_out.features[-1][idx_t]
            form_pooled = backbone_out.pooled[idx_t]
            form_targets = None

            if "form_entities" in batch:
                form_targets = {
                    "bboxes": batch["form_entities"]["bboxes"][: len(form_idx)],
                    "labels": batch["form_entities"]["labels"][: len(form_idx)],
                    "mask": batch["form_entities"]["mask"][: len(form_idx)],
                }

            form_out = self.form_extractor(form_features, form_pooled, form_targets)
            output["form_understanding"] = form_out

            if "loss" in form_out:
                w = self.task_weights.get("form_understanding", 1.0)
                total_loss = total_loss + w * form_out["loss"]
                loss_count += 1

        # ── Total loss ────────────────────────────────────────
        if loss_count > 0:
            output["loss"] = total_loss / loss_count
        else:
            output["loss"] = total_loss

        return output

    def export_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 640, 640),
        opset_version: int = 14,
    ) -> None:
        """
        Export the backbone + individual heads to ONNX format.
        Exports the backbone separately for maximum portability.
        """
        self.eval()
        device = next(self.parameters()).device
        dummy_input = torch.randn(*input_shape, device=device)

        # Export backbone
        import os
        base_dir = os.path.dirname(output_path)
        os.makedirs(base_dir, exist_ok=True)

        # Save full model state dict (PyTorch) as a companion
        torch.save(self.state_dict(), output_path.replace(".onnx", ".pt"))

        # Export backbone to ONNX
        backbone_path = output_path.replace(".onnx", "_backbone.onnx")
        torch.onnx.export(
            self.backbone,
            dummy_input,
            backbone_path,
            opset_version=opset_version,
            input_names=["image"],
            output_names=["features", "pooled"],
            dynamic_axes={
                "image": {0: "batch_size"},
            },
        )
        logger.info("Exported backbone to: %s", backbone_path)
        logger.info("Saved full checkpoint to: %s", output_path.replace(".onnx", ".pt"))
