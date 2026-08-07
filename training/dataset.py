"""
dataset.py — PyTorch Dataset and DataLoader for multi-task training
====================================================================
Wraps UnifiedSample objects into a torch Dataset with on-the-fly
image loading, preprocessing, and augmentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from PIL import Image

from training.schema import (
    TaskType,
    UnifiedSample,
    RVLCDIP_CLASSES,
)
from training.augmentation import get_augment_fn, compute_sample_weights

logger = logging.getLogger(__name__)


class UnifiedDataset(Dataset):
    """
    PyTorch Dataset that wraps UnifiedSample objects.

    Handles:
    - On-the-fly image loading and resizing
    - Task-specific augmentation
    - Consistent tensor output for multi-task training
    """

    def __init__(
        self,
        samples: List[UnifiedSample],
        transform: Optional[Callable] = None,
        augment: bool = False,
        detection_size: Tuple[int, int] = (640, 640),
        classification_size: Tuple[int, int] = (224, 224),
        normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.augment = augment
        self.detection_size = detection_size
        self.classification_size = classification_size
        self.normalize_mean = torch.tensor(normalize_mean).view(3, 1, 1)
        self.normalize_std = torch.tensor(normalize_std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # ── Load image ────────────────────────────────────────
        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to load image %s: %s", sample.image_path, exc)
            # Return a blank placeholder
            size = (
                self.classification_size
                if sample.task == TaskType.DOC_CLASSIFICATION
                else self.detection_size
            )
            image = Image.new("RGB", size, (128, 128, 128))

        # ── Augmentation (training only) ──────────────────────
        if self.augment and sample.augmented:
            aug_fn = get_augment_fn(sample.task)
            image = aug_fn(image)

        # ── Resize based on task ──────────────────────────────
        if sample.task == TaskType.DOC_CLASSIFICATION:
            target_size = self.classification_size
        else:
            target_size = self.detection_size

        image = image.resize(target_size, Image.LANCZOS)

        # ── Custom transform ──────────────────────────────────
        if self.transform:
            image = self.transform(image)

        # ── To tensor + normalize ─────────────────────────────
        img_tensor = torch.from_numpy(
            np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )
        img_tensor = (img_tensor - self.normalize_mean) / self.normalize_std

        # ── Build output dict ─────────────────────────────────
        output: Dict[str, Any] = {
            "image": img_tensor,
            "task": sample.task.value,
            "source": sample.source.value,
            "image_path": sample.image_path,
        }

        # Task-specific targets
        if sample.task in (TaskType.TEXT_DETECTION, TaskType.DESKTOP_TEXT):
            output["text_regions"] = self._encode_text_regions(sample)

        if sample.task == TaskType.DOC_CLASSIFICATION:
            output["doc_class_id"] = sample.doc_class_id or 0

        if sample.task == TaskType.FORM_UNDERSTANDING:
            output["form_entities"] = self._encode_form_entities(sample)

        return output

    def _encode_text_regions(self, sample: UnifiedSample) -> Dict[str, torch.Tensor]:
        """Encode text regions into fixed-size tensors."""
        max_regions = 100  # Pad/truncate to fixed count
        bboxes = torch.zeros(max_regions, 4)
        labels = torch.zeros(max_regions, dtype=torch.long)  # 0=background, 1=text
        mask = torch.zeros(max_regions, dtype=torch.bool)

        regions = sample.text_regions[:max_regions]
        for i, r in enumerate(regions):
            bboxes[i] = torch.tensor(r.bbox)
            labels[i] = 1 if r.is_legible else 0
            mask[i] = True

        return {
            "bboxes": bboxes,
            "labels": labels,
            "mask": mask,
            "num_regions": len(regions),
        }

    def _encode_form_entities(self, sample: UnifiedSample) -> Dict[str, torch.Tensor]:
        """Encode form entities into fixed-size tensors."""
        max_entities = 64
        bboxes = torch.zeros(max_entities, 4)
        labels = torch.zeros(max_entities, dtype=torch.long)
        mask = torch.zeros(max_entities, dtype=torch.bool)

        label_map = {"question": 0, "answer": 1, "header": 2, "other": 3}

        entities = sample.entities[:max_entities]
        for i, e in enumerate(entities):
            if len(e.bbox) == 4 and sample.image_width > 0 and sample.image_height > 0:
                norm_bbox = [
                    e.bbox[0] / sample.image_width,
                    e.bbox[1] / sample.image_height,
                    e.bbox[2] / sample.image_width,
                    e.bbox[3] / sample.image_height,
                ]
                bboxes[i] = torch.tensor(norm_bbox)
            labels[i] = label_map.get(e.label.value, 3)
            mask[i] = True

        return {
            "bboxes": bboxes,
            "labels": labels,
            "mask": mask,
            "num_entities": len(entities),
        }


def multi_task_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collation function for multi-task batches.
    Handles variable-sized annotations by grouping samples by task.
    """
    collated: Dict[str, Any] = {
        "image": torch.stack([b["image"] for b in batch]),
        "task": [b["task"] for b in batch],
        "source": [b["source"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }

    # Collate task-specific fields where present
    if any("doc_class_id" in b for b in batch):
        collated["doc_class_id"] = torch.tensor([
            b.get("doc_class_id", 0) for b in batch
        ], dtype=torch.long)

    if any("text_regions" in b for b in batch):
        tr_batch = [b["text_regions"] for b in batch if "text_regions" in b]
        if tr_batch:
            collated["text_regions"] = {
                "bboxes": torch.stack([t["bboxes"] for t in tr_batch]),
                "labels": torch.stack([t["labels"] for t in tr_batch]),
                "mask": torch.stack([t["mask"] for t in tr_batch]),
            }

    if any("form_entities" in b for b in batch):
        fe_batch = [b["form_entities"] for b in batch if "form_entities" in b]
        if fe_batch:
            collated["form_entities"] = {
                "bboxes": torch.stack([f["bboxes"] for f in fe_batch]),
                "labels": torch.stack([f["labels"] for f in fe_batch]),
                "mask": torch.stack([f["mask"] for f in fe_batch]),
            }

    return collated


def create_dataloaders(
    train_samples: List[UnifiedSample],
    val_samples: List[UnifiedSample],
    test_samples: List[UnifiedSample],
    batch_size: int = 8,
    num_workers: int = 2,
    task_weights: Optional[Dict[str, float]] = None,
    detection_size: Tuple[int, int] = (640, 640),
    classification_size: Tuple[int, int] = (224, 224),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders with weighted sampling for training.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    train_ds = UnifiedDataset(
        train_samples,
        augment=True,
        detection_size=detection_size,
        classification_size=classification_size,
    )
    val_ds = UnifiedDataset(
        val_samples,
        augment=False,
        detection_size=detection_size,
        classification_size=classification_size,
    )
    test_ds = UnifiedDataset(
        test_samples,
        augment=False,
        detection_size=detection_size,
        classification_size=classification_size,
    )

    # Weighted sampling for balanced training
    weights = compute_sample_weights(train_samples, task_weights)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_samples),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=multi_task_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multi_task_collate,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multi_task_collate,
        pin_memory=True,
    )

    logger.info(
        "DataLoaders created — Train: %d batches, Val: %d batches, Test: %d batches.",
        len(train_loader), len(val_loader), len(test_loader),
    )

    return train_loader, val_loader, test_loader
