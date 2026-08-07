"""
cocotext.py — COCO-Text dataset loader
========================================
Parses the COCO-Text v2 JSON annotation format.
Images come from MSCOCO; annotations add text-specific fields:
  - Bounding boxes [x, y, width, height]
  - Legibility (legible / illegible)
  - Type (machine-printed / handwritten)
  - Language (English / non-English)
  - Transcription (for legible text)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from training.schema import (
    DataSource,
    SplitType,
    TaskType,
    TextRegion,
    UnifiedSample,
)

logger = logging.getLogger(__name__)


def _xywh_to_xyxy(bbox: List[float]) -> List[float]:
    """Convert [x, y, width, height] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _normalize_bbox(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    """Normalize pixel bbox to (0,1) scale."""
    if img_w <= 0 or img_h <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        max(0.0, min(1.0, bbox[0] / img_w)),
        max(0.0, min(1.0, bbox[1] / img_h)),
        max(0.0, min(1.0, bbox[2] / img_w)),
        max(0.0, min(1.0, bbox[3] / img_h)),
    ]


def load(
    root_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[UnifiedSample]:
    """
    Load COCO-Text dataset.

    Expected directory structure:
        root_dir/
        ├── images/
        │   ├── COCO_train2014_000000000009.jpg
        │   └── ...
        └── annotations/
            └── cocotext.v2.json

    Parameters
    ----------
    root_dir   : Path to COCO-Text root directory
    split      : "train", "val", or "test"
    max_samples: Limit number of samples (None = use all)

    Returns
    -------
    List of UnifiedSample objects with task=TEXT_DETECTION
    """
    root = Path(root_dir)
    ann_file = root / "annotations" / "cocotext.v2.json"
    images_dir = root / "images"

    if not ann_file.exists():
        logger.warning("COCO-Text annotation file not found: %s", ann_file)
        return []

    if not images_dir.exists():
        logger.warning("COCO-Text images dir not found: %s", images_dir)
        return []

    # ── Load annotations ──────────────────────────────────────
    logger.info("Loading COCO-Text annotations from %s ...", ann_file)
    with open(ann_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Map split name to COCO-Text set value
    split_map = {"train": "train", "val": "val", "test": "test"}
    target_set = split_map.get(split, "train")

    # Build image info lookup: img_id -> {file_name, width, height, set}
    imgs_info: Dict[str, dict] = {}
    for img_id, img_data in coco_data.get("imgs", {}).items():
        if img_data.get("set") == target_set:
            imgs_info[str(img_id)] = img_data

    # Group annotations by image ID
    anns_by_image: Dict[str, List[dict]] = {}
    for ann_id, ann_data in coco_data.get("anns", {}).items():
        img_id = str(ann_data.get("image_id", ""))
        if img_id in imgs_info:
            anns_by_image.setdefault(img_id, []).append(ann_data)

    # ── Build samples ─────────────────────────────────────────
    split_type = {
        "train": SplitType.TRAIN,
        "val": SplitType.VAL,
        "test": SplitType.TEST,
    }.get(split, SplitType.TRAIN)

    samples: List[UnifiedSample] = []

    for img_id, img_info in sorted(imgs_info.items()):
        file_name = img_info.get("file_name", "")
        img_path = images_dir / file_name

        if not img_path.exists():
            # Try alternate naming (COCO uses various naming conventions)
            alt_path = images_dir / f"COCO_train2014_{int(img_id):012d}.jpg"
            if alt_path.exists():
                img_path = alt_path
            else:
                continue

        img_w = img_info.get("width", 0)
        img_h = img_info.get("height", 0)

        # If dimensions not in JSON, read from image
        if img_w <= 0 or img_h <= 0:
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception:
                continue

        # Parse text regions for this image
        text_regions: List[TextRegion] = []
        annotations = anns_by_image.get(img_id, [])

        for ann in annotations:
            raw_bbox = ann.get("bbox", [0, 0, 0, 0])
            if len(raw_bbox) != 4:
                continue

            pixel_bbox = _xywh_to_xyxy(raw_bbox)
            norm_bbox = _normalize_bbox(pixel_bbox, img_w, img_h)

            # Attributes
            is_legible = ann.get("legibility", "legible") == "legible"
            language = "en" if ann.get("language", "english") == "english" else "other"
            transcription = ann.get("utf8_string", "") if is_legible else ""

            text_regions.append(TextRegion(
                bbox=norm_bbox,
                bbox_raw=pixel_bbox,
                transcription=transcription,
                is_legible=is_legible,
                language=language,
                confidence=1.0,
            ))

        # Build full OCR text
        ocr_text = " ".join(
            r.transcription for r in text_regions
            if r.is_legible and r.transcription
        )

        samples.append(UnifiedSample(
            image_path=str(img_path.resolve()),
            task=TaskType.TEXT_DETECTION,
            source=DataSource.COCOTEXT,
            split=split_type,
            text_regions=text_regions,
            ocr_text=ocr_text or None,
            image_width=img_w,
            image_height=img_h,
            has_manual_annotations=True,
        ))

        if max_samples and len(samples) >= max_samples:
            break

    logger.info(
        "COCO-Text [%s]: loaded %d samples with %d total text regions.",
        split, len(samples),
        sum(len(s.text_regions) for s in samples),
    )
    return samples
