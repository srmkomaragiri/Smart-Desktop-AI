"""
icdar2015.py — ICDAR 2015 Incidental Scene Text loader
=======================================================
Parses the ICDAR 2015 Challenge 4 format:
  - Images: `img_*.jpg`
  - Annotations: `gt_img_*.txt` with lines of:
    x1,y1,x2,y2,x3,y3,x4,y4,transcription

Quadrilateral bounding boxes are converted to axis-aligned [x1,y1,x2,y2]
and normalized to (0,1) relative to image dimensions.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional

from PIL import Image

from training.schema import (
    DataSource,
    SplitType,
    TaskType,
    TextRegion,
    UnifiedSample,
)

logger = logging.getLogger(__name__)


def _parse_annotation_line(line: str) -> Optional[dict]:
    """
    Parse a single annotation line from ICDAR 2015 GT file.
    Format: x1,y1,x2,y2,x3,y3,x4,y4,transcription
    The transcription may contain commas, so we split carefully.
    """
    line = line.strip()
    if not line:
        return None

    # Handle BOM in UTF-8-BOM files
    line = line.lstrip("\ufeff")

    # The first 8 values are coordinates, everything after is transcription
    parts = line.split(",")
    if len(parts) < 9:
        return None

    try:
        coords = [int(p.strip()) for p in parts[:8]]
    except ValueError:
        logger.warning("Invalid coordinates in line: %s", line[:80])
        return None

    transcription = ",".join(parts[8:]).strip()

    # "###" means illegible / don't-care
    is_legible = transcription != "###"

    return {
        "coords": coords,  # [x1,y1, x2,y2, x3,y3, x4,y4]
        "transcription": transcription if is_legible else "",
        "is_legible": is_legible,
    }


def _quad_to_bbox(coords: List[int]) -> List[float]:
    """
    Convert quadrilateral (8 values) to axis-aligned bounding box.
    Returns [min_x, min_y, max_x, max_y] in pixel coordinates.
    """
    xs = coords[0::2]  # x1, x2, x3, x4
    ys = coords[1::2]  # y1, y2, y3, y4
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _normalize_bbox(bbox: List[float], w: int, h: int) -> List[float]:
    """Normalize pixel bbox to (0,1) scale."""
    if w <= 0 or h <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        max(0.0, min(1.0, bbox[0] / w)),
        max(0.0, min(1.0, bbox[1] / h)),
        max(0.0, min(1.0, bbox[2] / w)),
        max(0.0, min(1.0, bbox[3] / h)),
    ]


def load(
    root_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[UnifiedSample]:
    """
    Load ICDAR 2015 dataset.

    Expected directory structure:
        root_dir/
        ├── train/
        │   ├── images/
        │   │   ├── img_1.jpg
        │   │   └── ...
        │   └── gt/
        │       ├── gt_img_1.txt
        │       └── ...
        └── test/
            ├── images/
            └── gt/

    Parameters
    ----------
    root_dir   : Path to ICDAR 2015 root directory
    split      : "train" or "test"
    max_samples: Limit number of samples (None = use all)

    Returns
    -------
    List of UnifiedSample objects with task=TEXT_DETECTION
    """
    root = Path(root_dir)
    split_dir = root / split

    images_dir = split_dir / "images"
    gt_dir = split_dir / "gt"

    if not images_dir.exists():
        logger.warning("ICDAR 2015 images dir not found: %s", images_dir)
        return []

    if not gt_dir.exists():
        logger.warning("ICDAR 2015 GT dir not found: %s", gt_dir)
        return []

    # Map: image number -> image path
    image_files = {}
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        for fp in images_dir.glob(ext):
            # Extract number from filename like img_1.jpg, img_23.jpg
            name = fp.stem
            num = "".join(c for c in name if c.isdigit())
            if num:
                image_files[num] = fp

    samples: List[UnifiedSample] = []
    split_type = SplitType.TRAIN if split == "train" else SplitType.TEST

    for gt_file in sorted(gt_dir.glob("gt_*.txt")):
        # Extract number from gt_img_1.txt
        gt_name = gt_file.stem
        num = "".join(c for c in gt_name if c.isdigit())

        if num not in image_files:
            logger.debug("No image for GT file: %s", gt_file.name)
            continue

        img_path = image_files[num]

        # Get image dimensions
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception as exc:
            logger.warning("Cannot open image %s: %s", img_path, exc)
            continue

        # Parse annotations
        text_regions: List[TextRegion] = []
        try:
            with open(gt_file, "r", encoding="utf-8-sig") as f:
                for line in f:
                    parsed = _parse_annotation_line(line)
                    if parsed is None:
                        continue

                    pixel_bbox = _quad_to_bbox(parsed["coords"])
                    norm_bbox = _normalize_bbox(pixel_bbox, w, h)

                    text_regions.append(TextRegion(
                        bbox=norm_bbox,
                        bbox_raw=pixel_bbox,
                        transcription=parsed["transcription"],
                        is_legible=parsed["is_legible"],
                        confidence=1.0,
                        language="en",
                    ))
        except Exception as exc:
            logger.warning("Failed to parse GT file %s: %s", gt_file, exc)
            continue

        # Build full OCR text from legible regions
        ocr_text = " ".join(
            r.transcription for r in text_regions
            if r.is_legible and r.transcription
        )

        samples.append(UnifiedSample(
            image_path=str(img_path.resolve()),
            task=TaskType.TEXT_DETECTION,
            source=DataSource.ICDAR2015,
            split=split_type,
            text_regions=text_regions,
            ocr_text=ocr_text or None,
            image_width=w,
            image_height=h,
            has_manual_annotations=True,
        ))

        if max_samples and len(samples) >= max_samples:
            break

    logger.info(
        "ICDAR 2015 [%s]: loaded %d samples with %d total text regions.",
        split, len(samples),
        sum(len(s.text_regions) for s in samples),
    )
    return samples
