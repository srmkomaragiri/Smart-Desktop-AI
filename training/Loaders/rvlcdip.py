"""
rvlcdip.py — RVL-CDIP Document Classification loader
=====================================================
Parses the RVL-CDIP label format:
  path/to/image.tif  category_index

400K grayscale TIFF images across 16 document categories.
Converts TIFF→RGB on load for consistency with the rest of the pipeline.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional

from PIL import Image

from training.schema import (
    DataSource,
    RVLCDIP_CLASSES,
    RVLCDIP_ID_TO_CLASS,
    SplitType,
    TaskType,
    UnifiedSample,
)

logger = logging.getLogger(__name__)


def _parse_label_file(label_path: Path) -> List[dict]:
    """
    Parse an RVL-CDIP label file.
    Each line: relative_image_path  class_index
    """
    entries = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.rsplit(None, 1)  # Split from right to handle spaces in paths
                if len(parts) != 2:
                    logger.debug("Skipping malformed line %d in %s", line_num, label_path)
                    continue

                img_rel, class_str = parts
                try:
                    class_id = int(class_str)
                except ValueError:
                    continue

                if class_id < 0 or class_id >= len(RVLCDIP_CLASSES):
                    logger.debug("Invalid class ID %d at line %d", class_id, line_num)
                    continue

                entries.append({
                    "image_rel": img_rel,
                    "class_id": class_id,
                    "class_name": RVLCDIP_ID_TO_CLASS[class_id],
                })
    except FileNotFoundError:
        logger.warning("Label file not found: %s", label_path)

    return entries


def load(
    root_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[UnifiedSample]:
    """
    Load RVL-CDIP dataset.

    Expected directory structure:
        root_dir/
        ├── images/
        │   ├── imagesa/
        │   │   ├── a/
        │   │   │   └── *.tif
        │   │   └── ...
        │   └── ...
        └── labels/
            ├── train.txt
            ├── val.txt
            └── test.txt

    Parameters
    ----------
    root_dir   : Path to RVL-CDIP root directory
    split      : "train", "val", or "test"
    max_samples: Limit number of samples (None = use all; recommended: 10000)

    Returns
    -------
    List of UnifiedSample objects with task=DOC_CLASSIFICATION
    """
    root = Path(root_dir)
    label_file = root / "labels" / f"{split}.txt"
    images_root = root / "images"

    if not label_file.exists():
        logger.warning("RVL-CDIP label file not found: %s", label_file)
        return []

    entries = _parse_label_file(label_file)
    logger.info("RVL-CDIP [%s]: parsed %d entries from label file.", split, len(entries))

    # Subsample if needed (stratified by class)
    if max_samples and len(entries) > max_samples:
        # Stratified sampling: proportional samples per class
        by_class: dict = {}
        for e in entries:
            by_class.setdefault(e["class_id"], []).append(e)

        per_class = max(1, max_samples // len(RVLCDIP_CLASSES))
        sampled = []
        rng = random.Random(42)  # Deterministic seed
        for cls_entries in by_class.values():
            rng.shuffle(cls_entries)
            sampled.extend(cls_entries[:per_class])

        # If we need a few more to reach max_samples, take from largest classes
        remaining = max_samples - len(sampled)
        if remaining > 0:
            leftover = [e for cls_list in by_class.values() for e in cls_list[per_class:]]
            rng.shuffle(leftover)
            sampled.extend(leftover[:remaining])

        entries = sampled
        logger.info("RVL-CDIP [%s]: subsampled to %d entries.", split, len(entries))

    # ── Build samples ─────────────────────────────────────────
    split_type = {
        "train": SplitType.TRAIN,
        "val": SplitType.VAL,
        "test": SplitType.TEST,
    }.get(split, SplitType.TRAIN)

    samples: List[UnifiedSample] = []

    for entry in entries:
        img_path = images_root / entry["image_rel"]

        if not img_path.exists():
            # Try without images/ prefix (some versions have flat structure)
            img_path = root / entry["image_rel"]
            if not img_path.exists():
                continue

        # Get image dimensions (don't load full image here for speed)
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception as exc:
            logger.debug("Cannot read %s: %s", img_path, exc)
            continue

        samples.append(UnifiedSample(
            image_path=str(img_path.resolve()),
            task=TaskType.DOC_CLASSIFICATION,
            source=DataSource.RVLCDIP,
            split=split_type,
            doc_class=entry["class_name"],
            doc_class_id=entry["class_id"],
            image_width=w,
            image_height=h,
            has_manual_annotations=True,
        ))

        if max_samples and len(samples) >= max_samples:
            break

    # Log class distribution
    from collections import Counter
    class_counts = Counter(s.doc_class for s in samples)
    logger.info(
        "RVL-CDIP [%s]: loaded %d samples across %d classes.",
        split, len(samples), len(class_counts),
    )
    for cls_name, count in sorted(class_counts.items()):
        logger.debug("  %s: %d", cls_name, count)

    return samples
