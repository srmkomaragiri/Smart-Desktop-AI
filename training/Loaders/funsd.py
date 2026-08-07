"""
funsd.py — FUNSD (Form Understanding in Noisy Scanned Documents) loader
========================================================================
Parses FUNSD JSON annotations containing:
  - Semantic entities (question, answer, header, other)
  - Entity bounding boxes and word-level data
  - Entity linking (key-value relationships)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from PIL import Image

from training.schema import (
    DataSource,
    FormEntity,
    FormEntityLabel,
    SplitType,
    TaskType,
    TextRegion,
    UnifiedSample,
)

logger = logging.getLogger(__name__)

# Map FUNSD label strings to our FormEntityLabel enum
_LABEL_MAP = {
    "question": FormEntityLabel.QUESTION,
    "answer": FormEntityLabel.ANSWER,
    "header": FormEntityLabel.HEADER,
    "other": FormEntityLabel.OTHER,
}


def _parse_document_json(json_path: Path, img_path: Path) -> Optional[UnifiedSample]:
    """
    Parse a single FUNSD document annotation JSON.

    Expected JSON structure:
    {
        "form": [
            {
                "id": 0,
                "text": "COMMERCIAL CABLE COMPANY",
                "label": "header",
                "box": [left, top, right, bottom],
                "words": [{"text": "COMMERCIAL", "box": [l, t, r, b]}, ...],
                "linking": [[0, 1], ...]
            },
            ...
        ]
    }
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", json_path, exc)
        return None

    form_data = data.get("form", [])
    if not form_data:
        logger.debug("Empty form data in %s", json_path)
        return None

    # Get image dimensions
    try:
        with Image.open(img_path) as img:
            w, h = img.size
    except Exception as exc:
        logger.warning("Cannot open image %s: %s", img_path, exc)
        return None

    # Parse entities
    entities: List[FormEntity] = []
    text_regions: List[TextRegion] = []

    for item in form_data:
        entity_id = item.get("id", -1)
        label_str = item.get("label", "other").lower()
        label = _LABEL_MAP.get(label_str, FormEntityLabel.OTHER)
        text = item.get("text", "")
        bbox = item.get("box", [0, 0, 0, 0])
        words = item.get("words", [])
        linking = item.get("linking", [])

        entities.append(FormEntity(
            id=entity_id,
            label=label,
            text=text,
            bbox=bbox,
            words=words,
            linking=linking,
        ))

        # Also create TextRegion for text detection compatibility
        if bbox and len(bbox) == 4 and w > 0 and h > 0:
            norm_bbox = [
                max(0.0, min(1.0, bbox[0] / w)),
                max(0.0, min(1.0, bbox[1] / h)),
                max(0.0, min(1.0, bbox[2] / w)),
                max(0.0, min(1.0, bbox[3] / h)),
            ]
            text_regions.append(TextRegion(
                bbox=norm_bbox,
                bbox_raw=[float(x) for x in bbox],
                transcription=text,
                is_legible=bool(text.strip()),
                confidence=1.0,
            ))

    # Build full text
    full_text = " ".join(e.text for e in entities if e.text.strip())

    return UnifiedSample(
        image_path=str(img_path.resolve()),
        task=TaskType.FORM_UNDERSTANDING,
        source=DataSource.FUNSD,
        text_regions=text_regions,
        entities=entities,
        ocr_text=full_text or None,
        image_width=w,
        image_height=h,
        has_manual_annotations=True,
    )


def load(
    root_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[UnifiedSample]:
    """
    Load FUNSD dataset.

    Expected directory structure:
        root_dir/
        ├── training_data/
        │   ├── annotations/
        │   │   ├── 0000971160.json
        │   │   └── ...
        │   └── images/
        │       ├── 0000971160.png
        │       └── ...
        └── testing_data/
            ├── annotations/
            └── images/

    Parameters
    ----------
    root_dir   : Path to FUNSD root directory
    split      : "train" or "test"
    max_samples: Limit number of samples (None = use all)

    Returns
    -------
    List of UnifiedSample objects with task=FORM_UNDERSTANDING
    """
    root = Path(root_dir)

    # FUNSD uses training_data / testing_data directory names
    if split in ("train", "val"):
        data_dir = root / "training_data"
    else:
        data_dir = root / "testing_data"

    ann_dir = data_dir / "annotations"
    img_dir = data_dir / "images"

    if not ann_dir.exists():
        logger.warning("FUNSD annotations dir not found: %s", ann_dir)
        return []

    if not img_dir.exists():
        logger.warning("FUNSD images dir not found: %s", img_dir)
        return []

    split_type = {
        "train": SplitType.TRAIN,
        "val": SplitType.VAL,
        "test": SplitType.TEST,
    }.get(split, SplitType.TRAIN)

    samples: List[UnifiedSample] = []

    for json_file in sorted(ann_dir.glob("*.json")):
        # Find corresponding image
        img_stem = json_file.stem
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif"):
            candidate = img_dir / f"{img_stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            logger.debug("No image found for annotation: %s", json_file.name)
            continue

        sample = _parse_document_json(json_file, img_path)
        if sample is not None:
            sample.split = split_type
            samples.append(sample)

        if max_samples and len(samples) >= max_samples:
            break

    # Log entity distribution
    from collections import Counter
    entity_labels = Counter(
        e.label.value for s in samples for e in s.entities
    )
    logger.info(
        "FUNSD [%s]: loaded %d documents with %d total entities.",
        split, len(samples),
        sum(len(s.entities) for s in samples),
    )
    for label, count in sorted(entity_labels.items()):
        logger.debug("  %s: %d", label, count)

    return samples
