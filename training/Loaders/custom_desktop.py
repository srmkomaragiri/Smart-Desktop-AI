"""
custom_desktop.py — Custom Desktop Screenshots loader (private dataset)
=======================================================================
Auto-generates annotations from desktop screenshots using Tesseract OCR
with word-level bounding box extraction (image_to_data).

This loader is critical for the strict requirement that both public
and private datasets are included in training.

The pipeline is designed to accept manual annotations when available:
  - If a .json annotation file exists alongside the image, it's used.
  - Otherwise, Tesseract auto-generates pseudo-annotations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter

from training.schema import (
    DataSource,
    SplitType,
    TaskType,
    TextRegion,
    UnifiedSample,
)

logger = logging.getLogger(__name__)

# Try importing Tesseract
try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Preprocess image for better Tesseract accuracy."""
    gray = image.convert("L")
    enhanced = ImageEnhance.Contrast(gray).enhance(1.5)
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    return sharpened


def _extract_word_regions(
    image: Image.Image,
    tesseract_cmd: Optional[str] = None,
    confidence_threshold: int = 30,
) -> Tuple[List[TextRegion], str, float]:
    """
    Run Tesseract with word-level bounding box extraction.

    Returns
    -------
    (text_regions, full_text, avg_confidence)
    """
    if not _TESSERACT_AVAILABLE:
        logger.warning("pytesseract not installed — cannot auto-annotate.")
        return [], "", 0.0

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    processed = _preprocess_for_ocr(image)
    w, h = image.size

    try:
        data = pytesseract.image_to_data(
            processed,
            config="--psm 3 --oem 3",
            lang="eng",
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:
        logger.warning("Tesseract failed: %s", exc)
        return [], "", 0.0

    text_regions: List[TextRegion] = []
    all_words: List[str] = []
    confidences: List[float] = []

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = int(data["conf"][i]) if str(data["conf"][i]) != "-1" else -1

        if not word or conf < confidence_threshold:
            continue

        # Pixel bounding box
        left = data["left"][i]
        top = data["top"][i]
        width = data["width"][i]
        height = data["height"][i]

        pixel_bbox = [float(left), float(top), float(left + width), float(top + height)]

        # Normalized bbox
        norm_bbox = [
            max(0.0, min(1.0, left / w)),
            max(0.0, min(1.0, top / h)),
            max(0.0, min(1.0, (left + width) / w)),
            max(0.0, min(1.0, (top + height) / h)),
        ]

        text_regions.append(TextRegion(
            bbox=norm_bbox,
            bbox_raw=pixel_bbox,
            transcription=word,
            confidence=conf / 100.0,
            is_legible=True,
            language="en",
        ))

        all_words.append(word)
        confidences.append(conf)

    full_text = " ".join(all_words)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return text_regions, full_text, avg_conf


def _load_manual_annotations(
    json_path: Path, img_w: int, img_h: int,
) -> Optional[Tuple[List[TextRegion], str]]:
    """
    Load manual annotations from a JSON file if it exists.

    Expected JSON format:
    {
        "text_regions": [
            {
                "bbox": [x1, y1, x2, y2],  // pixel coords
                "transcription": "text",
                "is_legible": true
            },
            ...
        ],
        "full_text": "complete text content"
    }
    """
    if not json_path.exists():
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    regions = []
    for r in data.get("text_regions", []):
        bbox = r.get("bbox", [0, 0, 0, 0])
        norm_bbox = [
            max(0.0, min(1.0, bbox[0] / img_w)),
            max(0.0, min(1.0, bbox[1] / img_h)),
            max(0.0, min(1.0, bbox[2] / img_w)),
            max(0.0, min(1.0, bbox[3] / img_h)),
        ] if img_w > 0 and img_h > 0 else [0.0, 0.0, 0.0, 0.0]

        regions.append(TextRegion(
            bbox=norm_bbox,
            bbox_raw=[float(x) for x in bbox],
            transcription=r.get("transcription", ""),
            is_legible=r.get("is_legible", True),
            confidence=1.0,
        ))

    full_text = data.get("full_text", "")
    return regions, full_text


def load(
    root_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    tesseract_cmd: Optional[str] = None,
) -> List[UnifiedSample]:
    """
    Load Custom Desktop Screenshots dataset (private).

    Expected directory structure:
        root_dir/
        ├── 2026-03-29_23-22-59.jpg
        ├── 2026-03-29_23-22-59.json  (optional manual annotations)
        └── ...

    Parameters
    ----------
    root_dir     : Path to screenshots directory
    split        : "train", "val", or "test" (all images from one pool, split later)
    max_samples  : Limit number of samples (None = use all)
    tesseract_cmd: Path to Tesseract executable

    Returns
    -------
    List of UnifiedSample objects with task=DESKTOP_TEXT
    """
    root = Path(root_dir)
    if not root.exists():
        logger.warning("Custom desktop screenshots dir not found: %s", root)
        return []

    # Collect all image files
    image_files = sorted(
        fp for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for fp in root.glob(ext)
    )

    if not image_files:
        logger.warning("No images found in %s", root)
        return []

    logger.info("Custom Desktop: found %d images in %s", len(image_files), root)

    samples: List[UnifiedSample] = []

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                w, h = img_rgb.size
        except Exception as exc:
            logger.warning("Cannot open %s: %s", img_path, exc)
            continue

        # Check for manual annotations first
        json_path = img_path.with_suffix(".json")
        manual = _load_manual_annotations(json_path, w, h)

        if manual is not None:
            text_regions, full_text = manual
            has_manual = True
            avg_conf = 100.0
            logger.debug("Using manual annotations for %s", img_path.name)
        else:
            # Auto-generate using Tesseract
            text_regions, full_text, avg_conf = _extract_word_regions(
                img_rgb,
                tesseract_cmd=tesseract_cmd,
                confidence_threshold=30,
            )
            has_manual = False
            logger.debug(
                "Auto-annotated %s: %d regions, %.0f%% avg conf",
                img_path.name, len(text_regions), avg_conf,
            )

        samples.append(UnifiedSample(
            image_path=str(img_path.resolve()),
            task=TaskType.DESKTOP_TEXT,
            source=DataSource.CUSTOM_DESKTOP,
            split=SplitType.TRAIN,  # Will be re-split later
            text_regions=text_regions,
            ocr_text=full_text or None,
            ocr_confidence=avg_conf,
            image_width=w,
            image_height=h,
            has_manual_annotations=has_manual,
        ))

        if max_samples and len(samples) >= max_samples:
            break

    logger.info(
        "Custom Desktop: loaded %d samples (%d manual, %d auto-annotated).",
        len(samples),
        sum(1 for s in samples if s.has_manual_annotations),
        sum(1 for s in samples if not s.has_manual_annotations),
    )

    return samples
