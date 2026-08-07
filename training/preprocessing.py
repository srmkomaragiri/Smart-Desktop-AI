"""
preprocessing.py — Normalization and cleaning pipeline
=======================================================
Applied to all UnifiedSample objects after loading, before training:
  1. Image normalization (resize, RGB, pixel scaling)
  2. Bbox format normalization (all → [x1,y1,x2,y2] in 0-1)
  3. Text cleaning (unicode, control chars, encoding)
  4. OCR enrichment (optional Tesseract for datasets without OCR)
  5. Deduplication (hash-based)
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

from training.schema import (
    DataSource,
    TaskType,
    TextRegion,
    UnifiedSample,
)

logger = logging.getLogger(__name__)


# ── Text cleaning ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean and normalize text content:
    - Remove control characters (except newlines/tabs)
    - Normalize unicode (NFC)
    - Strip excessive whitespace
    - Remove null bytes
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Normalize unicode to NFC
    text = unicodedata.normalize("NFC", text)

    # Remove control characters except \n \t \r
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Collapse excessive whitespace (keep single newlines)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── Bbox validation ──────────────────────────────────────────

def validate_bbox(bbox: List[float]) -> bool:
    """Check that a normalized bbox is valid."""
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    if any(v < 0.0 or v > 1.0 for v in bbox):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    # Filter extremely tiny regions (likely noise)
    area = (x2 - x1) * (y2 - y1)
    if area < 1e-6:
        return False
    return True


def clamp_bbox(bbox: List[float]) -> List[float]:
    """Clamp bbox values to [0, 1] range and ensure x1 < x2, y1 < y2."""
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    # Ensure proper ordering
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


# ── Image validation ─────────────────────────────────────────

def validate_image(image_path: str) -> bool:
    """Check that an image file exists and can be opened."""
    path = Path(image_path)
    if not path.exists():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image width and height."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (0, 0)


# ── Deduplication ─────────────────────────────────────────────

def _compute_image_hash(image_path: str, hash_size: int = 8) -> str:
    """
    Compute a perceptual hash of an image for deduplication.
    Uses average hash (aHash) — fast and good enough for exact/near dupes.
    """
    try:
        with Image.open(image_path) as img:
            # Resize to small square
            small = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
            pixels = list(small.getdata())
            avg = sum(pixels) / len(pixels)
            bits = "".join("1" if p >= avg else "0" for p in pixels)
            return hashlib.md5(bits.encode()).hexdigest()
    except Exception:
        # Fallback to file hash
        try:
            data = Path(image_path).read_bytes()
            return hashlib.md5(data).hexdigest()
        except Exception:
            # Last resort: hash the path itself so same-path samples dedup
            return hashlib.md5(image_path.encode()).hexdigest()


def deduplicate(
    samples: List[UnifiedSample],
    cross_dataset: bool = True,
) -> List[UnifiedSample]:
    """
    Remove duplicate samples based on image content hash.

    Parameters
    ----------
    samples        : List of samples to deduplicate
    cross_dataset  : If True, deduplicate across datasets. If False, only within.

    Returns
    -------
    Deduplicated list (preserves order, keeps first occurrence)
    """
    seen_hashes: Set[str] = set()
    unique: List[UnifiedSample] = []
    removed = 0

    for sample in samples:
        img_hash = _compute_image_hash(sample.image_path)
        if not img_hash:
            unique.append(sample)  # Can't hash → keep
            continue

        # Use source-qualified hash if not cross-dataset
        key = img_hash if cross_dataset else f"{sample.source.value}:{img_hash}"

        if key in seen_hashes:
            removed += 1
            continue

        seen_hashes.add(key)
        unique.append(sample)

    if removed > 0:
        logger.info("Deduplication: removed %d duplicates (kept %d).", removed, len(unique))

    return unique


# ── Main preprocessing ───────────────────────────────────────

def preprocess_sample(sample: UnifiedSample) -> Optional[UnifiedSample]:
    """
    Apply all preprocessing steps to a single sample.
    Returns None if the sample is invalid and should be discarded.
    """
    # 1. Validate image exists
    if not Path(sample.image_path).exists():
        logger.debug("Image not found: %s", sample.image_path)
        return None

    # 2. Update image dimensions if missing
    if sample.image_width <= 0 or sample.image_height <= 0:
        w, h = get_image_dimensions(sample.image_path)
        if w <= 0 or h <= 0:
            logger.debug("Invalid image dimensions: %s", sample.image_path)
            return None
        sample.image_width = w
        sample.image_height = h

    # 3. Clean text content
    if sample.ocr_text:
        sample.ocr_text = clean_text(sample.ocr_text)

    # 4. Clean and validate text regions
    valid_regions: List[TextRegion] = []
    for region in sample.text_regions:
        # Clean transcription
        region.transcription = clean_text(region.transcription)

        # Clamp and validate bbox
        region.bbox = clamp_bbox(region.bbox)
        if validate_bbox(region.bbox):
            valid_regions.append(region)

    sample.text_regions = valid_regions

    # 5. Clean form entities
    for entity in sample.entities:
        entity.text = clean_text(entity.text)

    # 6. Validate doc_class_id
    if sample.task == TaskType.DOC_CLASSIFICATION:
        if sample.doc_class_id is None or sample.doc_class_id < 0:
            logger.debug("Invalid doc_class_id for: %s", sample.image_path)
            return None

    return sample


def preprocess_all(
    samples: List[UnifiedSample],
    remove_duplicates: bool = True,
) -> List[UnifiedSample]:
    """
    Preprocess a full list of samples — clean, validate, deduplicate.

    Parameters
    ----------
    samples           : Raw samples from loaders
    remove_duplicates : Whether to run deduplication

    Returns
    -------
    Cleaned list of valid UnifiedSample objects
    """
    logger.info("Preprocessing %d samples...", len(samples))

    # Clean & validate
    processed = []
    discarded = 0
    for s in samples:
        result = preprocess_sample(s)
        if result is not None:
            processed.append(result)
        else:
            discarded += 1

    logger.info("Preprocessing: kept %d, discarded %d.", len(processed), discarded)

    # Deduplicate
    if remove_duplicates:
        processed = deduplicate(processed, cross_dataset=True)

    return processed
