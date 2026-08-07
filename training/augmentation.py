"""
augmentation.py — Task-specific data augmentation and balancing
================================================================
Applies different augmentation strategies per task type and handles
dataset balancing through oversampling / undersampling / weighted sampling.
"""

from __future__ import annotations

import copy
import logging
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from training.schema import (
    DataSource,
    TaskType,
    TextRegion,
    UnifiedSample,
)

logger = logging.getLogger(__name__)


# ── Image augmentation primitives ─────────────────────────────

def _random_rotation(image: Image.Image, max_angle: float = 15.0) -> Tuple[Image.Image, float]:
    """Rotate image by a random angle within [-max_angle, +max_angle]."""
    angle = random.uniform(-max_angle, max_angle)
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(128, 128, 128))
    return rotated, angle


def _color_jitter(
    image: Image.Image,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.2,
) -> Image.Image:
    """Apply random brightness, contrast, and saturation jitter."""
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-brightness, brightness)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-contrast, contrast)
        image = ImageEnhance.Contrast(image).enhance(factor)
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-saturation, saturation)
        image = ImageEnhance.Color(image).enhance(factor)
    return image


def _gaussian_blur(image: Image.Image, max_radius: float = 2.0) -> Image.Image:
    """Apply Gaussian blur with random radius."""
    radius = random.uniform(0.1, max_radius)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _add_noise(image: Image.Image, intensity: float = 0.05) -> Image.Image:
    """Add random Gaussian noise to image."""
    arr = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, intensity * 255, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _random_scale(image: Image.Image, scale_range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
    """Randomly scale the image."""
    scale = random.uniform(*scale_range)
    new_w = int(image.width * scale)
    new_h = int(image.height * scale)
    if new_w < 32 or new_h < 32:
        return image
    scaled = image.resize((new_w, new_h), Image.LANCZOS)
    # Pad or crop back to original size
    result = Image.new("RGB", (image.width, image.height), (128, 128, 128))
    paste_x = (image.width - new_w) // 2
    paste_y = (image.height - new_h) // 2
    result.paste(scaled, (max(0, paste_x), max(0, paste_y)))
    return result


def _simulate_scanner_artifacts(image: Image.Image) -> Image.Image:
    """Simulate scanner noise: slight skew, brightness variation, edge shadows."""
    # Random small rotation (simulating paper skew)
    angle = random.uniform(-2, 2)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    # Slight brightness variation
    factor = random.uniform(0.9, 1.1)
    image = ImageEnhance.Brightness(image).enhance(factor)
    return image


def _theme_variation(image: Image.Image) -> Image.Image:
    """
    Simulate light/dark theme switch for desktop screenshots.
    Randomly inverts colors to simulate dark ↔ light mode transitions.
    """
    if random.random() < 0.4:
        image = ImageOps.invert(image.convert("RGB"))
    # Random slight tint (simulating different monitor color temperatures)
    if random.random() < 0.3:
        arr = np.array(image, dtype=np.float32)
        tint = np.array([
            random.uniform(0.95, 1.05),
            random.uniform(0.95, 1.05),
            random.uniform(0.95, 1.05),
        ])
        arr = np.clip(arr * tint, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
    return image


# ── Task-specific augmentation strategies ────────────────────

def augment_text_detection(image: Image.Image) -> Image.Image:
    """Augmentations for scene text detection (ICDAR, COCO-Text)."""
    if random.random() < 0.5:
        image, _ = _random_rotation(image, max_angle=15.0)
    if random.random() < 0.6:
        image = _color_jitter(image, brightness=0.3, contrast=0.3, saturation=0.2)
    if random.random() < 0.3:
        image = _gaussian_blur(image, max_radius=1.5)
    if random.random() < 0.2:
        image = _random_scale(image, scale_range=(0.85, 1.15))
    return image


def augment_doc_classification(image: Image.Image) -> Image.Image:
    """Augmentations for document classification (RVLCDIP)."""
    if random.random() < 0.4:
        image, _ = _random_rotation(image, max_angle=5.0)
    if random.random() < 0.5:
        image = _color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.1)
    if random.random() < 0.3:
        image = _add_noise(image, intensity=0.03)
    return image


def augment_form_understanding(image: Image.Image) -> Image.Image:
    """Augmentations for form understanding (FUNSD)."""
    if random.random() < 0.4:
        image, _ = _random_rotation(image, max_angle=3.0)
    if random.random() < 0.3:
        image = _random_scale(image, scale_range=(0.9, 1.1))
    if random.random() < 0.4:
        image = _simulate_scanner_artifacts(image)
    return image


def augment_desktop_text(image: Image.Image) -> Image.Image:
    """Augmentations for desktop text recognition (Custom Dataset)."""
    if random.random() < 0.4:
        image = _theme_variation(image)
    if random.random() < 0.3:
        image = _random_scale(image, scale_range=(0.8, 1.2))
    if random.random() < 0.4:
        image = _color_jitter(image, brightness=0.25, contrast=0.25, saturation=0.15)
    if random.random() < 0.2:
        image = _gaussian_blur(image, max_radius=1.0)
    if random.random() < 0.2:
        image = _add_noise(image, intensity=0.03)
    return image


# Map task → augmentation function
_AUGMENT_FN = {
    TaskType.TEXT_DETECTION: augment_text_detection,
    TaskType.DOC_CLASSIFICATION: augment_doc_classification,
    TaskType.FORM_UNDERSTANDING: augment_form_understanding,
    TaskType.DESKTOP_TEXT: augment_desktop_text,
}


def get_augment_fn(task: TaskType):
    """Get the appropriate augmentation function for a task type."""
    return _AUGMENT_FN.get(task, augment_text_detection)


# ── Dataset balancing ────────────────────────────────────────

def compute_sample_weights(
    samples: List[UnifiedSample],
    task_weights: Optional[Dict[str, float]] = None,
) -> List[float]:
    """
    Compute per-sample weights for weighted random sampling.
    Ensures balanced representation across datasets and tasks.

    Parameters
    ----------
    samples      : Full list of training samples
    task_weights : Optional per-task weight overrides (e.g. {"desktop_text": 1.5})

    Returns
    -------
    List of weights (same length as samples)
    """
    if task_weights is None:
        task_weights = {}

    # Count samples per source
    source_counts = Counter(s.source.value for s in samples)
    total = len(samples)

    # Inverse frequency weighting per source
    source_weights = {}
    n_sources = len(source_counts)
    for source, count in source_counts.items():
        # Weight inversely proportional to dataset size
        source_weights[source] = total / (n_sources * count)

    weights = []
    for s in samples:
        w = source_weights.get(s.source.value, 1.0)
        # Apply task-specific weight boost
        task_boost = task_weights.get(s.task.value, 1.0)
        weights.append(w * task_boost)

    # Normalize
    w_sum = sum(weights)
    if w_sum > 0:
        weights = [w / w_sum * total for w in weights]

    return weights


def oversample_dataset(
    samples: List[UnifiedSample],
    source: DataSource,
    factor: int,
) -> List[UnifiedSample]:
    """
    Create augmented copies of samples from a specific source.
    The augmented flag is set to True on copies.

    Parameters
    ----------
    samples  : Full training sample list
    source   : Which dataset to oversample
    factor   : How many copies to create (total = original + factor × original)

    Returns
    -------
    List of new augmented samples to ADD to the training set
    """
    source_samples = [s for s in samples if s.source == source]
    if not source_samples:
        logger.warning("No samples found for source %s — cannot oversample.", source.value)
        return []

    augmented: List[UnifiedSample] = []
    for _ in range(factor):
        for original in source_samples:
            aug_copy = copy.deepcopy(original)
            aug_copy.augmented = True
            augmented.append(aug_copy)

    logger.info(
        "Oversampled %s: %d originals × %d = %d augmented copies.",
        source.value, len(source_samples), factor, len(augmented),
    )
    return augmented


def balance_datasets(
    samples: List[UnifiedSample],
    augment_config: Optional[Dict[str, int]] = None,
    max_per_source: Optional[Dict[str, int]] = None,
) -> List[UnifiedSample]:
    """
    Balance the dataset through oversampling small datasets and
    capping large ones.

    Parameters
    ----------
    augment_config : {source_name: augment_factor} for oversampling
    max_per_source : {source_name: max_samples} for capping

    Returns
    -------
    Balanced list of samples
    """
    if augment_config is None:
        augment_config = {}
    if max_per_source is None:
        max_per_source = {}

    result = list(samples)  # Copy

    # Oversampling for small datasets
    for source_name, factor in augment_config.items():
        try:
            source = DataSource(source_name)
        except ValueError:
            logger.warning("Unknown source for augmentation: %s", source_name)
            continue
        augmented = oversample_dataset(result, source, factor)
        result.extend(augmented)

    # Capping for large datasets
    for source_name, max_n in max_per_source.items():
        try:
            source = DataSource(source_name)
        except ValueError:
            continue

        source_samples = [s for s in result if s.source == source]
        if len(source_samples) > max_n:
            # Keep originals first, then augmented
            originals = [s for s in source_samples if not s.augmented]
            augmented = [s for s in source_samples if s.augmented]

            # Keep all originals up to max, fill rest with augmented
            keep = originals[:max_n]
            if len(keep) < max_n:
                keep.extend(augmented[: max_n - len(keep)])

            keep_set = set(id(s) for s in keep)
            result = [
                s for s in result
                if s.source != source or id(s) in keep_set
            ]
            logger.info("Capped %s from %d to %d samples.", source_name, len(source_samples), len(keep))

    # Log final distribution
    dist = Counter(s.source.value for s in result)
    logger.info("Balanced distribution: %s (total: %d)", dict(dist), len(result))

    return result
