"""
schema.py — Unified data schema for multi-dataset training pipeline
====================================================================
All dataset loaders produce UnifiedSample objects, ensuring a consistent
interface regardless of the source dataset or task type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ── Enums ─────────────────────────────────────────────────────

class TaskType(str, Enum):
    TEXT_DETECTION = "text_detection"
    DOC_CLASSIFICATION = "doc_classification"
    FORM_UNDERSTANDING = "form_understanding"
    DESKTOP_TEXT = "desktop_text"


class DataSource(str, Enum):
    ICDAR2015 = "icdar2015"
    COCOTEXT = "cocotext"
    RVLCDIP = "rvlcdip"
    FUNSD = "funsd"
    CUSTOM_DESKTOP = "custom_desktop"


class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class FormEntityLabel(str, Enum):
    QUESTION = "question"
    ANSWER = "answer"
    HEADER = "header"
    OTHER = "other"


# ── RVLCDIP class labels ─────────────────────────────────────

RVLCDIP_CLASSES: List[str] = [
    "letter", "form", "email", "handwritten",
    "advertisement", "scientific_report", "scientific_publication",
    "specification", "file_folder", "news_article",
    "budget", "invoice", "presentation", "questionnaire",
    "resume", "memo",
]

RVLCDIP_CLASS_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(RVLCDIP_CLASSES)}
RVLCDIP_ID_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(RVLCDIP_CLASSES)}


# ── Data structures ──────────────────────────────────────────

@dataclass
class TextRegion:
    """A detected text region with bounding box and transcription."""

    bbox: List[float]           # Normalized [x1, y1, x2, y2] in (0, 1)
    transcription: str = ""     # Ground-truth text content
    confidence: float = 1.0     # Detection / OCR confidence (0-1)
    is_legible: bool = True     # Whether text is legible
    language: str = "en"        # Language code
    bbox_raw: Optional[List[float]] = None  # Original unnormalized coords

    def area(self) -> float:
        """Compute normalized area."""
        if len(self.bbox) < 4:
            return 0.0
        w = abs(self.bbox[2] - self.bbox[0])
        h = abs(self.bbox[3] - self.bbox[1])
        return w * h

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "transcription": self.transcription,
            "confidence": self.confidence,
            "is_legible": self.is_legible,
            "language": self.language,
        }


@dataclass
class FormEntity:
    """A semantic entity in a form document (FUNSD format)."""

    id: int
    label: FormEntityLabel
    text: str
    bbox: List[int]                          # [left, top, right, bottom]
    words: List[Dict] = field(default_factory=list)  # [{text, box}, ...]
    linking: List[List[int]] = field(default_factory=list)  # [[from_id, to_id], ...]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label.value,
            "text": self.text,
            "bbox": self.bbox,
            "words": self.words,
            "linking": self.linking,
        }


@dataclass
class UnifiedSample:
    """
    Universal sample container used across all datasets and tasks.

    Every dataset loader produces a list of UnifiedSample objects. Fields
    not relevant to a particular task are left as None / empty list.
    """

    # ── Identity ──────────────────────────────────────────────
    image_path: str                          # Absolute path to image file
    task: TaskType                           # Primary task for this sample
    source: DataSource                       # Origin dataset
    split: SplitType = SplitType.TRAIN       # Data split assignment

    # ── Text detection (ICDAR, COCO-Text, Custom Desktop) ────
    text_regions: List[TextRegion] = field(default_factory=list)

    # ── Document classification (RVLCDIP) ────────────────────
    doc_class: Optional[str] = None          # Human-readable class name
    doc_class_id: Optional[int] = None       # Numeric class index (0-15)

    # ── Form understanding (FUNSD) ───────────────────────────
    entities: List[FormEntity] = field(default_factory=list)

    # ── OCR ground truth (all datasets) ──────────────────────
    ocr_text: Optional[str] = None           # Full page text
    ocr_confidence: Optional[float] = None   # Average confidence (0-100)

    # ── Image metadata ───────────────────────────────────────
    image_width: int = 0
    image_height: int = 0

    # ── Augmentation flag ────────────────────────────────────
    augmented: bool = False

    # ── Manual annotation support (future-proof) ─────────────
    has_manual_annotations: bool = False

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "image_path": self.image_path,
            "task": self.task.value,
            "source": self.source.value,
            "split": self.split.value,
            "text_regions": [r.to_dict() for r in self.text_regions],
            "doc_class": self.doc_class,
            "doc_class_id": self.doc_class_id,
            "entities": [e.to_dict() for e in self.entities],
            "ocr_text": self.ocr_text,
            "ocr_confidence": self.ocr_confidence,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "augmented": self.augmented,
            "has_manual_annotations": self.has_manual_annotations,
        }


@dataclass
class DatasetSplit:
    """Named container for train / val / test splits."""

    train: List[UnifiedSample] = field(default_factory=list)
    val: List[UnifiedSample] = field(default_factory=list)
    test: List[UnifiedSample] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def summary(self) -> str:
        from collections import Counter
        source_counts = Counter(s.source.value for s in self.train + self.val + self.test)
        task_counts = Counter(s.task.value for s in self.train + self.val + self.test)
        lines = [
            f"Total samples : {self.total}",
            f"  Train       : {len(self.train)}",
            f"  Validation  : {len(self.val)}",
            f"  Test        : {len(self.test)}",
            "",
            "By source:",
            *[f"  {k:20s}: {v}" for k, v in sorted(source_counts.items())],
            "",
            "By task:",
            *[f"  {k:20s}: {v}" for k, v in sorted(task_counts.items())],
        ]
        return "\n".join(lines)
