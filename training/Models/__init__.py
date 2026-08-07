"""
training.models — Multi-task model architectures
=================================================
Shared backbone with task-specific heads for VRAM-efficient training.
"""

from .backbone import create_backbone, BackboneOutput
from .text_detector import TextDetectionHead
from .doc_classifier import DocClassificationHead
from .form_extractor import FormExtractionHead
from .multi_task import MultiTaskModel

__all__ = [
    "create_backbone",
    "BackboneOutput",
    "TextDetectionHead",
    "DocClassificationHead",
    "FormExtractionHead",
    "MultiTaskModel",
]
