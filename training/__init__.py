"""
training — Unified Multi-Dataset Training & Evaluation Pipeline
================================================================
Integrates 5 datasets (Custom Desktop, ICDAR 2015, COCO-Text, RVLCDIP, FUNSD)
into a single pipeline for multi-task training:
  • Scene text detection (ICDAR 2015, COCO-Text)
  • Document classification (RVLCDIP)
  • Form understanding & key-value extraction (FUNSD)
  • Domain-specific desktop text recognition (Custom Desktop)

Usage:
    python -m training.pipeline --config training/config.yaml --stage all
"""

__version__ = "1.0.0"
