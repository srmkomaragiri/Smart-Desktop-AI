"""
training.loaders — Dataset-specific loaders
============================================
Each loader implements: load(root_dir, split, max_samples) -> List[UnifiedSample]
"""

from .icdar2015 import load as load_icdar2015
from .cocotext import load as load_cocotext
from .rvlcdip import load as load_rvlcdip
from .funsd import load as load_funsd
from .custom_desktop import load as load_custom_desktop

__all__ = [
    "load_icdar2015",
    "load_cocotext",
    "load_rvlcdip",
    "load_funsd",
    "load_custom_desktop",
]
