"""
download.py — Dataset download and verification utility
========================================================
Downloads and verifies public datasets:
  - ICDAR 2015 (requires manual download — provides instructions)
  - COCO-Text (cocotext.v2.json + MSCOCO images)
  - RVLCDIP (via HuggingFace datasets, subset)
  - FUNSD (via HuggingFace datasets)
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress bar using requests."""
    try:
        import requests
        from tqdm import tqdm

        dest.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc or dest.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        logger.info("Downloaded: %s", dest)
        return True

    except Exception as exc:
        logger.error("Download failed for %s: %s", url, exc)
        return False


def _extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """Extract a ZIP file."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        logger.info("Extracted: %s → %s", zip_path, dest_dir)
        return True
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        return False


def verify_dataset(root_dir: str, dataset: str) -> dict:
    """
    Verify that a dataset is properly downloaded and structured.

    Returns dict with:
        - valid: bool
        - message: str
        - n_images: int
        - n_annotations: int
    """
    root = Path(root_dir)

    if dataset == "icdar2015":
        train_imgs = root / "train" / "images"
        train_gt = root / "train" / "gt"
        valid = train_imgs.exists() and train_gt.exists()
        n_images = len(list(train_imgs.glob("*"))) if train_imgs.exists() else 0
        n_anns = len(list(train_gt.glob("*.txt"))) if train_gt.exists() else 0
        return {
            "valid": valid and n_images > 0,
            "message": f"Found {n_images} images, {n_anns} annotations"
            if valid else "Missing train/images or train/gt directories",
            "n_images": n_images,
            "n_annotations": n_anns,
        }

    elif dataset == "cocotext":
        ann_file = root / "annotations" / "cocotext.v2.json"
        img_dir = root / "images"
        valid = ann_file.exists() and img_dir.exists()
        n_images = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        return {
            "valid": valid and n_images > 0,
            "message": f"Found {n_images} images, annotation file present"
            if valid else "Missing annotations/cocotext.v2.json or images/ directory",
            "n_images": n_images,
            "n_annotations": 1 if ann_file.exists() else 0,
        }

    elif dataset == "rvlcdip":
        labels_dir = root / "labels"
        images_dir = root / "images"
        valid = labels_dir.exists()
        train_file = labels_dir / "train.txt" if labels_dir.exists() else None
        n_labels = 0
        if train_file and train_file.exists():
            with open(train_file) as f:
                n_labels = sum(1 for _ in f)
        n_images = 0
        if images_dir.exists():
            for ext in ("*.tif", "*.tiff", "*.png", "*.jpg"):
                n_images += len(list(images_dir.rglob(ext)))
        return {
            "valid": valid and n_labels > 0,
            "message": f"Found {n_images} images, {n_labels} label entries"
            if valid else "Missing labels/ directory",
            "n_images": n_images,
            "n_annotations": n_labels,
        }

    elif dataset == "funsd":
        train_dir = root / "training_data"
        valid = train_dir.exists()
        n_anns = 0
        n_images = 0
        if valid:
            ann_dir = train_dir / "annotations"
            img_dir = train_dir / "images"
            n_anns = len(list(ann_dir.glob("*.json"))) if ann_dir.exists() else 0
            n_images = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        return {
            "valid": valid and n_anns > 0,
            "message": f"Found {n_images} images, {n_anns} annotations"
            if valid else "Missing training_data/ directory",
            "n_images": n_images,
            "n_annotations": n_anns,
        }

    elif dataset == "custom_desktop":
        n_images = 0
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            n_images += len(list(root.glob(ext)))
        return {
            "valid": n_images > 0,
            "message": f"Found {n_images} screenshots",
            "n_images": n_images,
            "n_annotations": 0,
        }

    return {"valid": False, "message": f"Unknown dataset: {dataset}", "n_images": 0, "n_annotations": 0}


def download_funsd(dest_dir: str) -> bool:
    """Download FUNSD dataset."""
    root = Path(dest_dir)
    if (root / "training_data").exists():
        logger.info("FUNSD already downloaded at %s", root)
        return True

    # Try HuggingFace datasets
    try:
        from datasets import load_dataset
        logger.info("Downloading FUNSD via HuggingFace datasets...")
        ds = load_dataset("nielsr/funsd", trust_remote_code=True)

        # Save to expected format
        for split_name, split_data in [("training_data", ds.get("train")), ("testing_data", ds.get("test"))]:
            if split_data is None:
                continue

            ann_dir = root / split_name / "annotations"
            img_dir = root / split_name / "images"
            ann_dir.mkdir(parents=True, exist_ok=True)
            img_dir.mkdir(parents=True, exist_ok=True)

            for idx, item in enumerate(split_data):
                # Save image
                if "image" in item:
                    img = item["image"]
                    img_path = img_dir / f"{idx:010d}.png"
                    img.save(img_path)

                    # Save annotation
                    import json
                    ann_data = {}
                    for key in ("words", "bboxes", "ner_tags"):
                        if key in item:
                            ann_data[key] = item[key]

                    ann_path = ann_dir / f"{idx:010d}.json"
                    with open(ann_path, "w") as f:
                        json.dump(ann_data, f)

        logger.info("FUNSD downloaded successfully to %s", root)
        return True

    except Exception as exc:
        logger.warning("HuggingFace download failed: %s", exc)

    # Fallback: direct download
    url = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
    zip_path = root / "funsd.zip"
    if _download_file(url, zip_path, "FUNSD"):
        if _extract_zip(zip_path, root):
            zip_path.unlink(missing_ok=True)
            return True
    return False


def download_rvlcdip_subset(dest_dir: str, max_samples: int = 10000) -> bool:
    """Download a subset of RVL-CDIP via HuggingFace."""
    root = Path(dest_dir)
    if (root / "labels").exists():
        logger.info("RVL-CDIP already present at %s", root)
        return True

    try:
        from datasets import load_dataset
        logger.info("Downloading RVL-CDIP subset (%d samples) via HuggingFace...", max_samples)

        ds = load_dataset(
            "rvl_cdip",
            split=f"train[:{max_samples}]",
            trust_remote_code=True,
        )

        # Save images and create label files
        img_dir = root / "images"
        labels_dir = root / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        label_lines = []
        for idx, item in enumerate(ds):
            img = item["image"]
            label = item["label"]

            img_path = img_dir / f"{idx:06d}.png"
            img.save(img_path)

            rel_path = f"{idx:06d}.png"
            label_lines.append(f"{rel_path} {label}")

        # Write label file
        with open(labels_dir / "train.txt", "w") as f:
            f.write("\n".join(label_lines))

        logger.info("RVL-CDIP subset downloaded: %d samples to %s", len(label_lines), root)
        return True

    except Exception as exc:
        logger.error("RVL-CDIP download failed: %s", exc)
        return False


def print_download_instructions() -> None:
    """Print manual download instructions for datasets requiring registration."""
    instructions = """
╔══════════════════════════════════════════════════════════════╗
║           DATASET DOWNLOAD INSTRUCTIONS                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. ICDAR 2015 (Manual download required)                    ║
║     URL: https://rrc.cvc.uab.es/?ch=4&com=downloads         ║
║     → Download Challenge 4 Training + Test data              ║
║     → Extract to: training/data/icdar2015/                   ║
║       Structure: train/images/, train/gt/, test/images/...   ║
║                                                              ║
║  2. COCO-Text (Manual download required)                     ║
║     Annotations: https://bgshih.github.io/cocotext/          ║
║     Images: https://cocodataset.org/#download                ║
║     → Download cocotext.v2.json + MSCOCO train2014 images    ║
║     → Place in: training/data/cocotext/                      ║
║       Structure: annotations/cocotext.v2.json, images/       ║
║                                                              ║
║  3. RVLCDIP (Auto-download via HuggingFace)                 ║
║     Run: python -m training.download --dataset rvlcdip       ║
║                                                              ║
║  4. FUNSD (Auto-download via HuggingFace)                    ║
║     Run: python -m training.download --dataset funsd         ║
║                                                              ║
║  5. Custom Desktop (Already available in screenshots/)       ║
║     No action needed.                                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(instructions)


# ── CLI entry point ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Download datasets for training pipeline")
    parser.add_argument("--dataset", choices=["funsd", "rvlcdip", "all", "verify", "instructions"],
                        default="instructions", help="Dataset to download")
    parser.add_argument("--dest", default="./training/data", help="Destination directory")
    parser.add_argument("--max-samples", type=int, default=10000, help="Max samples for RVLCDIP")
    args = parser.parse_args()

    if args.dataset == "instructions":
        print_download_instructions()
        sys.exit(0)

    if args.dataset == "verify":
        for ds_name in ["icdar2015", "cocotext", "rvlcdip", "funsd"]:
            result = verify_dataset(f"{args.dest}/{ds_name}", ds_name)
            status = "✓" if result["valid"] else "✗"
            print(f"  {status} {ds_name:15s} — {result['message']}")
        # Also check custom desktop
        result = verify_dataset("./screenshots", "custom_desktop")
        status = "✓" if result["valid"] else "✗"
        print(f"  {status} {'custom_desktop':15s} — {result['message']}")
        sys.exit(0)

    if args.dataset in ("funsd", "all"):
        download_funsd(f"{args.dest}/funsd")

    if args.dataset in ("rvlcdip", "all"):
        download_rvlcdip_subset(f"{args.dest}/rvlcdip", args.max_samples)
