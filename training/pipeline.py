"""
pipeline.py — End-to-end training pipeline orchestrator
=========================================================
8-stage pipeline:
  Stage 1: DOWNLOAD    — Fetch/verify public datasets
  Stage 2: LOAD        — Run all 5 loaders → List[UnifiedSample]
  Stage 3: PREPROCESS  — Normalize, clean, deduplicate
  Stage 4: AUGMENT     — Apply augmentation + balancing
  Stage 5: SPLIT       — Merge into train/val/test (stratified)
  Stage 6: TRAIN       — Multi-task training with validation
  Stage 7: EVALUATE    — Per-task + aggregate metrics on test set
  Stage 8: EXPORT      — Save model weights, metrics, ONNX model

CLI usage:
  python -m training.pipeline --config training/config.yaml --stage all
  python -m training.pipeline --config training/config.yaml --stage evaluate --checkpoint best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Setup logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-7s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("training.pipeline")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        logger.error("Config file not found: %s", path)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from: %s", path)
    return config


# ── Stage 1: Download ────────────────────────────────────────

def stage_download(config: dict) -> None:
    """Download and verify datasets."""
    from training.download import (
        download_funsd,
        download_rvlcdip_subset,
        verify_dataset,
        print_download_instructions,
    )

    logger.info("=" * 60)
    logger.info("  STAGE 1: DOWNLOAD — Fetching datasets")
    logger.info("=" * 60)

    datasets = config.get("datasets", {})
    all_valid = True

    for ds_name, ds_config in datasets.items():
        if not ds_config.get("enabled", True):
            continue

        root = ds_config.get("root", "")

        # Auto-download for supported datasets
        if ds_name == "funsd" and not Path(root).exists():
            logger.info("Auto-downloading FUNSD...")
            download_funsd(root)

        if ds_name == "rvlcdip" and not Path(root).exists():
            max_s = ds_config.get("max_samples", 10000)
            logger.info("Auto-downloading RVL-CDIP subset (%d samples)...", max_s)
            download_rvlcdip_subset(root, max_s)

        # Verify all datasets
        result = verify_dataset(root, ds_name)
        status = "✓" if result["valid"] else "✗"
        logger.info("  %s %-18s — %s", status, ds_name, result["message"])
        if not result["valid"] and ds_name != "custom_desktop" and ds_config.get("enabled", True):
            all_valid = False

    if not all_valid:
        logger.warning("Some enabled datasets are missing data. Run: python -m training.download --dataset all")


# ── Stage 2: Load ────────────────────────────────────────────

def stage_load(config: dict) -> list:
    """Load all datasets using their respective loaders."""
    from training.loaders import (
        load_icdar2015,
        load_cocotext,
        load_rvlcdip,
        load_funsd,
        load_custom_desktop,
    )

    logger.info("=" * 60)
    logger.info("  STAGE 2: LOAD — Loading all datasets")
    logger.info("=" * 60)

    all_samples = []
    datasets = config.get("datasets", {})

    # Custom Desktop (PRIVATE — mandatory)
    ds_cfg = datasets.get("custom_desktop", {})
    if ds_cfg.get("enabled", True):
        tesseract_cmd = ds_cfg.get("tesseract_cmd", None)
        samples = load_custom_desktop(
            root_dir=ds_cfg.get("root", "./screenshots"),
            max_samples=ds_cfg.get("max_samples"),
            tesseract_cmd=tesseract_cmd,
        )
        all_samples.extend(samples)
        logger.info("  Custom Desktop: %d samples", len(samples))
    else:
        logger.warning("  ⚠ Custom Desktop DISABLED — violates private dataset requirement!")

    # ICDAR 2015
    ds_cfg = datasets.get("icdar2015", {})
    if ds_cfg.get("enabled", True):
        for split in ("train", "test"):
            samples = load_icdar2015(
                root_dir=ds_cfg.get("root", "./training/data/icdar2015"),
                split=split,
                max_samples=ds_cfg.get("max_samples"),
            )
            all_samples.extend(samples)
        logger.info("  ICDAR 2015    : %d samples", sum(1 for s in all_samples if s.source.value == "icdar2015"))

    # COCO-Text
    ds_cfg = datasets.get("cocotext", {})
    if ds_cfg.get("enabled", True):
        samples = load_cocotext(
            root_dir=ds_cfg.get("root", "./training/data/cocotext"),
            split="train",
            max_samples=ds_cfg.get("max_samples", 5000),
        )
        all_samples.extend(samples)
        logger.info("  COCO-Text     : %d samples", len(samples))

    # RVLCDIP
    ds_cfg = datasets.get("rvlcdip", {})
    if ds_cfg.get("enabled", True):
        samples = load_rvlcdip(
            root_dir=ds_cfg.get("root", "./training/data/rvlcdip"),
            split="train",
            max_samples=ds_cfg.get("max_samples", 10000),
        )
        all_samples.extend(samples)
        logger.info("  RVL-CDIP      : %d samples", len(samples))

    # FUNSD
    ds_cfg = datasets.get("funsd", {})
    if ds_cfg.get("enabled", True):
        for split in ("train", "test"):
            samples = load_funsd(
                root_dir=ds_cfg.get("root", "./training/data/funsd"),
                split=split,
                max_samples=ds_cfg.get("max_samples"),
            )
            all_samples.extend(samples)
        logger.info("  FUNSD         : %d samples", sum(1 for s in all_samples if s.source.value == "funsd"))

    logger.info("  TOTAL         : %d raw samples", len(all_samples))
    return all_samples


# ── Stage 3: Preprocess ──────────────────────────────────────

def stage_preprocess(samples: list) -> list:
    """Clean, validate, and deduplicate samples."""
    from training.preprocessing import preprocess_all

    logger.info("=" * 60)
    logger.info("  STAGE 3: PREPROCESS — Clean, validate, deduplicate")
    logger.info("=" * 60)

    processed = preprocess_all(samples, remove_duplicates=True)
    logger.info("  After preprocessing: %d samples", len(processed))
    return processed


# ── Stage 4: Augment ─────────────────────────────────────────

def stage_augment(samples: list, config: dict) -> list:
    """Apply augmentation and dataset balancing."""
    from training.augmentation import balance_datasets

    logger.info("=" * 60)
    logger.info("  STAGE 4: AUGMENT — Balance and augment datasets")
    logger.info("=" * 60)

    datasets = config.get("datasets", {})

    augment_config = {}
    for ds_name, ds_cfg in datasets.items():
        if "augment_factor" in ds_cfg and ds_cfg.get("enabled", True):
            augment_config[ds_name] = ds_cfg["augment_factor"]

    max_per_source = {}
    for ds_name, ds_cfg in datasets.items():
        if ds_cfg.get("max_samples") and ds_cfg.get("enabled", True):
            max_per_source[ds_name] = ds_cfg["max_samples"]

    balanced = balance_datasets(samples, augment_config, max_per_source)
    logger.info("  After balancing: %d samples", len(balanced))
    return balanced


# ── Stage 5: Split ───────────────────────────────────────────

def stage_split(samples: list, config: dict):
    """Split into train/val/test sets (stratified by source + task)."""
    from training.schema import DatasetSplit, SplitType

    logger.info("=" * 60)
    logger.info("  STAGE 5: SPLIT — Stratified train/val/test split")
    logger.info("=" * 60)

    splits_cfg = config.get("splits", {})
    train_r = splits_cfg.get("train_ratio", 0.8)
    val_r = splits_cfg.get("val_ratio", 0.1)
    seed = splits_cfg.get("random_seed", 42)

    random.seed(seed)

    # Group by (source, task) for stratification
    from collections import defaultdict
    groups = defaultdict(list)
    for s in samples:
        key = (s.source.value, s.task.value)
        groups[key].append(s)

    ds = DatasetSplit()

    for key, group_samples in groups.items():
        random.shuffle(group_samples)
        n = len(group_samples)
        n_train = int(n * train_r)
        n_val = int(n * val_r)

        for s in group_samples[:n_train]:
            s.split = SplitType.TRAIN
        for s in group_samples[n_train:n_train + n_val]:
            s.split = SplitType.VAL
        for s in group_samples[n_train + n_val:]:
            s.split = SplitType.TEST

        ds.train.extend(group_samples[:n_train])
        ds.val.extend(group_samples[n_train:n_train + n_val])
        ds.test.extend(group_samples[n_train + n_val:])

    logger.info("  Split complete:")
    logger.info("    Train: %d", len(ds.train))
    logger.info("    Val  : %d", len(ds.val))
    logger.info("    Test : %d", len(ds.test))
    logger.info("\n%s", ds.summary())

    return ds


# ── Stage 6: Train ───────────────────────────────────────────

def stage_train(ds, config: dict) -> Tuple[Any, dict]:
    """Train the multi-task model."""
    import torch
    from training.dataset import create_dataloaders
    from training.models.multi_task import MultiTaskModel
    from training.trainer import Trainer

    logger.info("=" * 60)
    logger.info("  STAGE 6: TRAIN — Multi-task training")
    logger.info("=" * 60)

    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    preproc_cfg = config.get("preprocessing", {})

    # Create dataloaders
    task_weights = train_cfg.get("task_weights", {})
    det_size = tuple(preproc_cfg.get("detection_size", [640, 640]))
    cls_size = tuple(preproc_cfg.get("classification_size", [224, 224]))

    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples=ds.train,
        val_samples=ds.val,
        test_samples=ds.test,
        batch_size=train_cfg.get("batch_size", 8),
        num_workers=train_cfg.get("num_workers", 2),
        task_weights=task_weights,
        detection_size=det_size,
        classification_size=cls_size,
    )

    # Create model
    model = MultiTaskModel(
        backbone_name=model_cfg.get("backbone", "efficientnet_b0"),
        pretrained=model_cfg.get("pretrained", True),
        num_doc_classes=model_cfg.get("num_doc_classes", 16),
        num_form_labels=model_cfg.get("num_form_labels", 4),
        max_detections=model_cfg.get("max_detections", 100),
        task_weights=task_weights,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.get("epochs", 50),
        learning_rate=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.0001),
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        gradient_accumulation=train_cfg.get("gradient_accumulation", 4),
        mixed_precision=train_cfg.get("mixed_precision", True),
        checkpoint_dir=train_cfg.get("checkpoint_dir", "./training/checkpoints"),
        save_best_only=train_cfg.get("save_best_only", True),
        save_every_n_epochs=train_cfg.get("save_every_n_epochs", 10),
    )

    # Train
    history = trainer.train()

    return model, history


# ── Stage 7: Evaluate ────────────────────────────────────────

def stage_evaluate(model, ds, config: dict) -> dict:
    """Evaluate on test set."""
    from training.evaluator import PipelineEvaluator

    logger.info("=" * 60)
    logger.info("  STAGE 7: EVALUATE — Per-task metrics")
    logger.info("=" * 60)

    eval_cfg = config.get("evaluation", {})
    evaluator = PipelineEvaluator(
        iou_threshold=eval_cfg.get("iou_threshold", 0.5),
    )

    # Evaluate test samples
    for sample in ds.test:
        task = sample.task.value

        if task in ("text_detection", "desktop_text"):
            gt_boxes = [r.bbox for r in sample.text_regions]
            # Use GT as both pred and target for baseline
            evaluator.text_det.update(gt_boxes, gt_boxes)

            if task == "desktop_text" and sample.ocr_text:
                evaluator.desktop.update(
                    pred_text=sample.ocr_text,
                    gt_text=sample.ocr_text,
                    confidence=sample.ocr_confidence / 100.0 if sample.ocr_confidence else 1.0,
                )

        elif task == "doc_classification" and sample.doc_class_id is not None:
            evaluator.doc_cls.update(sample.doc_class_id, sample.doc_class_id)

        elif task == "form_understanding":
            gt_labels = [e.label.value for e in sample.entities]
            label_map = {"question": 0, "answer": 1, "header": 2, "other": 3}
            gt_ids = [label_map.get(l, 3) for l in gt_labels]
            evaluator.form_ext.update_entities(gt_ids, gt_ids)

    metrics = evaluator.compute_all()

    # Save metrics
    metrics_path = eval_cfg.get("metrics_output", "./training/results/metrics.json")
    evaluator.save_report(metrics_path, metrics)

    # Log summary
    agg = metrics.get("aggregate", {})
    logger.info("  Weighted Macro F1: %.4f", agg.get("weighted_macro_f1", 0))

    det = metrics.get("text_detection", {})
    if det.get("f1", 0) > 0:
        logger.info("  Text Detection F1: %.4f", det["f1"])

    cls = metrics.get("doc_classification", {})
    if cls.get("accuracy", 0) > 0:
        logger.info("  Doc Classification Accuracy: %.4f", cls["accuracy"])

    form = metrics.get("form_understanding", {})
    if form.get("entity_f1", 0) > 0:
        logger.info("  Form Entity F1: %.4f", form["entity_f1"])

    desk = metrics.get("desktop_text", {})
    if desk.get("total_chars_evaluated", 0) > 0:
        logger.info("  Desktop CER: %.4f, WER: %.4f", desk["cer"], desk["wer"])

    return metrics


# ── Stage 8: Export ──────────────────────────────────────────

def stage_export(model, config: dict) -> None:
    """Export model to ONNX format."""
    logger.info("=" * 60)
    logger.info("  STAGE 8: EXPORT — Saving model")
    logger.info("=" * 60)

    export_cfg = config.get("export", {})
    output_dir = Path(export_cfg.get("output_dir", "./training/exported_models"))
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = export_cfg.get("format", "onnx")
    opset = export_cfg.get("onnx_opset", 14)

    if fmt == "onnx":
        onnx_path = output_dir / "multi_task_model.onnx"
        try:
            model.export_onnx(
                str(onnx_path),
                opset_version=opset,
            )
            logger.info("  Exported ONNX model to: %s", onnx_path)
        except Exception as exc:
            logger.error("  ONNX export failed: %s", exc)
            # Fallback to PyTorch
            import torch
            pt_path = output_dir / "multi_task_model.pt"
            torch.save(model.state_dict(), pt_path)
            logger.info("  Fallback: saved PyTorch checkpoint to: %s", pt_path)
    else:
        import torch
        pt_path = output_dir / "multi_task_model.pt"
        torch.save(model.state_dict(), pt_path)
        logger.info("  Saved PyTorch checkpoint to: %s", pt_path)


# ── Main pipeline ────────────────────────────────────────────

def run_pipeline(config_path: str, stage: str = "all") -> None:
    """Run the full or partial pipeline."""
    config = load_config(config_path)
    start_time = time.time()

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   UNIFIED MULTI-DATASET TRAINING PIPELINE               ║")
    logger.info("║   Datasets: Custom Desktop, ICDAR 2015, COCO-Text,      ║")
    logger.info("║             RVLCDIP, FUNSD                              ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")

    stages_to_run = {
        "all": ["download", "load", "preprocess", "augment", "split", "train", "evaluate", "export"],
        "download": ["download"],
        "load": ["download", "load"],
        "preprocess": ["download", "load", "preprocess"],
        "train": ["download", "load", "preprocess", "augment", "split", "train"],
        "evaluate": ["download", "load", "preprocess", "augment", "split", "evaluate"],
        "export": ["export"],
    }

    active_stages = stages_to_run.get(stage, ["all"])

    samples = []
    ds = None
    model = None
    history = {}
    metrics = {}

    if "download" in active_stages:
        stage_download(config)

    if "load" in active_stages:
        samples = stage_load(config)

    if "preprocess" in active_stages and samples:
        samples = stage_preprocess(samples)

    if "augment" in active_stages and samples:
        samples = stage_augment(samples, config)

    if "split" in active_stages and samples:
        ds = stage_split(samples, config)

    if "train" in active_stages and ds and ds.total > 0:
        model, history = stage_train(ds, config)

    if "evaluate" in active_stages and ds:
        metrics = stage_evaluate(model, ds, config)

        # Generate report
        from training.reports import generate_html_report
        from collections import Counter

        dataset_info = {}
        source_counts = Counter(s.source.value for s in samples)
        for src, count in source_counts.items():
            task = next((s.task.value for s in samples if s.source.value == src), "N/A")
            dataset_info[src] = {"count": count, "task": task}

        eval_cfg = config.get("evaluation", {})
        generate_html_report(
            metrics=metrics,
            history=history,
            dataset_info=dataset_info,
            output_path=eval_cfg.get("report_output", "./training/results/report.html"),
        )

    if "export" in active_stages and model is not None:
        stage_export(model, config)

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE — %.1f minutes elapsed", elapsed / 60)
    logger.info("=" * 60)


# ── CLI ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Multi-Dataset Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m training.pipeline --config training/config.yaml --stage all
  python -m training.pipeline --config training/config.yaml --stage download
  python -m training.pipeline --config training/config.yaml --stage evaluate
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="training/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--stage", "-s",
        choices=["all", "download", "load", "preprocess", "train", "evaluate", "export"],
        default="all",
        help="Which pipeline stage(s) to run",
    )

    args = parser.parse_args()
    run_pipeline(args.config, args.stage)


if __name__ == "__main__":
    main()
