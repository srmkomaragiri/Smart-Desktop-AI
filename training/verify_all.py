"""
verify_all.py - Comprehensive verification of the training pipeline
====================================================================
Tests every module: schema, loaders, preprocessing, augmentation,
evaluator, reports, download, config, and (optionally) torch models.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTHONIOENCODING"] = "utf-8"

passed = 0
failed = 0
skipped = 0

def check(name, fn):
    global passed, failed, skipped
    try:
        result = fn()
        if result == "SKIP":
            skipped += 1
            print(f"  [SKIP] {name}")
        else:
            passed += 1
            print(f"  [PASS] {name}")
    except Exception as e:
        failed += 1
        print(f"  [FAIL] {name}: {e}")


print("=" * 60)
print("  COMPREHENSIVE PIPELINE VERIFICATION")
print("=" * 60)

# ── 1. Schema ──────────────────────────────────────────────────
print("\n--- Schema ---")

def test_schema_imports():
    from training.schema import (
        UnifiedSample, TextRegion, FormEntity, DatasetSplit,
        TaskType, DataSource, SplitType, RVLCDIP_CLASSES,
    )
    assert len(RVLCDIP_CLASSES) == 16, f"Expected 16 RVLCDIP classes, got {len(RVLCDIP_CLASSES)}"
check("Schema imports + RVLCDIP_CLASSES", test_schema_imports)

def test_schema_serialization():
    from training.schema import UnifiedSample, TaskType, DataSource
    s = UnifiedSample(image_path="test.jpg", task=TaskType.TEXT_DETECTION, source=DataSource.ICDAR2015)
    d = s.to_dict()
    assert d["task"] == "text_detection"
    assert d["source"] == "icdar2015"
check("Schema serialization", test_schema_serialization)

def test_dataset_split():
    from training.schema import DatasetSplit
    ds = DatasetSplit()
    assert ds.total == 0
check("DatasetSplit empty init", test_dataset_split)

# ── 2. Preprocessing ──────────────────────────────────────────
print("\n--- Preprocessing ---")

def test_clean_text():
    from training.preprocessing import clean_text
    assert clean_text("hello\x00world\n\n\ntest") == "helloworld\n\ntest"
check("clean_text", test_clean_text)

def test_validate_bbox():
    from training.preprocessing import validate_bbox
    assert validate_bbox([0.1, 0.2, 0.8, 0.9]) is True
    assert validate_bbox([0.5, 0.5, 0.5, 0.5]) is False  # zero area
check("validate_bbox", test_validate_bbox)

def test_clamp_bbox():
    from training.preprocessing import clamp_bbox
    result = clamp_bbox([0.9, 0.1, 0.2, 0.8])
    assert result == [0.2, 0.1, 0.9, 0.8], f"clamp_bbox swap failed: {result}"
check("clamp_bbox", test_clamp_bbox)

def test_deduplicate():
    from training.preprocessing import deduplicate
    from training.schema import UnifiedSample, TaskType, DataSource
    s1 = UnifiedSample(image_path="a.jpg", task=TaskType.TEXT_DETECTION, source=DataSource.ICDAR2015)
    s2 = UnifiedSample(image_path="a.jpg", task=TaskType.TEXT_DETECTION, source=DataSource.ICDAR2015)
    s3 = UnifiedSample(image_path="b.jpg", task=TaskType.TEXT_DETECTION, source=DataSource.ICDAR2015)
    result = deduplicate([s1, s2, s3])
    assert len(result) == 2, f"Expected 2 unique, got {len(result)}"
check("deduplicate", test_deduplicate)

# ── 3. Evaluator ──────────────────────────────────────────────
print("\n--- Evaluator ---")

def test_iou():
    from training.evaluator import compute_iou
    iou = compute_iou([0, 0, 1, 1], [0.5, 0.5, 1, 1])
    assert abs(iou - 0.25) < 0.01, f"IoU expected ~0.25, got {iou}"
check("compute_iou", test_iou)

def test_edit_distance():
    from training.evaluator import edit_distance
    d = edit_distance("kitten", "sitting")
    assert d == 3, f"edit_distance expected 3, got {d}"
check("edit_distance", test_edit_distance)

def test_evaluator_perfect():
    from training.evaluator import PipelineEvaluator
    ev = PipelineEvaluator()
    ev.text_det.update([[0.1, 0.1, 0.5, 0.5]], [[0.1, 0.1, 0.5, 0.5]])
    ev.doc_cls.update(0, 0)
    ev.doc_cls.update(1, 1)
    ev.desktop.update("hello world", "hello world", 0.9)
    r = ev.compute_all()
    assert r["text_detection"]["f1"] == 1.0, f"Detection F1 != 1.0"
    assert r["doc_classification"]["accuracy"] == 1.0, f"Classification acc != 1.0"
    assert r["desktop_text"]["cer"] == 0.0, f"CER != 0.0"
check("PipelineEvaluator (perfect scores)", test_evaluator_perfect)

# ── 4. Loaders (import) ───────────────────────────────────────
print("\n--- Loaders (import) ---")

def test_loader_icdar():
    from training.loaders.icdar2015 import load
check("icdar2015 import", test_loader_icdar)

def test_loader_coco():
    from training.loaders.cocotext import load
check("cocotext import", test_loader_coco)

def test_loader_rvlcdip():
    from training.loaders.rvlcdip import load
check("rvlcdip import", test_loader_rvlcdip)

def test_loader_funsd():
    from training.loaders.funsd import load
check("funsd import", test_loader_funsd)

def test_loader_desktop():
    from training.loaders.custom_desktop import load
check("custom_desktop import", test_loader_desktop)

# ── 5. Custom Desktop Loader (live) ───────────────────────────
print("\n--- Custom Desktop Loader (live data) ---")

desktop_samples = []

def test_load_screenshots():
    global desktop_samples
    from training.loaders.custom_desktop import load as load_desktop
    desktop_samples = load_desktop(
        root_dir="./screenshots",
        tesseract_cmd=r"F:\tesseract\tesseract.exe",
    )
    assert len(desktop_samples) > 0, "No screenshots loaded"
    print(f"    Loaded {len(desktop_samples)} screenshots")
check("Load private screenshots", test_load_screenshots)

def test_desktop_source_task():
    from training.schema import DataSource, TaskType
    assert all(s.source == DataSource.CUSTOM_DESKTOP for s in desktop_samples)
    assert all(s.task == TaskType.DESKTOP_TEXT for s in desktop_samples)
check("Source/task metadata correct", test_desktop_source_task)

# ── 6. Preprocessing on live data ─────────────────────────────
print("\n--- Preprocessing (live) ---")

clean_samples = []

def test_preprocess_live():
    global clean_samples
    from training.preprocessing import preprocess_all
    clean_samples = preprocess_all(desktop_samples, remove_duplicates=True)
    assert len(clean_samples) > 0
    print(f"    After cleaning: {len(clean_samples)} samples")
check("preprocess_all on screenshots", test_preprocess_live)

# ── 7. Augmentation & Balancing ────────────────────────────────
print("\n--- Augmentation ---")

def test_balance():
    from training.augmentation import balance_datasets
    balanced = balance_datasets(clean_samples, augment_config={"custom_desktop": 19})
    assert len(balanced) >= len(clean_samples) * 10
    aug_count = sum(1 for s in balanced if s.augmented)
    orig_count = sum(1 for s in balanced if not s.augmented)
    print(f"    Original: {orig_count}, Augmented: {aug_count}, Total: {len(balanced)}")
check("balance_datasets", test_balance)

def test_sample_weights():
    from training.augmentation import compute_sample_weights, balance_datasets
    balanced = balance_datasets(clean_samples, augment_config={"custom_desktop": 5})
    weights = compute_sample_weights(balanced)
    assert len(weights) == len(balanced)
    print(f"    Weights: min={min(weights):.4f}, max={max(weights):.4f}")
check("compute_sample_weights", test_sample_weights)

# ── 8. Download/Verify ─────────────────────────────────────────
print("\n--- Download Utility ---")

def test_verify_dataset():
    from training.download import verify_dataset
    result = verify_dataset("./screenshots", "custom_desktop")
    assert result["valid"], f"Validation failed: {result['message']}"
    print(f"    {result['message']}")
check("verify_dataset (custom_desktop)", test_verify_dataset)

# ── 9. Reports ─────────────────────────────────────────────────
print("\n--- Reports ---")

def test_html_report():
    from training.reports import generate_html_report
    from training.evaluator import PipelineEvaluator
    ev = PipelineEvaluator()
    ev.text_det.update([[0.1, 0.1, 0.5, 0.5]], [[0.1, 0.1, 0.5, 0.5]])
    ev.doc_cls.update(0, 0)
    ev.desktop.update("hello", "hello", 0.9)
    metrics = ev.compute_all()
    path = generate_html_report(
        metrics=metrics,
        history={"train_loss": [1.0, 0.8, 0.6], "val_loss": [1.1, 0.9, 0.7]},
        output_path="./training/results/verify_report.html",
    )
    assert os.path.exists(path), f"Report not found at {path}"
    size_kb = os.path.getsize(path) / 1024
    print(f"    Report: {path} ({size_kb:.1f} KB)")
check("generate_html_report", test_html_report)

# ── 10. Config ─────────────────────────────────────────────────
print("\n--- Config ---")

def test_config_yaml():
    import yaml
    with open("training/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    assert "datasets" in cfg
    assert "custom_desktop" in cfg["datasets"]
    assert cfg["datasets"]["custom_desktop"]["enabled"] is True
    assert cfg["training"]["task_weights"]["desktop_text"] == 1.5
    print(f"    Backbone: {cfg['model']['backbone']}")
    print(f"    Epochs: {cfg['training']['epochs']}, Batch: {cfg['training']['batch_size']}")
    print(f"    Datasets: {list(cfg['datasets'].keys())}")
check("config.yaml loading", test_config_yaml)

# ── 11. Torch + Models ────────────────────────────────────────
print("\n--- Torch / Models ---")

def test_torch_models():
    try:
        import torch
    except ImportError:
        return "SKIP"

    print(f"    PyTorch {torch.__version__}, GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"    VRAM: {vram:.1f} GB")

    from training.models.multi_task import MultiTaskModel
    model = MultiTaskModel(backbone_name="efficientnet_b0", pretrained=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    MultiTaskModel: {n_params/1e6:.2f}M params")

check("MultiTaskModel creation", test_torch_models)

def test_model_submodules():
    try:
        import torch
    except ImportError:
        return "SKIP"
    from training.models.backbone import EfficientNetB0Backbone, ResNet18Backbone, create_backbone
    from training.models.text_detector import TextDetectionHead
    from training.models.doc_classifier import DocClassificationHead
    from training.models.form_extractor import FormExtractionHead
check("Model submodule imports", test_model_submodules)

# ── 12. Pipeline (import) ─────────────────────────────────────
print("\n--- Pipeline ---")

def test_pipeline_import():
    from training.pipeline import run_pipeline, load_config
check("pipeline.py import", test_pipeline_import)

def test_trainer_import():
    try:
        import torch
    except ImportError:
        return "SKIP"
    from training.trainer import Trainer
check("trainer.py import", test_trainer_import)

def test_dataset_import():
    try:
        import torch
    except ImportError:
        return "SKIP"
    from training.dataset import UnifiedDataset
check("dataset.py import", test_dataset_import)

# ── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
total = passed + failed + skipped
print(f"  RESULTS: {passed} passed, {failed} failed, {skipped} skipped / {total} total")
if failed == 0:
    print("  ALL TESTS PASSED!")
else:
    print(f"  {failed} FAILURE(S) - see above for details")
print("=" * 60)

sys.exit(1 if failed > 0 else 0)
