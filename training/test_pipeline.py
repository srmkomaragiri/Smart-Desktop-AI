"""
test_pipeline.py — Integration test for the training pipeline
==============================================================
Tests all non-torch components: schema, loaders, preprocessing,
augmentation (balancing), evaluator, reports, and download verify.
Torch-dependent tests (dataset, models, trainer) require torch>=2.0.
"""
import sys
sys.path.insert(0, ".")

print("=" * 60)
print("  TRAINING PIPELINE INTEGRATION TEST")
print("=" * 60)

# ── Test 1: Schema ────────────────────────────────────────────
print("\n[1] Schema...")
from training.schema import (
    UnifiedSample, TextRegion, FormEntity, DatasetSplit,
    TaskType, DataSource, SplitType, RVLCDIP_CLASSES,
)
assert len(RVLCDIP_CLASSES) == 16
s = UnifiedSample(image_path="test.jpg", task=TaskType.TEXT_DETECTION, source=DataSource.ICDAR2015)
d = s.to_dict()
assert d["task"] == "text_detection"
assert d["source"] == "icdar2015"
ds = DatasetSplit()
assert ds.total == 0
print("    PASS: 16 RVLCDIP classes, schema serialization OK")

# ── Test 2: Preprocessing ────────────────────────────────────
print("\n[2] Preprocessing...")
from training.preprocessing import clean_text, validate_bbox, clamp_bbox, deduplicate
assert clean_text("hello\x00world\n\n\ntest") == "helloworld\n\ntest"
assert validate_bbox([0.1, 0.2, 0.8, 0.9]) is True
assert validate_bbox([0.5, 0.5, 0.5, 0.5]) is False  # zero area
assert clamp_bbox([0.9, 0.1, 0.2, 0.8]) == [0.2, 0.1, 0.9, 0.8]  # swapped
print("    PASS: text cleaning, bbox validation, clamping OK")

# ── Test 3: Evaluator ────────────────────────────────────────
print("\n[3] Evaluator...")
from training.evaluator import PipelineEvaluator, compute_iou, edit_distance
assert abs(compute_iou([0, 0, 1, 1], [0.5, 0.5, 1, 1]) - 0.25) < 0.01
assert edit_distance("kitten", "sitting") == 3
ev = PipelineEvaluator()
ev.text_det.update([[0.1, 0.1, 0.5, 0.5]], [[0.1, 0.1, 0.5, 0.5]])
ev.doc_cls.update(0, 0)
ev.doc_cls.update(1, 1)
ev.desktop.update("hello world", "hello world", 0.9)
r = ev.compute_all()
assert r["text_detection"]["f1"] == 1.0
assert r["doc_classification"]["accuracy"] == 1.0
assert r["desktop_text"]["cer"] == 0.0
print(f"    PASS: IoU={compute_iou([0,0,1,1],[0.5,0.5,1,1]):.2f}, "
      f"Detection F1={r['text_detection']['f1']:.1f}, "
      f"Classification Acc={r['doc_classification']['accuracy']:.1f}")

# ── Test 4: Custom Desktop Loader ─────────────────────────────
print("\n[4] Custom Desktop Loader (PRIVATE dataset)...")
from training.loaders.custom_desktop import load as load_desktop
samples = load_desktop(
    root_dir="./screenshots",
    tesseract_cmd=r"F:\tesseract\tesseract.exe",
)
n = len(samples)
print(f"    Loaded {n} screenshots")
assert n > 0, "FAILED: No screenshots loaded!"
total_regions = sum(len(s.text_regions) for s in samples)
print(f"    Total text regions: {total_regions}")
for s in samples[:3]:
    name = s.image_path.split("\\")[-1]
    print(f"      {name}: {len(s.text_regions)} regions, conf={s.ocr_confidence:.0f}%")
assert all(s.source == DataSource.CUSTOM_DESKTOP for s in samples)
assert all(s.task == TaskType.DESKTOP_TEXT for s in samples)
print(f"    PASS: All {n} samples have source=custom_desktop, task=desktop_text")

# ── Test 5: Full preprocessing on private data ───────────────
print("\n[5] Preprocessing pipeline on private data...")
from training.preprocessing import preprocess_all
clean = preprocess_all(samples, remove_duplicates=True)
print(f"    After preprocessing: {len(clean)} samples")
assert len(clean) > 0

# ── Test 6: Augmentation (balancing) ─────────────────────────
print("\n[6] Augmentation & balancing...")
from training.augmentation import balance_datasets, compute_sample_weights
balanced = balance_datasets(clean, augment_config={"custom_desktop": 19})
assert len(balanced) >= len(clean) * 10
aug_count = sum(1 for s in balanced if s.augmented)
orig_count = sum(1 for s in balanced if not s.augmented)
print(f"    Original: {orig_count}, Augmented: {aug_count}, Total: {len(balanced)}")
weights = compute_sample_weights(balanced)
assert len(weights) == len(balanced)
print(f"    Sample weights computed: min={min(weights):.4f}, max={max(weights):.4f}")
print("    PASS")

# ── Test 7: Download/verify utility ──────────────────────────
print("\n[7] Download/verify utility...")
from training.download import verify_dataset
result = verify_dataset("./screenshots", "custom_desktop")
assert result["valid"], f"Custom desktop validation failed: {result['message']}"
print(f"    Custom Desktop: {result['message']}")
print("    PASS")

# ── Test 8: HTML Report Generation ───────────────────────────
print("\n[8] HTML report generation...")
from training.reports import generate_html_report
path = generate_html_report(
    metrics=r,
    history={"train_loss": [1.0, 0.8, 0.6, 0.4, 0.3], "val_loss": [1.1, 0.9, 0.7, 0.5, 0.4]},
    output_path="./training/results/report.html",
)
import os
assert os.path.exists(path.replace("\\", "/")) or os.path.exists(path)
print(f"    Report: {path}")
print("    PASS")

# ── Test 9: Config loading ───────────────────────────────────
print("\n[9] Config YAML loading...")
import yaml
with open("training/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
assert "datasets" in cfg
assert "custom_desktop" in cfg["datasets"]
assert cfg["datasets"]["custom_desktop"]["enabled"] is True
assert cfg["training"]["task_weights"]["desktop_text"] == 1.5
print(f"    Datasets: {list(cfg['datasets'].keys())}")
print(f"    Desktop weight: {cfg['training']['task_weights']['desktop_text']}")
print("    PASS")

# ── Test 10: Torch availability check ────────────────────────
print("\n[10] Checking torch availability...")
try:
    import torch
    print(f"    PyTorch {torch.__version__} available — GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Test model creation
    from training.models.multi_task import MultiTaskModel
    model = MultiTaskModel(backbone_name="efficientnet_b0", pretrained=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    MultiTaskModel: {n_params/1e6:.2f}M params")
    print("    PASS: Model architecture validated")
except ImportError:
    print("    torch NOT installed — model tests skipped (install with: pip install torch torchvision)")
    print("    SKIP (non-critical)")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED ✓")
print("=" * 60)
