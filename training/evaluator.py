"""
evaluator.py — Per-task evaluation metrics
============================================
Computes task-specific metrics:
  - Text Detection: Precision, Recall, F1 @ IoU, H-mean
  - Document Classification: Accuracy, per-class P/R/F1, confusion matrix
  - Form Understanding: Entity-level F1, linking accuracy
  - Desktop Text: Character Error Rate (CER), Word Error Rate (WER)
  - Aggregate: Weighted macro-F1 across all tasks
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.schema import (
    RVLCDIP_CLASSES,
    RVLCDIP_ID_TO_CLASS,
    TaskType,
    UnifiedSample,
)

logger = logging.getLogger(__name__)


# ── Utility metrics ──────────────────────────────────────────

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two normalized [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_precision_recall_f1(
    tp: int, fp: int, fn: int,
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    try:
        import editdistance
        return editdistance.eval(s1, s2)
    except ImportError:
        # Fallback: dynamic programming
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]


# ── Task-specific evaluators ─────────────────────────────────

class TextDetectionEvaluator:
    """Evaluate text detection using IoU-based matching."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(
        self,
        pred_boxes: List[List[float]],
        gt_boxes: List[List[float]],
    ) -> None:
        """
        Update counts with predicted vs ground-truth boxes.
        Uses greedy IoU matching.
        """
        matched_gt = set()

        for pred in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                self.tp += 1
                matched_gt.add(best_gt_idx)
            else:
                self.fp += 1

        self.fn += len(gt_boxes) - len(matched_gt)

    def compute(self) -> Dict[str, float]:
        p, r, f1 = compute_precision_recall_f1(self.tp, self.fp, self.fn)
        # H-mean (harmonic mean of precision and recall) — ICDAR standard
        hmean = f1  # F1 IS the harmonic mean of P and R
        return {
            "precision": p,
            "recall": r,
            "f1": f1,
            "hmean": hmean,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


class DocClassificationEvaluator:
    """Evaluate document classification."""

    def __init__(self, num_classes: int = 16) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.predictions: List[int] = []
        self.ground_truth: List[int] = []

    def update(self, pred: int, gt: int) -> None:
        self.predictions.append(pred)
        self.ground_truth.append(gt)

    def update_batch(self, preds: List[int], gts: List[int]) -> None:
        self.predictions.extend(preds)
        self.ground_truth.extend(gts)

    def compute(self) -> Dict[str, Any]:
        """Compute accuracy, per-class metrics, confusion matrix."""
        if not self.predictions:
            return {"accuracy": 0.0, "per_class": {}, "confusion_matrix": []}

        correct = sum(p == g for p, g in zip(self.predictions, self.ground_truth))
        accuracy = correct / len(self.predictions)

        # Per-class metrics
        per_class = {}
        for cls_id in range(self.num_classes):
            tp = sum(1 for p, g in zip(self.predictions, self.ground_truth) if p == cls_id and g == cls_id)
            fp = sum(1 for p, g in zip(self.predictions, self.ground_truth) if p == cls_id and g != cls_id)
            fn = sum(1 for p, g in zip(self.predictions, self.ground_truth) if p != cls_id and g == cls_id)
            p, r, f1 = compute_precision_recall_f1(tp, fp, fn)
            cls_name = RVLCDIP_ID_TO_CLASS.get(cls_id, f"class_{cls_id}")
            per_class[cls_name] = {"precision": p, "recall": r, "f1": f1}

        # Confusion matrix
        confusion = [[0] * self.num_classes for _ in range(self.num_classes)]
        for p, g in zip(self.predictions, self.ground_truth):
            if 0 <= p < self.num_classes and 0 <= g < self.num_classes:
                confusion[g][p] += 1

        # Macro F1
        f1_scores = [v["f1"] for v in per_class.values()]
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class": per_class,
            "confusion_matrix": confusion,
        }


class FormExtractionEvaluator:
    """Evaluate form entity extraction."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.entity_preds: List[int] = []
        self.entity_gts: List[int] = []
        self.linking_correct = 0
        self.linking_total = 0

    def update_entities(self, pred_labels: List[int], gt_labels: List[int]) -> None:
        self.entity_preds.extend(pred_labels)
        self.entity_gts.extend(gt_labels)

    def update_linking(self, pred_links: List, gt_links: List) -> None:
        gt_set = set(tuple(l) for l in gt_links)
        pred_set = set(tuple(l) for l in pred_links)
        self.linking_correct += len(pred_set & gt_set)
        self.linking_total += len(gt_set)

    def compute(self) -> Dict[str, Any]:
        label_names = ["question", "answer", "header", "other"]
        per_label = {}

        for label_id, label_name in enumerate(label_names):
            tp = sum(1 for p, g in zip(self.entity_preds, self.entity_gts) if p == label_id and g == label_id)
            fp = sum(1 for p, g in zip(self.entity_preds, self.entity_gts) if p == label_id and g != label_id)
            fn = sum(1 for p, g in zip(self.entity_preds, self.entity_gts) if p != label_id and g == label_id)
            p, r, f1 = compute_precision_recall_f1(tp, fp, fn)
            per_label[label_name] = {"precision": p, "recall": r, "f1": f1}

        # Entity-level F1 (macro)
        f1_scores = [v["f1"] for v in per_label.values()]
        entity_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        # Linking accuracy
        linking_acc = (
            self.linking_correct / self.linking_total
            if self.linking_total > 0 else 0.0
        )

        return {
            "entity_f1": entity_f1,
            "linking_accuracy": linking_acc,
            "per_label": per_label,
        }


class DesktopTextEvaluator:
    """Evaluate desktop text recognition using CER and WER."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_chars = 0
        self.total_char_errors = 0
        self.total_words = 0
        self.total_word_errors = 0
        self.confidences: List[float] = []

    def update(self, pred_text: str, gt_text: str, confidence: float = 1.0) -> None:
        """Update with a single prediction/ground-truth pair."""
        # CER
        self.total_chars += len(gt_text)
        self.total_char_errors += edit_distance(pred_text, gt_text)

        # WER
        pred_words = pred_text.split()
        gt_words = gt_text.split()
        self.total_words += len(gt_words)
        self.total_word_errors += edit_distance(
            " ".join(pred_words), " ".join(gt_words)
        )

        self.confidences.append(confidence)

    def compute(self) -> Dict[str, float]:
        cer = (
            self.total_char_errors / self.total_chars
            if self.total_chars > 0 else 0.0
        )
        wer = (
            self.total_word_errors / self.total_words
            if self.total_words > 0 else 0.0
        )
        avg_confidence = (
            sum(self.confidences) / len(self.confidences)
            if self.confidences else 0.0
        )

        return {
            "cer": cer,
            "wer": wer,
            "avg_confidence": avg_confidence,
            "total_chars_evaluated": self.total_chars,
            "total_words_evaluated": self.total_words,
        }


# ── Aggregate evaluator ──────────────────────────────────────

class PipelineEvaluator:
    """
    Aggregates all per-task evaluators and produces a unified report.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.text_det = TextDetectionEvaluator(iou_threshold)
        self.doc_cls = DocClassificationEvaluator()
        self.form_ext = FormExtractionEvaluator()
        self.desktop = DesktopTextEvaluator()

    def reset_all(self) -> None:
        self.text_det.reset()
        self.doc_cls.reset()
        self.form_ext.reset()
        self.desktop.reset()

    def compute_all(self) -> Dict[str, Any]:
        """Compute all metrics and return a unified report."""
        det_metrics = self.text_det.compute()
        cls_metrics = self.doc_cls.compute()
        form_metrics = self.form_ext.compute()
        desktop_metrics = self.desktop.compute()

        # Aggregate weighted macro-F1
        task_f1s = []
        weights = []

        if det_metrics["tp"] + det_metrics["fp"] + det_metrics["fn"] > 0:
            task_f1s.append(det_metrics["f1"])
            weights.append(1.0)

        if cls_metrics.get("macro_f1", 0) > 0:
            task_f1s.append(cls_metrics["macro_f1"])
            weights.append(1.0)

        if form_metrics.get("entity_f1", 0) > 0:
            task_f1s.append(form_metrics["entity_f1"])
            weights.append(1.0)

        if desktop_metrics.get("total_chars_evaluated", 0) > 0:
            # Convert CER to an F1-like score (1 - CER)
            desktop_f1 = max(0.0, 1.0 - desktop_metrics["cer"])
            task_f1s.append(desktop_f1)
            weights.append(1.5)  # Private dataset gets higher weight

        if task_f1s:
            weighted_f1 = sum(f * w for f, w in zip(task_f1s, weights)) / sum(weights)
        else:
            weighted_f1 = 0.0

        return {
            "aggregate": {
                "weighted_macro_f1": weighted_f1,
            },
            "text_detection": det_metrics,
            "doc_classification": cls_metrics,
            "form_understanding": form_metrics,
            "desktop_text": desktop_metrics,
        }

    def save_report(self, output_path: str, report: Optional[Dict] = None) -> None:
        """Save metrics report as JSON."""
        if report is None:
            report = self.compute_all()

        # Convert numpy types for JSON serialization
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        import json
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=_convert)

        logger.info("Metrics report saved to: %s", path)
