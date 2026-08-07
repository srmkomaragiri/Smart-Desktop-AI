"""
reports.py — Structured evaluation reports
============================================
Generates HTML and JSON reports with training curves,
confusion matrices, per-dataset breakdowns, and cross-dataset
generalization analysis.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _generate_training_curves_svg(history: Dict[str, List[float]]) -> str:
    """Generate inline SVG for training/validation loss curves."""
    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])

    if not train_losses:
        return "<p>No training history available.</p>"

    n = len(train_losses)
    max_loss = max(max(train_losses, default=1), max(val_losses, default=1)) * 1.1
    if max_loss <= 0:
        max_loss = 1.0

    w, h = 600, 300
    margin = 50

    def to_xy(epoch: int, loss: float) -> tuple:
        x = margin + (epoch / max(1, n - 1)) * (w - 2 * margin)
        y = h - margin - (loss / max_loss) * (h - 2 * margin)
        return x, y

    # Build SVG
    parts = [
        f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg"'
        ' style="background:#1a1a2e;border-radius:8px;">',
        f'<text x="{w//2}" y="20" text-anchor="middle" fill="#e8e8f0" font-size="14">Training Curves</text>',
    ]

    # Grid lines
    for i in range(5):
        loss_val = max_loss * i / 4
        _, y = to_xy(0, loss_val)
        parts.append(f'<line x1="{margin}" y1="{y}" x2="{w-margin}" y2="{y}" stroke="#333" stroke-width="0.5"/>')
        parts.append(f'<text x="{margin-5}" y="{y+4}" text-anchor="end" fill="#888" font-size="10">{loss_val:.2f}</text>')

    # Train loss line
    if len(train_losses) > 1:
        points = " ".join(f"{to_xy(i, l)[0]},{to_xy(i, l)[1]}" for i, l in enumerate(train_losses))
        parts.append(f'<polyline points="{points}" fill="none" stroke="#4a90e2" stroke-width="2"/>')

    # Val loss line
    if len(val_losses) > 1:
        points = " ".join(f"{to_xy(i, l)[0]},{to_xy(i, l)[1]}" for i, l in enumerate(val_losses))
        parts.append(f'<polyline points="{points}" fill="none" stroke="#e24a90" stroke-width="2"/>')

    # Legend
    parts.append(f'<rect x="{w-160}" y="30" width="12" height="12" fill="#4a90e2"/>')
    parts.append(f'<text x="{w-144}" y="41" fill="#e8e8f0" font-size="11">Train Loss</text>')
    parts.append(f'<rect x="{w-160}" y="48" width="12" height="12" fill="#e24a90"/>')
    parts.append(f'<text x="{w-144}" y="59" fill="#e8e8f0" font-size="11">Val Loss</text>')

    # Axes labels
    parts.append(f'<text x="{w//2}" y="{h-5}" text-anchor="middle" fill="#888" font-size="11">Epoch</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _generate_confusion_matrix_html(matrix: List[List[int]], class_names: List[str]) -> str:
    """Generate an HTML table for the confusion matrix."""
    if not matrix:
        return "<p>No confusion matrix data.</p>"

    n = len(matrix)
    html = ['<div style="overflow-x:auto;"><table style="border-collapse:collapse;font-size:11px;">']

    # Header row
    html.append("<tr><th style='padding:4px;'>→ Pred</th>")
    for name in class_names[:n]:
        short = name[:6]
        html.append(f'<th style="padding:4px;writing-mode:vertical-rl;transform:rotate(180deg);">{short}</th>')
    html.append("</tr>")

    # Data rows
    max_val = max((max(row) for row in matrix), default=1) or 1
    for i, row in enumerate(matrix):
        name = class_names[i] if i < len(class_names) else f"c{i}"
        html.append(f'<tr><td style="padding:4px;font-weight:bold;">{name[:8]}</td>')
        for j, val in enumerate(row):
            intensity = int(val / max_val * 200) if max_val > 0 else 0
            bg = f"rgba(74,144,226,{intensity/255:.2f})" if i != j else f"rgba(46,204,113,{intensity/255:.2f})"
            html.append(f'<td style="padding:4px;text-align:center;background:{bg};">{val}</td>')
        html.append("</tr>")

    html.append("</table></div>")
    return "\n".join(html)


def generate_html_report(
    metrics: Dict[str, Any],
    history: Dict[str, List[float]],
    dataset_info: Optional[Dict[str, Any]] = None,
    output_path: str = "./training/results/report.html",
) -> str:
    """
    Generate a comprehensive HTML evaluation report.

    Parameters
    ----------
    metrics      : Output from PipelineEvaluator.compute_all()
    history      : Training history from Trainer.train()
    dataset_info : Dataset statistics (sample counts, etc.)
    output_path  : Where to save the HTML file

    Returns
    -------
    Absolute path to the generated report
    """
    from training.schema import RVLCDIP_CLASSES

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # CSS
    css = """
    body { font-family: 'Segoe UI', sans-serif; background: #0d0d1a; color: #e8e8f0; margin: 20px; }
    .container { max-width: 1100px; margin: 0 auto; }
    h1 { color: #4a90e2; border-bottom: 2px solid #4a90e2; padding-bottom: 10px; }
    h2 { color: #7b2fbe; margin-top: 30px; }
    h3 { color: #4a90e2; }
    .card { background: #12122a; border-radius: 12px; padding: 20px; margin: 16px 0; border: 1px solid #222244; }
    .metric { display: inline-block; text-align: center; padding: 12px 24px; margin: 6px; background: #1a1a3a; border-radius: 8px; }
    .metric .value { font-size: 28px; font-weight: bold; color: #4a90e2; }
    .metric .label { font-size: 12px; color: #8888aa; margin-top: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0; }
    th, td { padding: 8px 12px; border: 1px solid #333; text-align: left; }
    th { background: #1a1a3a; color: #4a90e2; }
    .good { color: #2ecc71; }
    .warn { color: #f39c12; }
    .bad { color: #e74c3c; }
    """

    # Build HTML sections
    sections = []

    # ── Aggregate metrics ─────────────────────────────────────
    agg = metrics.get("aggregate", {})
    weighted_f1 = agg.get("weighted_macro_f1", 0.0)
    f1_class = "good" if weighted_f1 >= 0.7 else ("warn" if weighted_f1 >= 0.5 else "bad")

    sections.append(f"""
    <div class="card">
        <h2>📊 Aggregate Performance</h2>
        <div class="metric">
            <div class="value {f1_class}">{weighted_f1:.1%}</div>
            <div class="label">Weighted Macro F1</div>
        </div>
    </div>
    """)

    # ── Text Detection ────────────────────────────────────────
    det = metrics.get("text_detection", {})
    if det.get("tp", 0) + det.get("fp", 0) + det.get("fn", 0) > 0:
        sections.append(f"""
        <div class="card">
            <h2>🔍 Text Detection (ICDAR 2015 / COCO-Text)</h2>
            <div class="metric"><div class="value">{det.get('precision', 0):.1%}</div><div class="label">Precision</div></div>
            <div class="metric"><div class="value">{det.get('recall', 0):.1%}</div><div class="label">Recall</div></div>
            <div class="metric"><div class="value">{det.get('f1', 0):.1%}</div><div class="label">F1-Score</div></div>
            <div class="metric"><div class="value">{det.get('hmean', 0):.1%}</div><div class="label">H-Mean</div></div>
        </div>
        """)

    # ── Document Classification ───────────────────────────────
    cls = metrics.get("doc_classification", {})
    if cls.get("accuracy", 0) > 0:
        cm_html = _generate_confusion_matrix_html(
            cls.get("confusion_matrix", []),
            RVLCDIP_CLASSES,
        )
        sections.append(f"""
        <div class="card">
            <h2>📄 Document Classification (RVL-CDIP)</h2>
            <div class="metric"><div class="value">{cls.get('accuracy', 0):.1%}</div><div class="label">Accuracy</div></div>
            <div class="metric"><div class="value">{cls.get('macro_f1', 0):.1%}</div><div class="label">Macro F1</div></div>
            <h3>Confusion Matrix</h3>
            {cm_html}
        </div>
        """)

    # ── Form Understanding ────────────────────────────────────
    form = metrics.get("form_understanding", {})
    if form.get("entity_f1", 0) > 0:
        per_label = form.get("per_label", {})
        label_rows = "".join(
            f"<tr><td>{name}</td><td>{m.get('precision',0):.1%}</td><td>{m.get('recall',0):.1%}</td><td>{m.get('f1',0):.1%}</td></tr>"
            for name, m in per_label.items()
        )
        sections.append(f"""
        <div class="card">
            <h2>📝 Form Understanding (FUNSD)</h2>
            <div class="metric"><div class="value">{form.get('entity_f1', 0):.1%}</div><div class="label">Entity F1</div></div>
            <div class="metric"><div class="value">{form.get('linking_accuracy', 0):.1%}</div><div class="label">Linking Acc</div></div>
            <table><tr><th>Label</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{label_rows}</table>
        </div>
        """)

    # ── Desktop Text ──────────────────────────────────────────
    desk = metrics.get("desktop_text", {})
    if desk.get("total_chars_evaluated", 0) > 0:
        sections.append(f"""
        <div class="card">
            <h2>🖥️ Desktop Text Recognition (Custom Dataset)</h2>
            <div class="metric"><div class="value">{desk.get('cer', 0):.1%}</div><div class="label">CER</div></div>
            <div class="metric"><div class="value">{desk.get('wer', 0):.1%}</div><div class="label">WER</div></div>
            <div class="metric"><div class="value">{desk.get('avg_confidence', 0):.1%}</div><div class="label">Avg Confidence</div></div>
        </div>
        """)

    # ── Training Curves ───────────────────────────────────────
    curves_svg = _generate_training_curves_svg(history)
    sections.append(f"""
    <div class="card">
        <h2>📈 Training Curves</h2>
        {curves_svg}
    </div>
    """)

    # ── Dataset Info ──────────────────────────────────────────
    if dataset_info:
        ds_rows = "".join(
            f"<tr><td>{name}</td><td>{info.get('count', 0)}</td><td>{info.get('task', 'N/A')}</td></tr>"
            for name, info in dataset_info.items()
        )
        sections.append(f"""
        <div class="card">
            <h2>📦 Dataset Summary</h2>
            <table><tr><th>Dataset</th><th>Samples</th><th>Task</th></tr>{ds_rows}</table>
        </div>
        """)

    # ── Assemble HTML ─────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Pipeline — Evaluation Report</title>
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <h1>🧠 Unified Training Pipeline — Evaluation Report</h1>
        <p style="color:#8888aa;">Generated: {timestamp}</p>
        {"".join(sections)}
        <div class="card" style="text-align:center;color:#8888aa;">
            <p>OS Agent v4 — Multi-Dataset Training Pipeline</p>
        </div>
    </div>
</body>
</html>"""

    # Save
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")

    logger.info("HTML report saved to: %s", path)
    return str(path.resolve())
