"""Generic evaluation metrics (no domain conclusions)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .. import __version__
from ..adapters.generic import GenericDatasetAdapter
from ..adapters.base import FieldMap
from ..core.io_util import atomic_write_text, utc_now_iso, write_json
from ..core.privacy import redact_value
from ..domain import json_safe
from ..report.html import render_html_report


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[，。！？、；：,.!?;:\"'“”‘’（）()\[\]{}]", "", t)
    return t


def confusion_binary(
    y_true: Sequence[Any], y_pred: Sequence[Any], positive: Any
) -> Dict[str, float]:
    """Binary confusion metrics without sklearn."""
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == positive and t == positive:
            tp += 1
        elif p == positive and t != positive:
            fp += 1
        elif p != positive and t != positive:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sensitivity": recall,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "accuracy": (tp + tn) / max(len(y_true), 1),
    }


def run_eval(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate predictions vs references with configurable fields."""
    pred_path = cfg.get("predictions")
    if not pred_path:
        raise ValueError("predictions path required")
    fm = FieldMap(
        prediction=str(cfg.get("prediction_field") or "prediction"),
        reference=str(cfg.get("reference_field") or "reference"),
        group_id=str(cfg.get("group_id_field") or "group_id"),
    )
    adapter = GenericDatasetAdapter(fm)
    records = list(adapter.iter_objects(Path(pred_path), sample_limit=cfg.get("sample_limit")))

    refs_by_id: Dict[str, Dict[str, Any]] = {}
    if cfg.get("references"):
        ref_recs = adapter.iter_objects(Path(cfg["references"]))
        for i, r in enumerate(ref_recs):
            key = str(r.get(fm.group_id) or r.get("id") or i)
            refs_by_id[key] = r

    pred_field = fm.prediction
    ref_field = fm.reference
    keywords = [k.strip() for k in (cfg.get("keywords") or []) if str(k).strip()]
    total = len(records)
    missing_pred = empty_pred = exact_match = norm_match = 0
    text_pairs = keyword_hits = keyword_total = 0
    y_true: List[Any] = []
    y_pred: List[Any] = []

    for i, sample in enumerate(records):
        pred_val = sample.get(pred_field, sample.get("output"))
        if pred_val is None:
            missing_pred += 1
            continue
        pred_text = str(pred_val)
        if pred_text.strip() == "":
            empty_pred += 1
        ref_val = sample.get(ref_field)
        if ref_val is None and refs_by_id:
            key = str(sample.get(fm.group_id) or sample.get("id") or i)
            ref_sample = refs_by_id.get(key)
            if ref_sample:
                ref_val = (
                    ref_sample.get(ref_field) or ref_sample.get("answer") or ref_sample.get("label")
                )
        if ref_val is not None:
            ref_text = str(ref_val)
            text_pairs += 1
            if pred_text == ref_text:
                exact_match += 1
            if normalize_text(pred_text) == normalize_text(ref_text):
                norm_match += 1
            if keywords:
                keyword_total += 1
                ref_norm = normalize_text(ref_text)
                pred_norm = normalize_text(pred_text)
                needed = [k for k in keywords if normalize_text(k) in ref_norm]
                if not needed or all(normalize_text(k) in pred_norm for k in needed):
                    keyword_hits += 1
        pl = sample.get(cfg.get("predicted_label_field") or "predicted_label")
        tl = sample.get(cfg.get("label_field") or "label")
        if pl is not None and tl is not None:
            y_pred.append(pl)
            y_true.append(tl)

    classification: Dict[str, Any] = {}
    if y_true and y_pred:
        unique = sorted(set(y_true) | set(y_pred), key=lambda x: str(x))
        if len(unique) == 2:
            positive = None
            for cand in (1, "1", True, "true", "positive", "pos", "yes"):
                if cand in unique:
                    positive = cand
                    break
            if positive is None:
                positive = unique[-1]
            classification = {
                "type": "binary",
                **confusion_binary(y_true, y_pred, positive),
                "positive_label": positive,
            }
        else:
            correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            classification = {
                "type": "multiclass",
                "accuracy": correct / max(len(y_true), 1),
                "num_labels": len(unique),
            }

    metrics = {
        "total_samples": total,
        "missing_predictions": missing_pred,
        "empty_predictions": empty_pred,
        "text_pairs": text_pairs,
        "exact_match_rate": exact_match / text_pairs if text_pairs else None,
        "normalized_match_rate": norm_match / text_pairs if text_pairs else None,
        "keyword_hit_rate": keyword_hits / keyword_total if keyword_total else None,
        "classification": classification,
    }
    has_references = text_pairs > 0 or bool(y_true)
    evaluation_mode = "reference_based" if has_references else "prediction_only"
    overall = (
        "FAIL"
        if total == 0
        else ("WARN" if missing_pred or empty_pred or not has_references else "PASS")
    )
    report: Dict[str, Any] = {
        "tool": "train_guard",
        "command": "eval",
        "version": __version__,
        "timestamp": utc_now_iso(),
        "overall_status": overall,
        "evaluation_mode": evaluation_mode,
        "metrics": metrics,
        "disclaimer": (
            "Prediction-only inspection; no references were available for correctness metrics."
            if not has_references
            else "Metrics reflect text/label consistency only and do not imply domain validity."
        ),
    }
    report = dict(json_safe(redact_value(report)))
    report_dir = Path(cfg.get("report_dir") or "reports/eval")
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "eval_report.json", report, overwrite=True)
    html_doc = render_html_report(
        "Train Guard — Eval Report",
        [{"title": "Status", "value": overall, "status": overall}],
        [
            {
                "title": "Metrics",
                "headers": ["Key", "Value"],
                "rows": [[k, v] for k, v in metrics.items() if k != "classification"],
            }
        ],
        report["disclaimer"],
    )
    atomic_write_text(report_dir / "eval_report.html", html_doc, overwrite=True)
    return report
