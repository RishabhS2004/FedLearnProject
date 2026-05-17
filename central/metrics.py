"""
Centralized Evaluation Metrics Module for RadioFed

Provides reusable functions for computing, exporting, and persisting
evaluation metrics across the federated learning pipeline.

Metrics computed:
- Accuracy, Precision (macro + weighted), Recall (macro + weighted)
- F1-score (macro + weighted), classification_report (dict + text)
- Per-class breakdown, per-SNR accuracy and F1

Export formats:
- JSON  → out/metrics/metrics_{timestamp}.json
- CSV   → out/metrics/metrics_{timestamp}.csv
- Text  → out/metrics/classification_report_{timestamp}.txt
"""

import os
import json
import csv
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    f1_score as sklearn_f1_score,
)

logger = logging.getLogger("federated_central")


# ── Configuration Defaults ───────────────────────────────────────────────────
MIN_SAMPLES_PER_SNR = 50


# ── Core Metrics Computation ─────────────────────────────────────────────────

def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground-truth labels (1-D array).
        y_pred: Predicted labels (1-D array, same length as y_true).
        class_names: Human-readable class names for the classification report.
                     If None, auto-generated as "Class 0", "Class 1", …

    Returns:
        dict with keys:
            accuracy              – float
            precision_macro       – float
            recall_macro          – float
            f1_macro              – float
            precision_weighted    – float
            recall_weighted       – float
            f1_weighted           – float
            classification_report – dict  (sklearn output_dict=True)
            classification_report_text – str (human-readable table)
            per_class             – dict[str, dict] per-class precision/recall/f1/support
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples."
        )

    accuracy = float(accuracy_score(y_true, y_pred))

    # Macro averages
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    # Weighted averages
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Classification report – dict form
    target_names = class_names if class_names else None
    report_dict = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    # Classification report – text form
    report_text = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
    )

    # Per-class breakdown (precision, recall, f1, support)
    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    per_class = {}
    for idx, label in enumerate(unique_labels):
        name = class_names[idx] if class_names and idx < len(class_names) else f"Class {label}"
        per_class[name] = {
            "precision": float(prec_pc[idx]),
            "recall": float(rec_pc[idx]),
            "f1": float(f1_pc[idx]),
            "support": int(sup_pc[idx]),
        }

    return {
        "accuracy": accuracy,
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "per_class": per_class,
    }


# ── Per-SNR Metrics ──────────────────────────────────────────────────────────

def compute_snr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snrs: np.ndarray,
    class_names: Optional[List[str]] = None,
    min_samples: int = MIN_SAMPLES_PER_SNR,
) -> Dict[str, Dict[float, float]]:
    """
    Compute accuracy and F1-score (macro + weighted) per SNR level.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        snrs: SNR value for each sample (same length as y_true).
        class_names: Optional class names (unused here, kept for API consistency).
        min_samples: Minimum number of samples required to evaluate an SNR level.

    Returns:
        dict with keys:
            per_snr_accuracy  – {snr_float: accuracy_float}
            per_snr_f1_macro  – {snr_float: f1_macro_float}
            per_snr_f1_weighted – {snr_float: f1_weighted_float}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    snrs = np.asarray(snrs)

    if len(y_true) != len(snrs):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"snrs has {len(snrs)} samples."
        )

    per_snr_accuracy = {}
    per_snr_f1_macro = {}
    per_snr_f1_weighted = {}

    for snr in sorted(np.unique(snrs)):
        mask = snrs == snr
        y_t = y_true[mask]
        y_p = y_pred[mask]

        count = len(y_t)
        if count < min_samples:
            logger.warning(
                f"Skipping SNR {snr:.1f} dB: insufficient samples "
                f"({count} < {min_samples} required)"
            )
            continue

        per_snr_accuracy[float(snr)] = float(accuracy_score(y_t, y_p))
        per_snr_f1_macro[float(snr)] = float(
            sklearn_f1_score(y_t, y_p, average="macro", zero_division=0)
        )
        per_snr_f1_weighted[float(snr)] = float(
            sklearn_f1_score(y_t, y_p, average="weighted", zero_division=0)
        )

    return {
        "per_snr_accuracy": per_snr_accuracy,
        "per_snr_f1_macro": per_snr_f1_macro,
        "per_snr_f1_weighted": per_snr_f1_weighted,
    }


# ── Export Helpers ────────────────────────────────────────────────────────────

def save_metrics_json(
    metrics: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Save metrics dictionary as a JSON file.

    Args:
        metrics: Metrics dictionary (must be JSON-serialisable).
        output_path: Destination path (directories created automatically).

    Returns:
        Absolute path of the written file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure numpy types are converted to native Python types
    serialisable = _make_serialisable(metrics)

    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=4)

    logger.info(f"Metrics JSON saved to {output_path}")
    return os.path.abspath(output_path)


def save_metrics_csv(
    metrics: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Save flat (scalar) metrics as a single-row CSV file.

    Non-scalar values (dicts, lists) are skipped. Per-class and per-SNR
    breakdowns are expanded into separate columns.

    Args:
        metrics: Metrics dictionary.
        output_path: Destination path.

    Returns:
        Absolute path of the written file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    flat = _flatten_metrics(metrics)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    logger.info(f"Metrics CSV saved to {output_path}")
    return os.path.abspath(output_path)


def save_classification_report_text(
    report_text: str,
    output_path: str,
) -> str:
    """
    Save the sklearn classification_report text to a plain-text file.

    Args:
        report_text: The report string returned by classification_report().
        output_path: Destination path.

    Returns:
        Absolute path of the written file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(report_text)

    logger.info(f"Classification report saved to {output_path}")
    return os.path.abspath(output_path)


# ── Convenience: compute + save in one call ──────────────────────────────────

def evaluate_and_export(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snrs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    output_dir: str = "out/metrics",
    timestamp: Optional[str] = None,
    min_samples_per_snr: int = MIN_SAMPLES_PER_SNR,
) -> Dict[str, Any]:
    """
    Compute all metrics, save JSON + CSV + classification report text.

    This is a convenience wrapper combining compute_full_metrics(),
    compute_snr_metrics(), and the three save_* functions.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        snrs: Per-sample SNR values (optional).
        class_names: Human-readable class names.
        output_dir: Base directory for output files.
        timestamp: Optional timestamp string; auto-generated if None.

    Returns:
        Combined metrics dict (same as compute_full_metrics + SNR metrics
        plus 'export_paths' key listing saved files).
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Core metrics
    metrics = compute_full_metrics(y_true, y_pred, class_names)

    # 2. Per-SNR metrics (if SNR data available)
    if snrs is not None:
        snr_metrics = compute_snr_metrics(
            y_true, y_pred, snrs, class_names, min_samples=min_samples_per_snr
        )
        metrics.update(snr_metrics)

    # 3. Add metadata
    metrics["timestamp"] = timestamp
    metrics["n_samples"] = int(len(y_true))

    # 4. Export
    export_paths = {}

    json_path = os.path.join(output_dir, f"metrics_{timestamp}.json")
    export_paths["json"] = save_metrics_json(metrics, json_path)

    csv_path = os.path.join(output_dir, f"metrics_{timestamp}.csv")
    export_paths["csv"] = save_metrics_csv(metrics, csv_path)

    report_path = os.path.join(output_dir, f"classification_report_{timestamp}.txt")
    export_paths["classification_report"] = save_classification_report_text(
        metrics["classification_report_text"], report_path
    )

    metrics["export_paths"] = export_paths

    logger.info(
        f"Evaluation complete: accuracy={metrics['accuracy']:.4f}, "
        f"F1_macro={metrics['f1_macro']:.4f}, "
        f"F1_weighted={metrics['f1_weighted']:.4f}"
    )

    return metrics


# ── Internal Helpers ─────────────────────────────────────────────────────────

def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serialisable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested metrics dict into a single-level dict for CSV export.

    Skips large nested objects (classification_report dict, export_paths).
    Expands per_class and per_snr dicts into separate columns.
    """
    flat = {}
    skip_keys = {"classification_report", "classification_report_text", "export_paths"}

    for key, value in metrics.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"

        if key in skip_keys:
            continue
        elif isinstance(value, dict):
            # Expand sub-dicts (per_class, per_snr_accuracy, etc.)
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, dict):
                    # per_class entries have nested precision/recall/f1/support
                    for metric_name, metric_val in sub_val.items():
                        col = f"{full_key}_{sub_key}_{metric_name}"
                        flat[col] = _to_native(metric_val)
                else:
                    col = f"{full_key}_{sub_key}"
                    flat[col] = _to_native(sub_val)
        elif isinstance(value, (list, np.ndarray)):
            # Skip arrays (e.g. confusion matrix) in CSV
            continue
        else:
            flat[full_key] = _to_native(value)

    return flat


def _to_native(val: Any) -> Any:
    """Convert a single numpy scalar to a native Python type."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val
