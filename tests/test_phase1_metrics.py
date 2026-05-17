"""
Phase 1 — Metrics Infrastructure Verification Script

Exercises all new functions in central/metrics.py and verifies
the output files are correctly generated.
"""

import numpy as np
import os
import sys
import json
import csv

# Ensure project root is on sys.path and set cwd
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

from central.metrics import (
    compute_full_metrics,
    compute_snr_metrics,
    evaluate_and_export,
    save_metrics_json,
    save_metrics_csv,
    save_classification_report_text,
)

def test_compute_full_metrics():
    """Test core metrics computation."""
    print("=" * 60)
    print("  TEST: compute_full_metrics")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 200
    n_classes = 4
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    noise_idx = np.random.choice(n_samples, size=30, replace=False)
    y_pred[noise_idx] = np.random.randint(0, n_classes, 30)

    class_names = ["AM-DSB", "AM-SSB", "WBFM", "GFSK"]

    m = compute_full_metrics(y_true, y_pred, class_names)

    print(f"  Accuracy:             {m['accuracy']:.4f}")
    print(f"  Precision (macro):    {m['precision_macro']:.4f}")
    print(f"  Recall (macro):       {m['recall_macro']:.4f}")
    print(f"  F1 (macro):           {m['f1_macro']:.4f}")
    print(f"  Precision (weighted): {m['precision_weighted']:.4f}")
    print(f"  Recall (weighted):    {m['recall_weighted']:.4f}")
    print(f"  F1 (weighted):        {m['f1_weighted']:.4f}")
    print(f"  Per-class keys:       {list(m['per_class'].keys())}")
    print()
    print("  Classification Report (text):")
    print(m["classification_report_text"])

    # Assertions
    assert 0 < m["accuracy"] <= 1.0, f"Bad accuracy: {m['accuracy']}"
    assert "f1_weighted" in m, "Missing f1_weighted"
    assert "precision_weighted" in m, "Missing precision_weighted"
    assert "classification_report" in m, "Missing classification_report dict"
    assert "classification_report_text" in m, "Missing classification_report_text"
    assert len(m["per_class"]) == n_classes, f"Expected {n_classes} classes, got {len(m['per_class'])}"

    print("  [PASS] compute_full_metrics\n")


def test_compute_snr_metrics():
    """Test per-SNR metrics computation."""
    print("=" * 60)
    print("  TEST: compute_snr_metrics")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 200
    n_classes = 4
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    noise_idx = np.random.choice(n_samples, size=30, replace=False)
    y_pred[noise_idx] = np.random.randint(0, n_classes, 30)

    snrs = np.random.choice([-20, -10, 0, 10, 18], len(y_true))
    snr_m = compute_snr_metrics(y_true, y_pred, snrs, min_samples=1)

    print(f"  SNR levels: {sorted(snr_m['per_snr_accuracy'].keys())}")
    for snr in sorted(snr_m["per_snr_accuracy"].keys()):
        print(
            f"    SNR {snr:+6.1f} dB: "
            f"acc={snr_m['per_snr_accuracy'][snr]:.4f}, "
            f"F1_macro={snr_m['per_snr_f1_macro'][snr]:.4f}, "
            f"F1_weighted={snr_m['per_snr_f1_weighted'][snr]:.4f}"
        )

    # Assertions
    assert len(snr_m["per_snr_accuracy"]) == 5, f"Expected 5 SNR levels"
    assert len(snr_m["per_snr_f1_macro"]) == 5, f"Expected 5 SNR F1 levels"
    assert len(snr_m["per_snr_f1_weighted"]) == 5, f"Expected 5 SNR F1 weighted levels"

    print("\n  [PASS] compute_snr_metrics\n")


def test_evaluate_and_export():
    """Test combined evaluate + export pipeline."""
    print("=" * 60)
    print("  TEST: evaluate_and_export")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 200
    n_classes = 4
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    noise_idx = np.random.choice(n_samples, size=30, replace=False)
    y_pred[noise_idx] = np.random.randint(0, n_classes, 30)
    snrs = np.random.choice([-20, -10, 0, 10, 18], len(y_true))
    class_names = ["AM-DSB", "AM-SSB", "WBFM", "GFSK"]

    result = evaluate_and_export(
        y_true, y_pred, snrs, class_names,
        min_samples_per_snr=1,
        output_dir="out/metrics",
        timestamp="test_run_001",
    )

    print(f"  Result keys: {sorted(result.keys())}")
    print(f"  Export paths:")
    for key, path in result["export_paths"].items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"    {key:25s}: {os.path.basename(path)} ({size} bytes, exists={exists})")

    # Verify JSON file
    json_path = result["export_paths"]["json"]
    with open(json_path) as f:
        data = json.load(f)
    print(f"\n  JSON keys: {sorted(data.keys())}")
    assert "f1_weighted" in data, "JSON missing f1_weighted"
    assert "per_snr_f1_macro" in data, "JSON missing per_snr_f1_macro"
    assert "per_snr_accuracy" in data, "JSON missing per_snr_accuracy"

    # Verify CSV file
    csv_path = result["export_paths"]["csv"]
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    print(f"  CSV columns ({len(row)}): {list(row.keys())[:10]}...")
    assert "accuracy" in row, "CSV missing accuracy column"
    assert "f1_weighted" in row, "CSV missing f1_weighted column"

    # Verify classification report text
    report_path = result["export_paths"]["classification_report"]
    with open(report_path) as f:
        report = f.read()
    print(f"  Classification report file: {len(report)} chars")
    assert "precision" in report.lower(), "Report missing precision"
    assert "recall" in report.lower(), "Report missing recall"

    print("\n  [PASS] evaluate_and_export\n")


def test_aggregator_integration():
    """Test that aggregator.evaluate_global_model returns new keys."""
    print("=" * 60)
    print("  TEST: aggregator integration")
    print("=" * 60)

    from sklearn.neighbors import KNeighborsClassifier
    from central.aggregator import evaluate_global_model, generate_synthetic_snr_values

    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 8).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X[:200], y[:200])

    X_test = X[200:]
    y_test = y[200:]
    snrs = generate_synthetic_snr_values(len(X_test))

    result = evaluate_global_model(model, X_test, y_test, snrs, min_samples_per_snr=1)

    # Check backward-compatible keys
    assert "accuracy" in result, "Missing backward-compat key: accuracy"
    assert "precision" in result, "Missing backward-compat key: precision"
    assert "recall" in result, "Missing backward-compat key: recall"
    assert "f1_score" in result, "Missing backward-compat key: f1_score"
    assert "confusion_matrix" in result, "Missing backward-compat key: confusion_matrix"
    assert "per_snr_accuracy" in result, "Missing backward-compat key: per_snr_accuracy"
    assert "predictions" in result, "Missing backward-compat key: predictions"
    assert hasattr(result["confusion_matrix"], "tolist"), "confusion_matrix should be numpy array"

    # Check NEW keys
    assert "f1_weighted" in result, "Missing new key: f1_weighted"
    assert "f1_macro" in result, "Missing new key: f1_macro"
    assert "precision_weighted" in result, "Missing new key: precision_weighted"
    assert "recall_weighted" in result, "Missing new key: recall_weighted"
    assert "per_snr_f1_macro" in result, "Missing new key: per_snr_f1_macro"
    assert "per_snr_f1_weighted" in result, "Missing new key: per_snr_f1_weighted"
    assert "classification_report_text" in result, "Missing new key: classification_report_text"

    print(f"  accuracy:          {result['accuracy']:.4f}")
    print(f"  precision (legacy): {result['precision']:.4f}")
    print(f"  f1_score (legacy):  {result['f1_score']:.4f}")
    print(f"  f1_macro (new):     {result['f1_macro']:.4f}")
    print(f"  f1_weighted (new):  {result['f1_weighted']:.4f}")
    print(f"  SNR accuracy keys:  {sorted(result['per_snr_accuracy'].keys())[:5]}...")
    print(f"  SNR F1_macro keys:  {sorted(result['per_snr_f1_macro'].keys())[:5]}...")

    # Verify that out/metrics/ files were generated
    metrics_files = [f for f in os.listdir("out/metrics") if not f.startswith("test_")]
    print(f"  out/metrics/ files: {len(metrics_files)}")
    for f in sorted(metrics_files)[-6:]:
        print(f"    {f}")

    print("\n  [PASS] aggregator integration\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PHASE 1 VERIFICATION: Metrics Infrastructure")
    print("=" * 60 + "\n")

    test_compute_full_metrics()
    test_compute_snr_metrics()
    test_evaluate_and_export()
    test_aggregator_integration()

    print("=" * 60)
    print("  ALL PHASE 1 TESTS PASSED")
    print("=" * 60)
