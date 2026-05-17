"""
Aggregation Logic for Central Server

This module implements aggregation strategies for KNN and Decision Tree models
in federated learning with Byzantine fault tolerance.
Based on the ML approach from amc-rml2016a-updated.ipynb.
"""

import os
import logging
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

from central.byzantine import (
    get_byzantine_aggregator,
    initialize_trust,
    get_trust_score,
    get_all_trust_scores
)
from central.metrics import (
    compute_full_metrics,
    compute_snr_metrics,
    evaluate_and_export,
    MIN_SAMPLES_PER_SNR,
)

logger = logging.getLogger("federated_central")


def aggregate_knn_models(
    client_models_info: List[Dict],
    n_neighbors: int = 5,
    evaluate: bool = True,
    byzantine_filtering: bool = True
) -> Dict:
    """
    Aggregate KNN models from multiple clients by merging training data.

    Strategy: Since KNN is instance-based, we merge all training data from
    clients and retrain a global KNN model on the combined dataset.

    Args:
        client_models_info: List of dicts with 'model_path', 'features_path',
                           'labels_path', and 'n_samples' keys
        n_neighbors: Number of neighbors for the global KNN model
        evaluate: Whether to evaluate the model and collect metrics
        byzantine_filtering: Whether to apply Byzantine fault tolerance

    Returns:
        dict: Aggregation result containing model, metrics, and defense report
    """
    if not client_models_info:
        logger.warning("No client models provided for KNN aggregation (all clients dropped out).")
        return {
            'global_model': None,
            'num_clients': 0,
            'total_samples': 0,
            'byzantine_report': None,
            'evaluation': None
        }

    logger.info(f"Starting KNN aggregation with {len(client_models_info)} clients")

    # Collect all training data from clients
    all_features = []
    all_labels = []
    all_snrs = []
    client_ids = []
    total_samples = 0
    valid_clients = 0

    for client_info in client_models_info:
        try:
            features_path = client_info.get('features_path')
            labels_path = client_info.get('labels_path')
            cid = client_info.get('client_id', f'unknown_{valid_clients}')

            if not features_path or not labels_path:
                logger.warning(f"Client {cid} missing data paths, skipping")
                continue

            if not os.path.exists(features_path) or not os.path.exists(labels_path):
                logger.warning(f"Client {cid} data files not found, skipping")
                continue

            with open(features_path, 'rb') as f:
                features = pickle.load(f)
            with open(labels_path, 'rb') as f:
                labels = pickle.load(f)

            features = np.array(features)
            labels = np.array(labels)

            if len(features) != len(labels):
                logger.warning(f"Client {cid} has mismatched features/labels, skipping")
                continue

            initialize_trust(cid)
            all_features.append(features)
            all_labels.append(labels)
            client_ids.append(cid)
            total_samples += len(features)
            valid_clients += 1

            # Try to load SNR values if available
            snrs_path = features_path.replace('_features.pkl', '_snrs.pkl')
            if os.path.exists(snrs_path):
                try:
                    with open(snrs_path, 'rb') as f:
                        snrs = pickle.load(f)
                    all_snrs.append(np.array(snrs))
                except Exception:
                    pass

            logger.info(f"Loaded {len(features)} samples from client {cid}")

        except Exception as e:
            logger.warning(f"Error loading data from client {client_info.get('client_id', 'unknown')}: {e}")
            continue

    if not all_features:
        raise ValueError("No valid client data could be loaded for KNN aggregation")

    # ── Byzantine Filtering ──
    defense_report = None
    if byzantine_filtering and len(all_features) >= 2:
        try:
            aggregator = get_byzantine_aggregator()
            result = aggregator.filter_and_aggregate(
                all_features, all_labels, client_ids
            )
            merged_features = result['features']
            merged_labels = result['labels']
            defense_report = result['defense_report']
            logger.info(f"Byzantine filtering: {result['defense_report']['accepted_count']}/"
                       f"{result['defense_report']['total_clients']} clients accepted")
        except Exception as e:
            logger.warning(f"Byzantine filtering failed, using all data: {e}")
            merged_features = np.vstack(all_features)
            merged_labels = np.concatenate(all_labels)
    else:
        merged_features = np.vstack(all_features)
        merged_labels = np.concatenate(all_labels)

    merged_snrs = np.concatenate(all_snrs) if all_snrs else None

    logger.info(f"Merged data: {merged_features.shape[0]} samples, {merged_features.shape[1]} features")

    feature_dim = merged_features.shape[1]

    # Split into train/test for evaluation
    if evaluate:
        if merged_snrs is not None and len(merged_snrs) == len(merged_labels):
            X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
                merged_features, merged_labels, merged_snrs,
                test_size=0.2, random_state=42, stratify=merged_labels
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                merged_features, merged_labels,
                test_size=0.2, random_state=42, stratify=merged_labels
            )
            snr_test = None
    else:
        X_train, y_train = merged_features, merged_labels
        X_test, y_test, snr_test = None, None, None

    # Train global KNN model
    global_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    train_start = time.time()
    global_knn.fit(X_train, y_train)
    training_time = time.time() - train_start

    log_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    logger.info(f"[{log_timestamp}] Global KNN model aggregation and training completed in {training_time:.3f}s")

    result = {
        'global_model': global_knn,
        'total_samples': total_samples,
        'num_clients': valid_clients,
        'feature_dim': feature_dim,
        'n_neighbors': n_neighbors,
        'training_time': training_time,
        'model_type': 'knn',
        'trust_scores': get_all_trust_scores(),
        'defense_report': defense_report
    }

    if evaluate and X_test is not None:
        logger.info("Evaluating global KNN model...")
        if snr_test is None:
            snr_test = generate_synthetic_snr_values(len(X_test))

        inference_start = time.time()
        predictions = global_knn.predict(X_test)
        inference_time = time.time() - inference_start
        inference_time_ms_per_sample = (inference_time / len(X_test)) * 1000

        eval_metrics = evaluate_global_model(global_knn, X_test, y_test, snr_test)

        result.update({
            'inference_time_ms_per_sample': inference_time_ms_per_sample,
            'accuracy': eval_metrics['accuracy'],
            'per_snr_accuracy': eval_metrics['per_snr_accuracy'],
            'confusion_matrix': eval_metrics['confusion_matrix'].tolist(),
            'n_test_samples': eval_metrics['n_samples'],
            # ── New metrics from central.metrics ──
            'precision_macro': eval_metrics.get('precision_macro'),
            'recall_macro': eval_metrics.get('recall_macro'),
            'f1_macro': eval_metrics.get('f1_macro'),
            'precision_weighted': eval_metrics.get('precision_weighted'),
            'recall_weighted': eval_metrics.get('recall_weighted'),
            'f1_weighted': eval_metrics.get('f1_weighted'),
            'per_snr_f1_macro': eval_metrics.get('per_snr_f1_macro', {}),
            'per_snr_f1_weighted': eval_metrics.get('per_snr_f1_weighted', {}),
        })

        logger.info(
            f"KNN evaluation: accuracy={eval_metrics['accuracy']:.4f}, "
            f"F1_macro={eval_metrics.get('f1_macro', 0):.4f}, "
            f"F1_weighted={eval_metrics.get('f1_weighted', 0):.4f}"
        )

    return result


def aggregate_dt_models(
    client_models_info: List[Dict],
    evaluate: bool = True,
    byzantine_filtering: bool = True
) -> Dict:
    """
    Aggregate Decision Tree models from multiple clients by merging training data.

    Args:
        client_models_info: List of dicts with data paths
        evaluate: Whether to evaluate the model
        byzantine_filtering: Whether to apply Byzantine fault tolerance

    Returns:
        dict: Aggregation result containing model and metrics
    """
    if not client_models_info:
        logger.warning("No client models provided for DT aggregation (all clients dropped out).")
        return {
            'global_model': None,
            'num_clients': 0,
            'total_samples': 0,
            'byzantine_report': None,
            'evaluation': None
        }

    logger.info(f"Starting DT aggregation with {len(client_models_info)} clients")

    all_features = []
    all_labels = []
    all_snrs = []
    client_ids = []
    total_samples = 0
    valid_clients = 0

    for client_info in client_models_info:
        try:
            features_path = client_info.get('features_path')
            labels_path = client_info.get('labels_path')
            cid = client_info.get('client_id', f'unknown_{valid_clients}')

            if not features_path or not labels_path:
                continue
            if not os.path.exists(features_path) or not os.path.exists(labels_path):
                continue

            with open(features_path, 'rb') as f:
                features = pickle.load(f)
            with open(labels_path, 'rb') as f:
                labels = pickle.load(f)

            features = np.array(features)
            labels = np.array(labels)

            if len(features) != len(labels):
                continue

            initialize_trust(cid)
            all_features.append(features)
            all_labels.append(labels)
            client_ids.append(cid)
            total_samples += len(features)
            valid_clients += 1

            snrs_path = features_path.replace('_features.pkl', '_snrs.pkl')
            if os.path.exists(snrs_path):
                try:
                    with open(snrs_path, 'rb') as f:
                        snrs = pickle.load(f)
                    all_snrs.append(np.array(snrs))
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Error loading DT data from client: {e}")
            continue

    if not all_features:
        raise ValueError("No valid client data for DT aggregation")

    # Byzantine filtering
    defense_report = None
    if byzantine_filtering and len(all_features) >= 2:
        try:
            aggregator = get_byzantine_aggregator()
            result = aggregator.filter_and_aggregate(
                all_features, all_labels, client_ids
            )
            merged_features = result['features']
            merged_labels = result['labels']
            defense_report = result['defense_report']
        except Exception as e:
            logger.warning(f"Byzantine filtering failed for DT: {e}")
            merged_features = np.vstack(all_features)
            merged_labels = np.concatenate(all_labels)
    else:
        merged_features = np.vstack(all_features)
        merged_labels = np.concatenate(all_labels)

    merged_snrs = np.concatenate(all_snrs) if all_snrs else None

    feature_dim = merged_features.shape[1]

    if evaluate:
        if merged_snrs is not None and len(merged_snrs) == len(merged_labels):
            X_train, X_test, y_train, y_test, _, snr_test = train_test_split(
                merged_features, merged_labels, merged_snrs,
                test_size=0.2, random_state=42, stratify=merged_labels
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                merged_features, merged_labels,
                test_size=0.2, random_state=42, stratify=merged_labels
            )
            snr_test = None
    else:
        X_train, y_train = merged_features, merged_labels
        X_test, y_test, snr_test = None, None, None

    # Train global Decision Tree
    global_dt = DecisionTreeClassifier(random_state=42)
    train_start = time.time()
    global_dt.fit(X_train, y_train)
    training_time = time.time() - train_start

    log_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    logger.info(f"[{log_timestamp}] Global DT model aggregation and training completed in {training_time:.3f}s")

    result = {
        'global_model': global_dt,
        'total_samples': total_samples,
        'num_clients': valid_clients,
        'feature_dim': feature_dim,
        'training_time': training_time,
        'model_type': 'dt',
        'trust_scores': get_all_trust_scores(),
        'defense_report': defense_report
    }

    if evaluate and X_test is not None:
        if snr_test is None:
            snr_test = generate_synthetic_snr_values(len(X_test))

        inference_start = time.time()
        predictions = global_dt.predict(X_test)
        inference_time = time.time() - inference_start
        inference_time_ms_per_sample = (inference_time / len(X_test)) * 1000

        eval_metrics = evaluate_global_model(global_dt, X_test, y_test, snr_test)

        result.update({
            'inference_time_ms_per_sample': inference_time_ms_per_sample,
            'accuracy': eval_metrics['accuracy'],
            'per_snr_accuracy': eval_metrics['per_snr_accuracy'],
            'confusion_matrix': eval_metrics['confusion_matrix'].tolist(),
            'n_test_samples': eval_metrics['n_samples'],
            # ── New metrics from central.metrics ──
            'precision_macro': eval_metrics.get('precision_macro'),
            'recall_macro': eval_metrics.get('recall_macro'),
            'f1_macro': eval_metrics.get('f1_macro'),
            'precision_weighted': eval_metrics.get('precision_weighted'),
            'recall_weighted': eval_metrics.get('recall_weighted'),
            'f1_weighted': eval_metrics.get('f1_weighted'),
            'per_snr_f1_macro': eval_metrics.get('per_snr_f1_macro', {}),
            'per_snr_f1_weighted': eval_metrics.get('per_snr_f1_weighted', {}),
        })

        logger.info(
            f"DT evaluation: accuracy={eval_metrics['accuracy']:.4f}, "
            f"F1_macro={eval_metrics.get('f1_macro', 0):.4f}, "
            f"F1_weighted={eval_metrics.get('f1_weighted', 0):.4f}"
        )

    return result


def save_knn_model(model, path: str) -> None:
    """Save KNN model to file using pickle."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"KNN model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save KNN model to {path}: {e}")
        raise IOError(f"Could not save KNN model: {e}") from e


def save_dt_model(model, path: str) -> None:
    """Save Decision Tree model to file using pickle."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"DT model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save DT model to {path}: {e}")
        raise IOError(f"Could not save DT model: {e}") from e


def load_knn_model(path: str):
    """Load KNN model from file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"KNN model file not found: {path}")
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"KNN model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load KNN model from {path}: {e}")
        raise RuntimeError(f"Invalid KNN model file: {path}") from e


def generate_synthetic_snr_values(n_samples: int, snr_range: Tuple[int, int] = (-20, 18)) -> np.ndarray:
    """Generate synthetic SNR values distributed across RadioML SNR levels."""
    snr_levels = list(range(snr_range[0], snr_range[1] + 1, 2))
    snr_values = []
    samples_per_snr = n_samples // len(snr_levels)
    remainder = n_samples % len(snr_levels)

    for i, snr in enumerate(snr_levels):
        count = samples_per_snr + (1 if i < remainder else 0)
        snr_values.extend([snr] * count)

    return np.array(snr_values)


def evaluate_global_model(
    model: object,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_snrs: np.ndarray = None,
    min_samples_per_snr: int = MIN_SAMPLES_PER_SNR,
) -> Dict:
    """
    Evaluate global model on validation/test set.

    Computes overall accuracy, per-SNR accuracy + F1 breakdown,
    confusion matrix, and saves metrics in JSON, CSV, and text formats.

    Returns:
        dict with backward-compatible keys:
            accuracy, precision, recall, f1_score, per_class_accuracy,
            per_snr_accuracy, confusion_matrix, n_samples, predictions
        plus new keys:
            precision_macro, recall_macro, f1_macro,
            precision_weighted, recall_weighted, f1_weighted,
            per_snr_f1_macro, per_snr_f1_weighted,
            classification_report_text
    """
    if len(test_features) != len(test_labels):
        raise ValueError("Number of test features must match number of test labels")

    if test_snrs is not None and len(test_snrs) != len(test_labels):
        raise ValueError("Number of SNR values must match number of test samples")

    logger.info(f"Evaluating global model on {len(test_features)} test samples")

    predictions = model.predict(test_features)
    conf_matrix = confusion_matrix(test_labels, predictions)

    # ── Compute comprehensive metrics via central.metrics ──
    full_metrics = compute_full_metrics(test_labels, predictions)

    # Per-SNR metrics (accuracy + F1)
    per_snr_accuracy = {}
    per_snr_f1_macro = {}
    per_snr_f1_weighted = {}
    if test_snrs is not None:
        snr_metrics = compute_snr_metrics(
            test_labels, predictions, test_snrs, min_samples=min_samples_per_snr
        )
        per_snr_accuracy = snr_metrics['per_snr_accuracy']
        per_snr_f1_macro = snr_metrics['per_snr_f1_macro']
        per_snr_f1_weighted = snr_metrics['per_snr_f1_weighted']

    # Per-class accuracy (backward-compatible format)
    row_sums = conf_matrix.sum(axis=1)
    per_class_accuracy = {}
    for idx, (diag, r_sum) in enumerate(zip(conf_matrix.diagonal(), row_sums)):
        class_name = f"Class {idx}"
        per_class_accuracy[class_name] = float(diag / r_sum) if r_sum > 0 else 0.0

    # ── Persist plots and reports (existing behavior) ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('out/plots', exist_ok=True)
    os.makedirs('out/reports', exist_ok=True)

    # Confusion matrix plot (normalized, indicating in filename as requested)
    class_labels = [f"Class {i}" for i in range(len(conf_matrix))]
    from central.evaluation_plots import plot_confusion_matrix_normalized
    plot_path = plot_confusion_matrix_normalized(
        conf_matrix=conf_matrix,
        class_names=class_labels,
        title="Normalized Confusion Matrix",
        output_path=f"out/plots/confusion_matrix_normalized_{timestamp}.png",
        timestamp=timestamp,
        accuracy=full_metrics["accuracy"],
        f1_macro=full_metrics["f1_macro"],
        f1_weighted=full_metrics["f1_weighted"],
    )

    # ── NEW: Publication-quality plots via central.evaluation_plots ──
    try:
        from central.evaluation_plots import generate_all_evaluation_plots
        eval_plot_paths = generate_all_evaluation_plots(
            conf_matrix=conf_matrix,
            per_snr_accuracy=per_snr_accuracy,
            per_snr_f1_macro=per_snr_f1_macro,
            per_snr_f1_weighted=per_snr_f1_weighted,
            class_names=class_labels,
            timestamp=timestamp,
            accuracy=full_metrics["accuracy"],
            f1_macro=full_metrics["f1_macro"],
            f1_weighted=full_metrics["f1_weighted"],
        )
        logger.info(f"Publication-quality evaluation plots generated: {list(eval_plot_paths.keys())}")
    except Exception as e:
        logger.warning(f"Could not generate evaluation plots: {e}")

    # Legacy eval report (existing format, enriched with new fields)
    report_data = {
        'timestamp': timestamp,
        'accuracy': full_metrics['accuracy'],
        'precision_macro': full_metrics['precision_macro'],
        'recall_macro': full_metrics['recall_macro'],
        'f1_score_macro': full_metrics['f1_macro'],
        'precision_weighted': full_metrics['precision_weighted'],
        'recall_weighted': full_metrics['recall_weighted'],
        'f1_score_weighted': full_metrics['f1_weighted'],
        'per_class_accuracy': per_class_accuracy,
        'per_snr_accuracy': per_snr_accuracy,
        'per_snr_f1_macro': per_snr_f1_macro,
        'per_snr_f1_weighted': per_snr_f1_weighted,
        'classification_report': full_metrics['classification_report'],
    }
    report_path = f'out/reports/eval_report_{timestamp}.json'
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=4)

    # ── NEW: Export metrics via central.metrics (JSON + CSV + report text) ──
    export_metrics = dict(full_metrics)
    export_metrics['per_snr_accuracy'] = per_snr_accuracy
    export_metrics['per_snr_f1_macro'] = per_snr_f1_macro
    export_metrics['per_snr_f1_weighted'] = per_snr_f1_weighted
    export_metrics['per_class_accuracy'] = per_class_accuracy
    try:
        from central.metrics import save_metrics_json, save_metrics_csv, save_classification_report_text
        os.makedirs('out/metrics', exist_ok=True)
        save_metrics_json(export_metrics, f'out/metrics/metrics_{timestamp}.json')
        save_metrics_csv(export_metrics, f'out/metrics/metrics_{timestamp}.csv')
        save_classification_report_text(
            full_metrics['classification_report_text'],
            f'out/metrics/classification_report_{timestamp}.txt'
        )
    except Exception as e:
        logger.warning(f"Could not export metrics to out/metrics/: {e}")

    logger.info(f"Evaluation report saved to {report_path}")
    logger.info(f"Confusion matrix plot saved to {plot_path}")

    # ── Return backward-compatible dict + new keys ──
    return {
        # Backward-compatible keys (existing consumers use these)
        'accuracy': full_metrics['accuracy'],
        'precision': full_metrics['precision_macro'],
        'recall': full_metrics['recall_macro'],
        'f1_score': full_metrics['f1_macro'],
        'per_class_accuracy': per_class_accuracy,
        'per_snr_accuracy': per_snr_accuracy,
        'confusion_matrix': conf_matrix,
        'n_samples': len(test_features),
        'predictions': predictions,
        # New enriched keys
        'precision_macro': full_metrics['precision_macro'],
        'recall_macro': full_metrics['recall_macro'],
        'f1_macro': full_metrics['f1_macro'],
        'precision_weighted': full_metrics['precision_weighted'],
        'recall_weighted': full_metrics['recall_weighted'],
        'f1_weighted': full_metrics['f1_weighted'],
        'per_snr_f1_macro': per_snr_f1_macro,
        'per_snr_f1_weighted': per_snr_f1_weighted,
        'classification_report_text': full_metrics['classification_report_text'],
    }


# ── FedAvg for MLP Neural Networks ───────────────────────────────────────────

def aggregate_mlp_fedavg(
    client_model_paths: List[str] = None,
    n_samples_per_client: List[int] = None,
    test_features: np.ndarray = None,
    test_labels: np.ndarray = None,
    client_models: List = None,
) -> Dict:
    """
    FedAvg: Average MLP neural network weights across clients,
    weighted by number of training samples.

    This is the canonical FL aggregation for neural networks (McMahan et al. 2017).
    Unlike data-centric aggregation, FedAvg averages model parameters directly.
    """
    models = []
    if client_models is not None:
        models = client_models
    elif client_model_paths is not None:
        for path in client_model_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    models.append(pickle.load(f))

    if not models:
        logger.warning("No MLP models found for FedAvg (all clients dropped out).")
        return {
            'global_model': None,
            'num_clients': 0,
            'total_samples': 0,
            'evaluation': None
        }

    # Verify all models are MLPClassifier
    from sklearn.neural_network import MLPClassifier
    for m in models:
        if not isinstance(m, MLPClassifier):
            raise ValueError(f"FedAvg requires MLPClassifier, got {type(m)}")

    # Weighted average of coefficients and intercepts
    total_samples = sum(n_samples_per_client[:len(models)])
    weights = [n / total_samples for n in n_samples_per_client[:len(models)]]

    # Average coefs_ and intercepts_
    avg_coefs = []
    avg_intercepts = []
    for layer_idx in range(len(models[0].coefs_)):
        layer_avg = sum(w * m.coefs_[layer_idx] for w, m in zip(weights, models))
        avg_coefs.append(layer_avg)
    for layer_idx in range(len(models[0].intercepts_)):
        layer_avg = sum(w * m.intercepts_[layer_idx] for w, m in zip(weights, models))
        avg_intercepts.append(layer_avg)

    # Create averaged model (clone structure from first model)
    global_model = pickle.loads(pickle.dumps(models[0]))
    global_model.coefs_ = avg_coefs
    global_model.intercepts_ = avg_intercepts

    result = {
        'global_model': global_model,
        'num_clients': len(models),
        'total_samples': total_samples,
        'model_type': 'mlp_fedavg',
        'aggregation_method': 'fedavg',
        'layer_sizes': [c.shape for c in avg_coefs],
    }

    # Evaluate if test data provided
    if test_features is not None and test_labels is not None:
        predictions = global_model.predict(test_features)
        result['accuracy'] = float(accuracy_score(test_labels, predictions))
        result['confusion_matrix'] = confusion_matrix(test_labels, predictions).tolist()
        logger.info(f"FedAvg MLP: {result['num_clients']} clients, accuracy={result['accuracy']:.4f}")

    return result
