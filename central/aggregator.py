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
from sklearn.metrics import accuracy_score, confusion_matrix
import time

from central.byzantine import (
    get_byzantine_aggregator,
    initialize_trust,
    get_trust_score,
    get_all_trust_scores
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
        raise ValueError("No client models provided for KNN aggregation")

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

    logger.info(f"Global KNN model trained in {training_time:.3f}s")

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
            'n_test_samples': eval_metrics['n_samples']
        })

        logger.info(f"KNN evaluation: accuracy={eval_metrics['accuracy']:.4f}")

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
        raise ValueError("No client models provided for DT aggregation")

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

    logger.info(f"Global DT model trained in {training_time:.3f}s")

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
            'n_test_samples': eval_metrics['n_samples']
        })

        logger.info(f"DT evaluation: accuracy={eval_metrics['accuracy']:.4f}")

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
    test_snrs: np.ndarray = None
) -> Dict:
    """
    Evaluate global model on validation/test set.

    Computes overall accuracy, per-SNR accuracy breakdown, and confusion matrix.
    """
    if len(test_features) != len(test_labels):
        raise ValueError("Number of test features must match number of test labels")

    if test_snrs is not None and len(test_snrs) != len(test_labels):
        raise ValueError("Number of SNR values must match number of test samples")

    logger.info(f"Evaluating global model on {len(test_features)} test samples")

    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    conf_matrix = confusion_matrix(test_labels, predictions)

    per_snr_accuracy = {}
    if test_snrs is not None:
        unique_snrs = np.unique(test_snrs)
        for snr in unique_snrs:
            snr_mask = test_snrs == snr
            snr_labels = test_labels[snr_mask]
            snr_predictions = predictions[snr_mask]

            if len(snr_labels) > 0:
                snr_accuracy = accuracy_score(snr_labels, snr_predictions)
                per_snr_accuracy[float(snr)] = snr_accuracy

    return {
        'accuracy': float(accuracy),
        'per_snr_accuracy': per_snr_accuracy,
        'confusion_matrix': conf_matrix,
        'n_samples': len(test_features),
        'predictions': predictions
    }


# ── FedAvg for MLP Neural Networks ───────────────────────────────────────────

def aggregate_mlp_fedavg(
    client_model_paths: List[str],
    n_samples_per_client: List[int],
    test_features: np.ndarray = None,
    test_labels: np.ndarray = None,
) -> Dict:
    """
    FedAvg: Average MLP neural network weights across clients,
    weighted by number of training samples.

    This is the canonical FL aggregation for neural networks (McMahan et al. 2017).
    Unlike data-centric aggregation, FedAvg averages model parameters directly.
    """
    models = []
    for path in client_model_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models.append(pickle.load(f))

    if not models:
        raise ValueError("No MLP models found for FedAvg")

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
