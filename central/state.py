"""
State Management for Central Server

This module handles configuration management and client registry for the
federated learning central server. It provides functions to load/save
configuration and track client upload information.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from threading import Lock


# In-memory client registry with thread-safe access
_client_registry: Dict[str, Dict] = {}
_registry_lock = Lock()

# In-memory aggregation results storage
_aggregation_results: Dict[str, Dict] = {}
_aggregation_lock = Lock()

# In-memory metrics storage for dashboard
_metrics_history: List[Dict] = []
_metrics_lock = Lock()

# Before/after aggregation state tracking
_before_aggregation_state: Dict[str, Dict] = {}
_after_aggregation_state: Dict[str, Dict] = {}
_aggregation_state_lock = Lock()

# Confusion matrix storage
_confusion_matrices: Dict[str, Dict] = {}
_confusion_matrix_lock = Lock()

# Per-SNR accuracy storage
_per_snr_accuracy: Dict[str, Dict] = {}
_per_snr_lock = Lock()

# Current training round
_current_round: int = 0
_round_lock = Lock()

# Auto-aggregation state
_auto_aggregation_state: Dict = {
    'enabled': True,
    'threshold': 2,
    'pending_uploads': 0,
    'current_round': 0,
    'clients_uploaded_this_round': [],
    'last_aggregation_time': None
}
_auto_aggregation_lock = Lock()


def load_config(config_path: str = "./central/config.json") -> Dict:
    """
    Load configuration from config.json file.
    
    Args:
        config_path (str): Path to the configuration file
    
    Returns:
        dict: Configuration dictionary with server settings
    
    Raises:
        FileNotFoundError: If config file does not exist
        json.JSONDecodeError: If config file is not valid JSON
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def save_config(config: Dict, config_path: str = "./central/config.json") -> None:
    """
    Save configuration to config.json file.
    
    Args:
        config (dict): Configuration dictionary to save
        config_path (str): Path to the configuration file
    
    Raises:
        IOError: If the file cannot be written
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def register_client_connection(client_id: str) -> None:
    """
    Register a client connection (before training).
    
    Args:
        client_id (str): Unique identifier for the client
    """
    with _registry_lock:
        if client_id not in _client_registry:
            _client_registry[client_id] = {
                'connected_at': datetime.now().isoformat(),
                'status': 'connected',
                'last_upload': None,
                'n_samples': 0,
                'weights_path': None,
                'training': False
            }
        else:
            _client_registry[client_id]['status'] = 'connected'
            _client_registry[client_id]['connected_at'] = datetime.now().isoformat()


def update_client_training_status(client_id: str, training: bool) -> None:
    """
    Update client training status.
    
    Args:
        client_id (str): Unique identifier for the client
        training (bool): Whether client is currently training
    """
    with _registry_lock:
        if client_id in _client_registry:
            _client_registry[client_id]['training'] = training
            _client_registry[client_id]['status'] = 'training' if training else 'idle'


def register_client_upload(
    client_id: str,
    n_samples: int,
    weights_path: str,
    model_type: str = 'neural',
    model_path: str = None,
    features_path: str = None,
    labels_path: str = None
) -> None:
    """
    Register or update a client's upload information in the registry.
    
    Args:
        client_id (str): Unique identifier for the client
        n_samples (int): Number of samples used for training
        weights_path (str): Path where the client's weights are stored
        model_type (str): Type of model ('neural', 'knn', 'dt')
        model_path (str): Path to serialized model file (for traditional ML)
        features_path (str): Path to training features (for KNN aggregation)
        labels_path (str): Path to training labels (for KNN aggregation)
    """
    with _registry_lock:
        if client_id not in _client_registry:
            _client_registry[client_id] = {}
        
        _client_registry[client_id].update({
            'last_upload': datetime.now().isoformat(),
            'n_samples': n_samples,
            'weights_path': weights_path,
            'model_type': model_type,
            'model_path': model_path,
            'features_path': features_path,
            'labels_path': labels_path,
            'status': 'weights_uploaded',
            'training': False
        })


def get_client_status() -> List[Dict]:
    """
    Retrieve status information for all registered clients.
    
    Returns:
        list: List of dictionaries containing client information
              Each dict contains: client_id, last_upload, n_samples, weights_path
    """
    with _registry_lock:
        client_list = []
        for client_id, info in _client_registry.items():
            client_info = {
                'client_id': client_id,
                'last_upload': info['last_upload'],
                'n_samples': info['n_samples'],
                'weights_path': info['weights_path']
            }
            client_list.append(client_info)
        
        return client_list


def get_client_info(client_id: str) -> Optional[Dict]:
    """
    Get information for a specific client.
    
    Args:
        client_id (str): Unique identifier for the client
    
    Returns:
        dict or None: Client information dictionary or None if not found
    """
    with _registry_lock:
        if client_id in _client_registry:
            return {
                'client_id': client_id,
                **_client_registry[client_id]
            }
        return None


def get_all_client_weights() -> List[Dict]:
    """
    Get weights information for all registered clients.
    
    Returns:
        list: List of dictionaries with client_id, weights_path, n_samples,
              model_type, model_path, features_path, labels_path
    """
    with _registry_lock:
        weights_info = []
        for client_id, info in _client_registry.items():
            weights_info.append({
                'client_id': client_id,
                'weights_path': info.get('weights_path'),
                'n_samples': info.get('n_samples', 0),
                'model_type': info.get('model_type', 'neural'),
                'model_path': info.get('model_path'),
                'features_path': info.get('features_path'),
                'labels_path': info.get('labels_path')
            })
        return weights_info


def clear_client_registry() -> None:
    """
    Clear all clients from the registry.
    
    This is useful for testing or resetting the server state.
    """
    with _registry_lock:
        _client_registry.clear()


def get_registry_stats() -> Dict:
    """
    Get statistics about the client registry.
    
    Returns:
        dict: Statistics including total clients and total samples
    """
    with _registry_lock:
        total_clients = len(_client_registry)
        total_samples = sum(info['n_samples'] for info in _client_registry.values())
        
        return {
            'total_clients': total_clients,
            'total_samples': total_samples,
            'client_ids': list(_client_registry.keys())
        }



def store_aggregation_result(
    model_type: str,
    result: Dict,
    timestamp: str
) -> None:
    """
    Store aggregation results for a specific model type.
    
    Args:
        model_type: Type of model ('neural', 'knn', 'dt')
        result: Aggregation result dictionary
        timestamp: ISO format timestamp
    """
    with _aggregation_lock:
        if model_type not in _aggregation_results:
            _aggregation_results[model_type] = []
        
        _aggregation_results[model_type].append({
            'result': result,
            'timestamp': timestamp
        })


def get_latest_aggregation_result(model_type: str) -> Optional[Dict]:
    """
    Get the most recent aggregation result for a model type.
    
    Args:
        model_type: Type of model ('neural', 'knn', 'dt')
    
    Returns:
        dict or None: Latest aggregation result or None if not found
    """
    with _aggregation_lock:
        if model_type in _aggregation_results and _aggregation_results[model_type]:
            return _aggregation_results[model_type][-1]
        return None


def get_all_aggregation_results(model_type: str) -> List[Dict]:
    """
    Get all aggregation results for a model type.
    
    Args:
        model_type: Type of model ('neural', 'knn', 'dt')
    
    Returns:
        list: List of aggregation results
    """
    with _aggregation_lock:
        return _aggregation_results.get(model_type, [])


def clear_aggregation_results() -> None:
    """Clear all aggregation results."""
    with _aggregation_lock:
        _aggregation_results.clear()


# ============================================================================
# Dashboard Metrics Management Functions
# ============================================================================

def store_training_metrics(
    client_id: str,
    model_type: str,
    round_num: int,
    training_time: float,
    inference_time: float,
    train_accuracy: float,
    test_accuracy: float,
    n_samples: int,
    timestamp: Optional[str] = None
) -> None:
    """
    Store training metrics from a client.
    
    Args:
        client_id: Unique identifier for the client
        model_type: Type of model ('knn' or 'dt')
        round_num: Current training round number
        training_time: Training time in seconds
        inference_time: Inference time in milliseconds per sample
        train_accuracy: Training accuracy (0-1)
        test_accuracy: Test accuracy (0-1)
        n_samples: Number of training samples
        timestamp: ISO format timestamp (auto-generated if None)
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    metrics = {
        'client_id': client_id,
        'model_type': model_type,
        'round': round_num,
        'training_time': training_time,
        'inference_time_ms_per_sample': inference_time,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'n_samples': n_samples,
        'timestamp': timestamp
    }
    
    with _metrics_lock:
        _metrics_history.append(metrics)


def store_aggregation_results(
    model_type: str,
    round_num: int,
    num_clients: int,
    total_samples: int,
    global_accuracy: float,
    per_snr_accuracy: Dict[int, float],
    confusion_matrix: List[List[int]],
    stage: str = 'after',
    timestamp: Optional[str] = None
) -> None:
    """
    Store aggregation results for dashboard display.
    
    Args:
        model_type: Type of model ('knn' or 'dt')
        round_num: Current training round number
        num_clients: Number of clients that participated
        total_samples: Total number of samples across all clients
        global_accuracy: Overall accuracy of global model (0-1)
        per_snr_accuracy: Dictionary mapping SNR values to accuracy
        confusion_matrix: 2D list representing confusion matrix
        stage: 'before' or 'after' aggregation
        timestamp: ISO format timestamp (auto-generated if None)
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    result = {
        'model_type': model_type,
        'round': round_num,
        'num_clients': num_clients,
        'total_samples': total_samples,
        'global_accuracy': global_accuracy,
        'per_snr_accuracy': per_snr_accuracy,
        'confusion_matrix': confusion_matrix,
        'timestamp': timestamp
    }
    
    with _aggregation_state_lock:
        if stage == 'before':
            _before_aggregation_state[model_type] = result
        else:
            _after_aggregation_state[model_type] = result
    
    # Also store confusion matrix separately for easy access
    with _confusion_matrix_lock:
        key = f"{model_type}_{stage}_round_{round_num}"
        _confusion_matrices[key] = {
            'matrix': confusion_matrix,
            'timestamp': timestamp
        }
    
    # Store per-SNR accuracy separately
    with _per_snr_lock:
        key = f"{model_type}_{stage}_round_{round_num}"
        _per_snr_accuracy[key] = {
            'accuracy': per_snr_accuracy,
            'timestamp': timestamp
        }


def get_metrics_history(
    model_type: Optional[str] = None,
    client_id: Optional[str] = None,
    round_num: Optional[int] = None
) -> List[Dict]:
    """
    Retrieve metrics history with optional filtering.
    
    Args:
        model_type: Filter by model type ('knn' or 'dt')
        client_id: Filter by client ID
        round_num: Filter by round number
    
    Returns:
        List of metrics dictionaries matching the filters
    """
    with _metrics_lock:
        results = list(_metrics_history)
    
    # Apply filters
    if model_type is not None:
        results = [m for m in results if m['model_type'] == model_type]
    
    if client_id is not None:
        results = [m for m in results if m['client_id'] == client_id]
    
    if round_num is not None:
        results = [m for m in results if m['round'] == round_num]
    
    return results


def get_latest_metrics(model_type: Optional[str] = None) -> Dict:
    """
    Get the most recent metrics for each client.
    
    Args:
        model_type: Filter by model type ('knn' or 'dt')
    
    Returns:
        Dictionary mapping client_id to their latest metrics
    """
    with _metrics_lock:
        metrics = list(_metrics_history)
    
    if model_type is not None:
        metrics = [m for m in metrics if m['model_type'] == model_type]
    
    # Group by client_id and get latest for each
    latest_by_client = {}
    for metric in metrics:
        client_id = metric['client_id']
        if client_id not in latest_by_client:
            latest_by_client[client_id] = metric
        else:
            # Compare timestamps
            if metric['timestamp'] > latest_by_client[client_id]['timestamp']:
                latest_by_client[client_id] = metric
    
    return latest_by_client


def get_aggregation_state(stage: str = 'after') -> Dict[str, Dict]:
    """
    Get before or after aggregation state for all model types.
    
    Args:
        stage: 'before' or 'after' aggregation
    
    Returns:
        Dictionary mapping model_type to aggregation results
    """
    with _aggregation_state_lock:
        if stage == 'before':
            return dict(_before_aggregation_state)
        else:
            return dict(_after_aggregation_state)


def get_confusion_matrix(
    model_type: str,
    stage: str = 'after',
    round_num: Optional[int] = None
) -> Optional[List[List[int]]]:
    """
    Get confusion matrix for a specific model and stage.
    
    Args:
        model_type: Type of model ('knn' or 'dt')
        stage: 'before' or 'after' aggregation
        round_num: Specific round number (latest if None)
    
    Returns:
        Confusion matrix as 2D list or None if not found
    """
    with _confusion_matrix_lock:
        if round_num is None:
            # Find latest round for this model and stage
            matching_keys = [k for k in _confusion_matrices.keys() 
                           if k.startswith(f"{model_type}_{stage}_round_")]
            if not matching_keys:
                return None
            # Sort by round number and get latest
            latest_key = sorted(matching_keys, 
                              key=lambda k: int(k.split('_')[-1]))[-1]
            return _confusion_matrices[latest_key]['matrix']
        else:
            key = f"{model_type}_{stage}_round_{round_num}"
            if key in _confusion_matrices:
                return _confusion_matrices[key]['matrix']
            return None


def get_per_snr_accuracy(
    model_type: str,
    stage: str = 'after',
    round_num: Optional[int] = None
) -> Optional[Dict[int, float]]:
    """
    Get per-SNR accuracy for a specific model and stage.
    
    Args:
        model_type: Type of model ('knn' or 'dt')
        stage: 'before' or 'after' aggregation
        round_num: Specific round number (latest if None)
    
    Returns:
        Dictionary mapping SNR to accuracy or None if not found
    """
    with _per_snr_lock:
        if round_num is None:
            # Find latest round for this model and stage
            matching_keys = [k for k in _per_snr_accuracy.keys() 
                           if k.startswith(f"{model_type}_{stage}_round_")]
            if not matching_keys:
                return None
            # Sort by round number and get latest
            latest_key = sorted(matching_keys, 
                              key=lambda k: int(k.split('_')[-1]))[-1]
            return _per_snr_accuracy[latest_key]['accuracy']
        else:
            key = f"{model_type}_{stage}_round_{round_num}"
            if key in _per_snr_accuracy:
                return _per_snr_accuracy[key]['accuracy']
            return None


def get_current_round() -> int:
    """
    Get the current training round number.
    
    Returns:
        Current round number
    """
    with _round_lock:
        return _current_round


def increment_round() -> int:
    """
    Increment and return the current training round number.
    
    Returns:
        New round number
    """
    with _round_lock:
        global _current_round
        _current_round += 1
        return _current_round


def set_round(round_num: int) -> None:
    """
    Set the current training round number.
    
    Args:
        round_num: Round number to set
    """
    with _round_lock:
        global _current_round
        _current_round = round_num


def clear_metrics() -> None:
    """
    Clear all metrics storage (useful for testing or reset).
    """
    with _metrics_lock:
        _metrics_history.clear()
    
    with _aggregation_state_lock:
        _before_aggregation_state.clear()
        _after_aggregation_state.clear()
    
    with _confusion_matrix_lock:
        _confusion_matrices.clear()
    
    with _per_snr_lock:
        _per_snr_accuracy.clear()
    
    with _round_lock:
        global _current_round
        _current_round = 0


def get_dashboard_summary() -> Dict:
    """
    Get a comprehensive summary of all metrics for dashboard display.
    
    Returns:
        Dictionary containing:
        - current_round: Current training round
        - connected_clients: List of connected client IDs
        - metrics_count: Number of stored metrics
        - before_aggregation: Before aggregation state
        - after_aggregation: After aggregation state
        - latest_metrics: Latest metrics per client
    """
    with _round_lock:
        current_round = _current_round
    
    with _registry_lock:
        connected_clients = list(_client_registry.keys())
    
    with _metrics_lock:
        metrics_count = len(_metrics_history)
    
    with _aggregation_state_lock:
        before_agg = dict(_before_aggregation_state)
        after_agg = dict(_after_aggregation_state)
    
    latest_metrics = get_latest_metrics()
    
    return {
        'current_round': current_round,
        'connected_clients': connected_clients,
        'num_connected_clients': len(connected_clients),
        'metrics_count': metrics_count,
        'before_aggregation': before_agg,
        'after_aggregation': after_agg,
        'latest_metrics': latest_metrics
    }


# ============================================================================
# Auto-Aggregation State Management Functions
# ============================================================================

def get_auto_aggregation_state() -> Dict:
    """
    Get the current auto-aggregation state.
    
    Returns:
        dict: Auto-aggregation state containing:
            - enabled: Whether auto-aggregation is enabled
            - threshold: Number of clients required to trigger aggregation
            - pending_uploads: Current number of pending uploads
            - current_round: Current aggregation round
            - clients_uploaded_this_round: List of client IDs that uploaded
            - last_aggregation_time: ISO timestamp of last aggregation
    """
    with _auto_aggregation_lock:
        return dict(_auto_aggregation_state)


def save_auto_aggregation_state(state: Dict) -> None:
    """
    Save the auto-aggregation state.
    
    Args:
        state: Dictionary containing auto-aggregation state fields
    """
    with _auto_aggregation_lock:
        global _auto_aggregation_state
        _auto_aggregation_state.update(state)


def validate_auto_aggregation_config(config: Dict) -> Dict:
    """
    Validate auto-aggregation configuration parameters.
    
    Validates that:
    - auto_aggregation_enabled is a boolean
    - auto_aggregation_threshold is a positive integer
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        dict: Validated configuration with corrected values
    
    Raises:
        ValueError: If configuration values are invalid and cannot be corrected
    """
    validated_config = dict(config)
    
    # Validate enabled flag
    enabled = config.get('auto_aggregation_enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError(
            f"auto_aggregation_enabled must be a boolean (true/false), got: {type(enabled).__name__}"
        )
    validated_config['auto_aggregation_enabled'] = enabled
    
    # Validate threshold
    threshold = config.get('auto_aggregation_threshold', 2)
    if not isinstance(threshold, int):
        raise ValueError(
            f"auto_aggregation_threshold must be an integer, got: {type(threshold).__name__}"
        )
    if threshold < 0:
        raise ValueError(
            f"auto_aggregation_threshold must be a positive integer, got: {threshold}"
        )
    validated_config['auto_aggregation_threshold'] = threshold
    
    return validated_config


def initialize_auto_aggregation_state(config_path: str = "./central/config.json") -> None:
    """
    Initialize auto-aggregation state from configuration file.
    Loads enabled flag and threshold value from config and sets default values.
    
    Args:
        config_path: Path to the configuration file
    
    Raises:
        ValueError: If configuration validation fails
    """
    global _auto_aggregation_state
    
    try:
        config = load_config(config_path)
        
        # Validate configuration
        validated_config = validate_auto_aggregation_config(config)
        
        enabled = validated_config.get('auto_aggregation_enabled', True)
        threshold = validated_config.get('auto_aggregation_threshold', 2)
        
        with _auto_aggregation_lock:
            _auto_aggregation_state = {
                'enabled': enabled,
                'threshold': threshold,
                'pending_uploads': 0,
                'current_round': 0,
                'clients_uploaded_this_round': [],
                'last_aggregation_time': None
            }
    except ValueError as e:
        # Re-raise validation errors
        raise
    except Exception as e:
        # If config loading fails, use defaults
        with _auto_aggregation_lock:
            _auto_aggregation_state = {
                'enabled': True,
                'threshold': 2,
                'pending_uploads': 0,
                'current_round': 0,
                'clients_uploaded_this_round': [],
                'last_aggregation_time': None
            }


def get_auto_aggregation_threshold() -> int:
    """
    Get the current auto-aggregation threshold.
    
    Returns:
        int: Number of clients required to trigger aggregation
    """
    with _auto_aggregation_lock:
        return _auto_aggregation_state['threshold']


def set_auto_aggregation_threshold(threshold: int) -> None:
    """
    Set the auto-aggregation threshold.
    
    Args:
        threshold: Number of clients required to trigger aggregation
    """
    if not isinstance(threshold, int) or threshold < 0:
        raise ValueError("Threshold must be a non-negative integer")
    
    with _auto_aggregation_lock:
        _auto_aggregation_state['threshold'] = threshold


def is_auto_aggregation_enabled() -> bool:
    """
    Check if auto-aggregation is enabled.
    
    Returns:
        bool: True if auto-aggregation is enabled
    """
    with _auto_aggregation_lock:
        return _auto_aggregation_state['enabled']


def set_auto_aggregation_enabled(enabled: bool) -> None:
    """
    Enable or disable auto-aggregation.
    
    Args:
        enabled: True to enable, False to disable
    """
    with _auto_aggregation_lock:
        _auto_aggregation_state['enabled'] = enabled


def get_pending_uploads_count() -> int:
    """
    Get the current number of pending uploads.
    
    Returns:
        int: Number of clients that have uploaded in current round
    """
    with _auto_aggregation_lock:
        return _auto_aggregation_state['pending_uploads']


def get_clients_uploaded_this_round() -> List[str]:
    """
    Get the list of clients that have uploaded in the current round.
    
    Returns:
        list: List of client IDs
    """
    with _auto_aggregation_lock:
        return list(_auto_aggregation_state['clients_uploaded_this_round'])


def track_client_upload(client_id: str) -> None:
    """
    Track that a client has uploaded weights in the current round.
    Thread-safe function that maintains the list of clients uploaded
    and increments the pending uploads counter.
    
    Args:
        client_id: Unique identifier for the client
    """
    with _auto_aggregation_lock:
        # Only add if not already in the list for this round
        if client_id not in _auto_aggregation_state['clients_uploaded_this_round']:
            _auto_aggregation_state['clients_uploaded_this_round'].append(client_id)
            _auto_aggregation_state['pending_uploads'] = len(_auto_aggregation_state['clients_uploaded_this_round'])


def should_trigger_aggregation() -> bool:
    """
    Check if auto-aggregation should be triggered based on current state.
    
    Checks two conditions:
    1. Auto-aggregation is enabled
    2. Pending uploads count has reached or exceeded the threshold
    
    Returns:
        bool: True if aggregation should be triggered, False otherwise
    """
    with _auto_aggregation_lock:
        enabled = _auto_aggregation_state['enabled']
        pending = _auto_aggregation_state['pending_uploads']
        threshold = _auto_aggregation_state['threshold']
        
        # Return True only if enabled and threshold is met
        return enabled and pending >= threshold


# ============================================================================
# Historical Metrics Storage Functions
# ============================================================================

# Path to metrics history file
METRICS_HISTORY_PATH = "./central/metrics_history.json"

# In-memory metrics history cache
_historical_metrics: Dict = {
    'rounds': []
}
_historical_metrics_lock = Lock()


def initialize_metrics_history() -> None:
    """
    Initialize the metrics history data structure.
    Creates the metrics_history.json file if it doesn't exist,
    or loads existing history from file.
    """
    global _historical_metrics
    
    with _historical_metrics_lock:
        if os.path.exists(METRICS_HISTORY_PATH):
            # Load existing history
            try:
                with open(METRICS_HISTORY_PATH, 'r') as f:
                    _historical_metrics = json.load(f)
                    
                # Ensure the structure is correct
                if 'rounds' not in _historical_metrics:
                    _historical_metrics['rounds'] = []
            except (json.JSONDecodeError, IOError) as e:
                # If file is corrupted, start fresh
                _historical_metrics = {'rounds': []}
        else:
            # Create new history structure
            _historical_metrics = {'rounds': []}
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(METRICS_HISTORY_PATH), exist_ok=True)
            
            # Save initial empty structure
            with open(METRICS_HISTORY_PATH, 'w') as f:
                json.dump(_historical_metrics, f, indent=2)


def load_metrics_history_from_file() -> Dict:
    """
    Load metrics history from persistent storage.
    
    Returns:
        dict: Metrics history dictionary with 'rounds' list
    
    Raises:
        FileNotFoundError: If history file doesn't exist
        json.JSONDecodeError: If file is corrupted
    """
    if not os.path.exists(METRICS_HISTORY_PATH):
        raise FileNotFoundError(f"Metrics history file not found: {METRICS_HISTORY_PATH}")
    
    with open(METRICS_HISTORY_PATH, 'r') as f:
        history = json.load(f)
    
    return history


def save_metrics_history_to_file(history: Dict) -> None:
    """
    Save metrics history to persistent storage.
    
    Args:
        history: Metrics history dictionary to save
    
    Raises:
        IOError: If file cannot be written
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(METRICS_HISTORY_PATH), exist_ok=True)
    
    with open(METRICS_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)


def capture_current_metrics() -> Dict:
    """
    Capture metrics from all uploaded client models before aggregation.
    
    Collects accuracy from all uploaded client models, computes per-SNR accuracy
    if data is available, and generates confusion matrices. This provides a
    baseline for comparison after aggregation.
    
    Returns:
        dict: Before-aggregation metrics containing:
            - knn_accuracy: Average KNN accuracy across clients (if available)
            - dt_accuracy: Average DT accuracy across clients (if available)
            - per_snr_accuracy: Dict mapping SNR to average accuracy
            - confusion_matrix: Aggregated confusion matrix
            - num_clients: Number of clients with uploaded models
            - timestamp: ISO format timestamp
    
    Raises:
        ValueError: If no client models are available
    """
    import numpy as np
    import pickle
    from sklearn.metrics import accuracy_score, confusion_matrix as compute_confusion_matrix
    
    # Get all client weights info
    client_weights = get_all_client_weights()
    
    if not client_weights:
        raise ValueError("No client models available for metrics capture")
    
    # Initialize metrics accumulators
    knn_accuracies = []
    all_per_snr_accuracy = {}
    all_predictions = []
    all_labels = []
    num_clients = 0
    
    for client_info in client_weights:
        try:
            model_type = client_info.get('model_type', 'knn')
            
            # Only process KNN models
            if model_type != 'knn':
                continue
            
            # Load model
            model_path = client_info.get('model_path')
            if not model_path or not os.path.exists(model_path):
                continue
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load test data
            features_path = client_info.get('features_path')
            labels_path = client_info.get('labels_path')
            
            if not features_path or not labels_path:
                continue
            if not os.path.exists(features_path) or not os.path.exists(labels_path):
                continue
            
            with open(features_path, 'rb') as f:
                features = pickle.load(f)
            with open(labels_path, 'rb') as f:
                labels = pickle.load(f)
            
            # Use a portion for testing (20%)
            features = np.array(features)
            labels = np.array(labels)
            n_test = len(features) // 5
            X_test = features[:n_test]
            y_test = labels[:n_test]
            
            # Get predictions
            predictions = model.predict(X_test)
            
            # Compute accuracy
            accuracy = accuracy_score(y_test, predictions)
            knn_accuracies.append(accuracy)
            
            # Store predictions and labels for confusion matrix
            all_predictions.extend(predictions)
            all_labels.extend(y_test)
            
            # Try to compute per-SNR accuracy
            snrs_path = features_path.replace('_features.pkl', '_snrs.pkl')
            if os.path.exists(snrs_path):
                try:
                    with open(snrs_path, 'rb') as f:
                        snrs = pickle.load(f)
                    snrs = np.array(snrs)[:n_test]
                    
                    unique_snrs = np.unique(snrs)
                    for snr in unique_snrs:
                        snr_mask = snrs == snr
                        snr_labels = y_test[snr_mask]
                        snr_predictions = predictions[snr_mask]
                        
                        if len(snr_labels) > 0:
                            snr_accuracy = accuracy_score(snr_labels, snr_predictions)
                            snr_key = float(snr)
                            if snr_key not in all_per_snr_accuracy:
                                all_per_snr_accuracy[snr_key] = []
                            all_per_snr_accuracy[snr_key].append(snr_accuracy)
                except:
                    pass
            
            num_clients += 1
            
        except Exception as e:
            # Skip clients with errors
            continue
    
    if num_clients == 0:
        raise ValueError("No valid client models could be evaluated")
    
    # Compute average metrics
    knn_avg_accuracy = float(np.mean(knn_accuracies)) if knn_accuracies else 0.0
    
    # Average per-SNR accuracy across clients
    per_snr_avg = {}
    for snr, accuracies in all_per_snr_accuracy.items():
        per_snr_avg[snr] = float(np.mean(accuracies))
    
    # Compute confusion matrix
    conf_matrix = []
    if all_predictions and all_labels:
        conf_matrix = compute_confusion_matrix(all_labels, all_predictions).tolist()
    
    return {
        'knn_accuracy': knn_avg_accuracy,
        'per_snr_accuracy': per_snr_avg,
        'confusion_matrix': conf_matrix,
        'num_clients': num_clients,
        'timestamp': datetime.now().isoformat()
    }


def evaluate_global_model() -> Dict:
    """
    Evaluate the global model after aggregation.
    
    Loads the global model after aggregation and evaluates it on a validation/test set.
    Computes overall accuracy, per-SNR accuracy, and generates confusion matrix.
    
    Returns:
        dict: After-aggregation metrics containing:
            - knn_accuracy: Global KNN model accuracy (if available)
            - dt_accuracy: Global DT model accuracy (if available)
            - per_snr_accuracy: Dict mapping SNR to accuracy
            - confusion_matrix: Confusion matrix as list
            - timestamp: ISO format timestamp
    
    Raises:
        ValueError: If no global model is available for evaluation
    """
    import numpy as np
    import pickle
    from sklearn.metrics import accuracy_score, confusion_matrix as compute_confusion_matrix
    
    # Paths to global models
    knn_model_path = "out/checkpoints/central/global_knn_model.pkl"
    if not os.path.exists(knn_model_path):
        knn_model_path = "./central/model_store/global_knn_model.pkl"
        
    dt_model_path = "out/checkpoints/central/global_dt_ensemble.pkl"
    if not os.path.exists(dt_model_path):
        dt_model_path = "./central/model_store/global_dt_ensemble.pkl"
    
    knn_accuracy = 0.0
    per_snr_accuracy = {}
    conf_matrix = []
    
    # Collect test data from clients
    client_weights = get_all_client_weights()
    all_test_features = []
    all_test_labels = []
    all_test_snrs = []
    
    for client_info in client_weights:
        try:
            features_path = client_info.get('features_path')
            labels_path = client_info.get('labels_path')
            
            if not features_path or not labels_path:
                continue
            if not os.path.exists(features_path) or not os.path.exists(labels_path):
                continue
            
            with open(features_path, 'rb') as f:
                features = pickle.load(f)
            with open(labels_path, 'rb') as f:
                labels = pickle.load(f)
            
            # Use a portion for testing (20%)
            features = np.array(features)
            labels = np.array(labels)
            n_test = len(features) // 5
            all_test_features.append(features[:n_test])
            all_test_labels.append(labels[:n_test])
            
            # Try to load SNR values
            snrs_path = features_path.replace('_features.pkl', '_snrs.pkl')
            if os.path.exists(snrs_path):
                try:
                    with open(snrs_path, 'rb') as f:
                        snrs = pickle.load(f)
                    all_test_snrs.append(np.array(snrs)[:n_test])
                except:
                    pass
        except:
            continue
    
    if not all_test_features:
        raise ValueError("No test data available for global model evaluation")
    
    # Merge test data
    X_test = np.vstack(all_test_features)
    y_test = np.concatenate(all_test_labels)
    snr_test = np.concatenate(all_test_snrs) if all_test_snrs else None
    
    all_predictions = []
    all_labels_for_cm = []
    
    # Evaluate KNN model if available
    if os.path.exists(knn_model_path):
        try:
            with open(knn_model_path, 'rb') as f:
                knn_model = pickle.load(f)
            
            predictions = knn_model.predict(X_test)
            knn_accuracy = float(accuracy_score(y_test, predictions))
            
            all_predictions.extend(predictions)
            all_labels_for_cm.extend(y_test)
            
            # Compute per-SNR accuracy for KNN
            if snr_test is not None:
                unique_snrs = np.unique(snr_test)
                for snr in unique_snrs:
                    snr_mask = snr_test == snr
                    snr_labels = y_test[snr_mask]
                    snr_predictions = predictions[snr_mask]
                    
                    if len(snr_labels) > 0:
                        snr_accuracy = accuracy_score(snr_labels, snr_predictions)
                        per_snr_accuracy[float(snr)] = snr_accuracy
        except Exception as e:
            pass
    
    # Compute confusion matrix
    if all_predictions and all_labels_for_cm:
        conf_matrix = compute_confusion_matrix(all_labels_for_cm, all_predictions).tolist()
    
    if knn_accuracy == 0.0:
        raise ValueError("No global KNN model available for evaluation")
    
    return {
        'knn_accuracy': knn_accuracy,
        'per_snr_accuracy': per_snr_accuracy,
        'confusion_matrix': conf_matrix,
        'timestamp': datetime.now().isoformat()
    }


def store_aggregation_round(before_metrics: Dict, after_metrics: Dict) -> None:
    """
    Store metrics for a completed aggregation round in historical storage.
    
    Combines before and after metrics, calculates improvement percentages,
    adds round number and timestamp, and appends to history. Keeps only
    the last 50 rounds to manage storage size.
    
    Args:
        before_metrics: Metrics captured before aggregation
        after_metrics: Metrics captured after aggregation
    
    Raises:
        IOError: If history cannot be saved to file
    """
    with _historical_metrics_lock:
        # Get current round number
        with _auto_aggregation_lock:
            round_num = _auto_aggregation_state['current_round']
        
        # Calculate improvement percentage
        knn_improvement = 0.0
        
        if before_metrics.get('knn_accuracy', 0) > 0:
            knn_improvement = after_metrics.get('knn_accuracy', 0) - before_metrics.get('knn_accuracy', 0)
        
        # Create round data
        round_data = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'num_clients': before_metrics.get('num_clients', 0),
            'before': {
                'knn_accuracy': before_metrics.get('knn_accuracy', 0.0),
                'per_snr_accuracy': before_metrics.get('per_snr_accuracy', {})
            },
            'after': {
                'knn_accuracy': after_metrics.get('knn_accuracy', 0.0),
                'per_snr_accuracy': after_metrics.get('per_snr_accuracy', {})
            },
            'improvement': {
                'knn': knn_improvement
            }
        }
        
        # Append to history
        _historical_metrics['rounds'].append(round_data)
        
        # Keep only last 50 rounds
        if len(_historical_metrics['rounds']) > 50:
            _historical_metrics['rounds'] = _historical_metrics['rounds'][-50:]
        
        # Save to persistent storage
        save_metrics_history_to_file(_historical_metrics)


def get_historical_metrics_history(last_n: int = 10) -> Dict:
    """
    Retrieve the last N rounds of metrics from historical storage.
    
    Args:
        last_n: Number of most recent rounds to retrieve (default: 10)
    
    Returns:
        dict: Dictionary containing 'rounds' list with last N rounds
    
    Raises:
        FileNotFoundError: If history file doesn't exist
        json.JSONDecodeError: If history file is corrupted
    """
    with _historical_metrics_lock:
        # Ensure history is initialized
        if not _historical_metrics.get('rounds'):
            # Try to load from file
            if os.path.exists(METRICS_HISTORY_PATH):
                try:
                    loaded_history = load_metrics_history_from_file()
                    _historical_metrics.update(loaded_history)
                except Exception as e:
                    # If loading fails, return empty history
                    return {'rounds': []}
        
        # Return last N rounds
        rounds = _historical_metrics.get('rounds', [])
        return {
            'rounds': rounds[-last_n:] if len(rounds) > last_n else rounds
        }


def get_accuracy_trends() -> Dict:
    """
    Get accuracy trends for plotting historical performance.
    
    Extracts round numbers and accuracy values (before/after) for
    KNN model from the last 10 rounds.
    
    Returns:
        dict: Dictionary containing:
            - rounds: List of round numbers
            - knn_before: List of KNN accuracy before aggregation
            - knn_after: List of KNN accuracy after aggregation
    """
    history = get_historical_metrics_history(last_n=10)
    
    rounds = []
    knn_before = []
    knn_after = []
    
    for round_data in history.get('rounds', []):
        rounds.append(round_data['round'])
        knn_before.append(round_data['before']['knn_accuracy'])
        knn_after.append(round_data['after']['knn_accuracy'])
    
    return {
        'rounds': rounds,
        'knn_before': knn_before,
        'knn_after': knn_after
    }


def get_latest_round_metrics() -> Optional[Dict]:
    """
    Get metrics from the most recent aggregation round.
    
    Returns:
        dict or None: Latest round metrics or None if no history exists
            Contains: round, timestamp, num_clients, before, after, improvement
    """
    history = get_historical_metrics_history(last_n=1)
    
    rounds = history.get('rounds', [])
    if not rounds:
        return None
    
    return rounds[-1]


def reset_aggregation_state() -> None:
    """
    Reset the aggregation state after a successful aggregation.
    
    Clears the list of clients that uploaded in the current round,
    resets pending uploads counter to 0, increments the current round,
    and updates the last aggregation time.
    """
    with _auto_aggregation_lock:
        _auto_aggregation_state['clients_uploaded_this_round'] = []
        _auto_aggregation_state['pending_uploads'] = 0
        _auto_aggregation_state['current_round'] += 1
        _auto_aggregation_state['last_aggregation_time'] = datetime.now().isoformat()
