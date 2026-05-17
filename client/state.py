"""
State Management for Client Application

This module handles configuration management and metrics tracking for the
federated learning client. It provides functions to load/save configuration
and persist training metrics.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


def load_config(config_path: str = "./client/config.json") -> Dict:
    """
    Load configuration from config.json file.
    
    Args:
        config_path (str): Path to the configuration file
    
    Returns:
        dict: Configuration dictionary with client settings
    
    Raises:
        FileNotFoundError: If config file does not exist
        json.JSONDecodeError: If config file is not valid JSON
        ValueError: If required configuration fields are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    
    _validate_config(config)
    
    return config


def save_config(config: Dict, config_path: str = "./client/config.json") -> None:
    """
    Save configuration to config.json file.
    
    Args:
        config (dict): Configuration dictionary to save
        config_path (str): Path to the configuration file
    
    Raises:
        IOError: If the file cannot be written
        ValueError: If required configuration fields are missing
    """
    
    _validate_config(config)
    
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def save_metrics(
    metrics: Dict[str, Any],
    metrics_path: str = "out/checkpoints/client/metrics.json"
) -> None:
    """
    Save training metrics to local/metrics.json file.
    
    Args:
        metrics (dict): Dictionary containing training metrics
        metrics_path (str): Path to the metrics file
    
    Raises:
        IOError: If the file cannot be written
    """
    
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    
    if 'timestamp' not in metrics:
        metrics['timestamp'] = datetime.now().isoformat()
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(metrics_path: str = "out/checkpoints/client/metrics.json") -> Optional[Dict]:
    """
    Load training metrics from local/metrics.json file.
    
    Args:
        metrics_path (str): Path to the metrics file
    
    Returns:
        dict or None: Metrics dictionary or None if file does not exist
    
    Raises:
        json.JSONDecodeError: If metrics file is not valid JSON
    """
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def _validate_config(config: Dict) -> None:
    """
    Validate that required configuration fields are present.
    
    Args:
        config (dict): Configuration dictionary to validate
    
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = ['client_id', 'server_url', 'dataset_path', 'local_model_path']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required configuration field missing: {field}")
        
        if not isinstance(config[field], str):
            raise ValueError(f"Configuration field '{field}' must be a string")
        
        if not config[field].strip():
            raise ValueError(f"Configuration field '{field}' cannot be empty")
    
    
    if 'training' in config:
        training = config['training']
        
        if not isinstance(training, dict):
            raise ValueError("'training' configuration must be a dictionary")
        
        
        training_fields = {
            'epochs': int,
            'batch_size': int,
            'learning_rate': (int, float)
        }
        
        for field, expected_type in training_fields.items():
            if field in training:
                if not isinstance(training[field], expected_type):
                    raise ValueError(
                        f"Training parameter '{field}' must be of type {expected_type}"
                    )
                
                
                if field in ['epochs', 'batch_size'] and training[field] <= 0:
                    raise ValueError(f"Training parameter '{field}' must be positive")
                
                if field == 'learning_rate' and training[field] <= 0:
                    raise ValueError("Learning rate must be positive")


def get_config_value(config: Dict, key: str, default: Any = None) -> Any:
    """
    Get a configuration value with optional default.
    
    Args:
        config (dict): Configuration dictionary
        key (str): Configuration key (supports nested keys with dot notation)
        default: Default value if key is not found
    
    Returns:
        Configuration value or default
    
    Example:
        get_config_value(config, 'training.epochs', 10)
    """
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


def update_config_value(config: Dict, key: str, value: Any) -> Dict:
    """
    Update a configuration value (supports nested keys).
    
    Args:
        config (dict): Configuration dictionary
        key (str): Configuration key (supports nested keys with dot notation)
        value: New value to set
    
    Returns:
        dict: Updated configuration dictionary
    
    Example:
        update_config_value(config, 'training.epochs', 20)
    """
    keys = key.split('.')
    current = config
    
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    current[keys[-1]] = value
    return config
