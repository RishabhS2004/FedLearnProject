import logging
import os
from pathlib import Path
from typing import Dict, Any


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the central server.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("federated_central")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
   
    if not logger.handlers:
        logger.addHandler(handler)
        
        # Add file handler
        os.makedirs("out/logs", exist_ok=True)
        file_handler = logging.FileHandler("out/logs/central.log")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_directories() -> None:
    """
    Create necessary directories for the central server if they don't exist.
    Creates: central/model_store/, data/ and out/ directories
    """
    directories = [
        "central/model_store",
        "data",
        "out/checkpoints/central",
        "out/checkpoints/client",
        "out/logs",
        "out/plots",
        "out/plots/confusion_matrix",
        "out/plots/byzantine",
        "out/metrics",
        "out/reports",
        "out/predictions"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured directory exists: {directory}")


def validate_weights(weights: Dict[str, Any]) -> bool:
    """
    Validate that weights dictionary has the expected format.
    
    Args:
        weights: Dictionary containing model weights
    
    Returns:
        True if weights are valid, False otherwise
    """
    if not isinstance(weights, dict):
        return False
    
    
    for key, value in weights.items():
        if not hasattr(value, 'shape'):  
            return False
    
    return True
