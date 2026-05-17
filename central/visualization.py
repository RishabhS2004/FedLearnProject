"""
Visualization Utilities for AMC Dashboard

This module provides visualization functions for the AMC federated learning dashboard,
matching the style and format of the reference notebook (knndtamc-amc-rml2016a-updated.ipynb).

All functions return matplotlib figures or pandas DataFrames compatible with Gradio.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import logging


logger = logging.getLogger(__name__)


class PlotCache:
    """
    Cache for generated plots to improve dashboard performance.
    
    Stores last 10 rounds of plots and invalidates cache when new aggregation completes.
    """
    
    def __init__(self, max_rounds: int = 10):
        """
        Initialize plot cache.
        
        Args:
            max_rounds: Maximum number of rounds to cache
        """
        self.max_rounds = max_rounds
        self._cache = {}
        self._current_round = 0
    
    def get(self, key: str, round_num: Optional[int] = None) -> Optional[plt.Figure]:
        """
        Get cached plot.
        
        Args:
            key: Cache key (e.g., 'confusion_matrix_knn')
            round_num: Round number (uses current round if None)
        
        Returns:
            Cached figure or None if not found
        """
        if round_num is None:
            round_num = self._current_round
        
        cache_key = f"{key}_round_{round_num}"
        return self._cache.get(cache_key)
    
    def set(self, key: str, figure: plt.Figure, round_num: Optional[int] = None) -> None:
        """
        Store plot in cache.
        
        Args:
            key: Cache key
            figure: Matplotlib figure to cache
            round_num: Round number (uses current round if None)
        """
        if round_num is None:
            round_num = self._current_round
        
        cache_key = f"{key}_round_{round_num}"
        self._cache[cache_key] = figure
        
        
        self._cleanup_old_rounds()
    
    def invalidate(self, round_num: Optional[int] = None) -> None:
        """
        Invalidate cache for a specific round or all rounds.
        
        Args:
            round_num: Round number to invalidate (invalidates all if None)
        """
        if round_num is None:
            self._cache.clear()
            logger.info("Plot cache cleared")
        else:
            keys_to_remove = [k for k in self._cache.keys() if f"_round_{round_num}" in k]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Plot cache invalidated for round {round_num}")
    
    def set_current_round(self, round_num: int) -> None:
        """
        Set current round number.
        
        Args:
            round_num: Current round number
        """
        self._current_round = round_num
    
    def _cleanup_old_rounds(self) -> None:
        """Remove plots from rounds older than max_rounds."""
        if self._current_round > self.max_rounds:
            min_round = self._current_round - self.max_rounds
            keys_to_remove = []
            
            for key in self._cache.keys():
                
                try:
                    round_str = key.split('_round_')[-1]
                    round_num = int(round_str)
                    if round_num < min_round:
                        keys_to_remove.append(key)
                except (ValueError, IndexError):
                    continue
            
            for key in keys_to_remove:
                del self._cache[key]
            
            if keys_to_remove:
                logger.info(f"Removed {len(keys_to_remove)} old plots from cache")



_plot_cache = PlotCache(max_rounds=10)



def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Generate confusion matrix heatmap using seaborn.
    
    Creates a heatmap visualization of the confusion matrix with annotations
    showing the count of predictions for each class combination.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names (e.g., ['AM', 'FM'])
        title: Plot title
    
    Returns:
        Matplotlib figure compatible with Gradio gr.Plot()
    
    Example:
        >>> y_true = np.array([0, 1, 0, 1, 1])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> fig = plot_confusion_matrix(y_true, y_pred, ['AM', 'FM'], 'KNN Confusion Matrix')
    """
    
    cm = sklearn_confusion_matrix(y_true, y_pred)
    
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar=True,
        square=True
    )
    
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    
    plt.tight_layout()
    
    return fig




def plot_accuracy_vs_snr(
    snr_values: List[float],
    knn_acc: List[float],
    baseline_acc: List[float],
    title: str = "KNN Model Accuracy vs SNR"
) -> plt.Figure:
    """
    Generate accuracy vs SNR line plot for KNN model.
    
    Creates a line plot comparing baseline and KNN accuracy
    across different SNR levels. Uses different markers and line styles for
    each curve.
    
    Args:
        snr_values: List of SNR values in dB (e.g., [-20, -18, ..., 18])
        knn_acc: KNN accuracy values (%)
        baseline_acc: Baseline accuracy values (%)
        title: Plot title
    
    Returns:
        Matplotlib figure compatible with Gradio gr.Plot()
    
    Example:
        >>> snr_values = list(range(-20, 20, 2))
        >>> baseline = [50.0] * len(snr_values)
        >>> knn_acc = [48.0 + i*2.5 for i in range(len(snr_values))]
        >>> fig = plot_accuracy_vs_snr(snr_values, knn_acc, baseline)
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    
    ax.plot(
        snr_values,
        baseline_acc,
        'k--',
        marker='o',
        label='Baseline',
        alpha=0.7,
        linewidth=1.5,
        markersize=6
    )
    
    
    ax.plot(
        snr_values,
        knn_acc,
        'r-',
        marker='s',
        label='KNN',
        linewidth=2,
        markersize=6
    )
    
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_title(title)
    
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    
    ax.legend(loc='best')
    
    
    plt.tight_layout()
    
    return fig



def plot_feature_distributions(
    dataframe: pd.DataFrame,
    feature_name: str,
    title: str = "Feature Distribution"
) -> plt.Figure:
    """
    Generate feature distribution histogram with KDE overlay.
    
    Creates a histogram showing the distribution of a feature across different
    modulation classes. Uses seaborn histplot with KDE and viridis palette.
    
    Args:
        dataframe: DataFrame with columns 'Feature' and 'Modulation'
        feature_name: Name of the feature being plotted
        title: Plot title
    
    Returns:
        Matplotlib figure compatible with Gradio gr.Plot()
    
    Example:
        >>> df = pd.DataFrame({
        ...     'Feature': np.concatenate([np.random.normal(0, 1, 500), 
        ...                                np.random.normal(2, 1.5, 500)]),
        ...     'Modulation': ['AM'] * 500 + ['FM'] * 500
        ... })
        >>> fig = plot_feature_distributions(df, 'amp_kurtosis', 'Amplitude Kurtosis')
    """
    
    if len(dataframe) > 1000:
        sampled_df = dataframe.sample(n=1000, random_state=42)
    else:
        sampled_df = dataframe
    
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    
    sns.histplot(
        data=sampled_df,
        x='Feature',
        hue='Modulation',
        kde=True,
        palette='viridis',
        ax=ax,
        alpha=0.6,
        stat='count'
    )
    
   
    ax.set_xlabel(feature_name.replace('_', ' ').title())
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    
    plt.tight_layout()
    
    return fig



def generate_complexity_table(
    knn_metrics: Dict[str, float]
) -> pd.DataFrame:
    """
    Generate computation complexity table for KNN model.
    
    Creates a pandas DataFrame showing training time and inference time
    for the KNN model.
    
    Args:
        knn_metrics: Dictionary with 'training_time' and 'inference_time' keys
    
    Returns:
        DataFrame with columns: Method, Training Time (seconds), 
        Average Inference Time (ms/sample)
    
    Example:
        >>> knn_metrics = {'training_time': 2.345, 'inference_time': 1.234}
        >>> df = generate_complexity_table(knn_metrics)
    """
    
    knn_training = knn_metrics.get('training_time', 0.0)
    knn_inference = knn_metrics.get('inference_time', 0.0)
    
    
    data = [
        ["K-Nearest Neighbors", round(knn_training, 3), round(knn_inference, 3)]
    ]
    
    
    df = pd.DataFrame(
        data,
        columns=["Method", "Training Time (seconds)", "Average Inference Time (ms/sample)"]
    )
    
    return df



def get_plot_cache() -> PlotCache:
    """
    Get the global plot cache instance.
    
    Returns:
        PlotCache instance
    """
    return _plot_cache


def invalidate_plot_cache(round_num: Optional[int] = None) -> None:
    """
    Invalidate plot cache for a specific round or all rounds.
    
    Args:
        round_num: Round number to invalidate (invalidates all if None)
    """
    _plot_cache.invalidate(round_num)


def set_cache_round(round_num: int) -> None:
    """
    Set current round number for plot cache.
    
    Args:
        round_num: Current round number
    """
    _plot_cache.set_current_round(round_num)



def create_confusion_matrix_from_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: Optional[float] = None
) -> plt.Figure:
    """
    Create confusion matrix with automatic title formatting for KNN.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        snr: SNR value for title (optional)
    
    Returns:
        Matplotlib figure
    """
    classes = ['AM', 'FM']
    
    
    if snr is not None:
        title = f"KNN Confusion Matrix (SNR = {snr} dB)"
    else:
        title = "KNN Confusion Matrix"
    
    return plot_confusion_matrix(y_true, y_pred, classes, title)


def create_feature_distribution_plot(
    features: np.ndarray,
    labels: np.ndarray,
    feature_name: str,
    class_names: List[str] = None
) -> plt.Figure:
    """
    Create feature distribution plot from raw data.
    
    Args:
        features: Feature values (1D array)
        labels: Class labels (1D array)
        feature_name: Name of the feature
        class_names: List of class names (defaults to ['AM', 'FM'])
    
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ['AM', 'FM']
    
    
    df = pd.DataFrame({
        'Feature': features,
        'Modulation': [class_names[int(label)] for label in labels]
    })
    
    
    title = f"{feature_name.replace('_', ' ').title()} Distribution"
    
    return plot_feature_distributions(df, feature_name, title)


def create_accuracy_comparison_plot(
    snr_values: List[float],
    knn_per_snr: Dict[float, float],
    baseline: float = 50.0
) -> plt.Figure:
    """
    Create accuracy vs SNR plot from per-SNR accuracy dictionary.
    
    Args:
        snr_values: List of SNR values
        knn_per_snr: Dictionary mapping SNR to KNN accuracy
        baseline: Baseline accuracy (default 50.0 for 2-class)
    
    Returns:
        Matplotlib figure
    """
    knn_acc = [knn_per_snr.get(snr, 0.0) for snr in snr_values]
    baseline_acc = [baseline] * len(snr_values)
    
    return plot_accuracy_vs_snr(
        snr_values,
        knn_acc,
        baseline_acc,
        "KNN Model Accuracy vs SNR"
    )

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
import os

def _setup_plot_style():
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    else:
        if 'seaborn-whitegrid' in plt.style.available:
            plt.style.use('seaborn-whitegrid')
        else:
            plt.grid(True, linestyle='--', alpha=0.7)

def generate_accuracy_vs_rounds_plot(history_data: dict, output_path: str):
    """Generate accuracy vs rounds plot."""
    rounds = []
    knn_after = []
    
    for round_data in history_data.get('rounds', []):
        rounds.append(round_data.get('round', 0))
        after_acc = round_data.get('after', {}).get('knn_accuracy', 0.0)
        knn_after.append(after_acc)
        
    if not rounds:
        return

    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, knn_after, marker='o', linewidth=2, markersize=8, color='#1f77b4')
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy')
    ax.set_title('Global Model Accuracy vs Rounds')
    ax.set_ylim([0, 1.05])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def generate_client_contribution_plot(latest_metrics: dict, output_path: str):
    """Generate client contribution plot (number of samples per client)."""
    if not latest_metrics:
        return
        
    client_ids = []
    samples = []
    
    for cid, metrics in latest_metrics.items():
        client_ids.append(cid)
        samples.append(metrics.get('n_samples', 0))
        
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if HAS_SEABORN:
        sns.barplot(x=client_ids, y=samples, ax=ax, palette="Blues_d")
    else:
        ax.bar(client_ids, samples, color='#1f77b4')
        
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Client Contribution (Training Samples)')
    plt.xticks(rotation=45, ha='right')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def generate_per_client_accuracy_plot(latest_metrics: dict, output_path: str):
    """Generate per-client accuracy plot."""
    if not latest_metrics:
        return
        
    client_ids = []
    accuracies = []
    
    for cid, metrics in latest_metrics.items():
        client_ids.append(cid)
        accuracies.append(metrics.get('test_accuracy', 0.0))
        
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if HAS_SEABORN:
        sns.barplot(x=client_ids, y=accuracies, ax=ax, palette="Greens_d")
    else:
        ax.bar(client_ids, accuracies, color='#2ca02c')
        
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Per-Client Model Accuracy')
    ax.set_ylim([0, 1.05])
    plt.xticks(rotation=45, ha='right')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
