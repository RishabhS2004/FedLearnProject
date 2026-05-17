"""
Publication-Quality Evaluation Plots for RadioFed

Generates thesis-ready, high-DPI plots for:
- Confusion matrices (raw + normalized)
- SNR-wise performance (accuracy, F1 macro, F1 weighted)
- Byzantine strategy comparison across federated rounds

Design decisions:
- Does NOT duplicate dashboard-oriented plotting in central/visualization.py
  (that module serves the Gradio/FastAPI dashboard with in-memory figures).
- This module focuses exclusively on file-based, publication-quality output
  with consistent academic styling (300+ DPI, tight layout, readable fonts).
- All functions accept pre-computed data and save to disk; no model loading.

Output directories:
- out/plots/confusion_matrix/  — confusion matrix heatmaps
- out/plots/                   — SNR evaluation plots
- out/plots/byzantine/         — Byzantine strategy comparison plots
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger("federated_central")

# ── Publication Style Configuration ──────────────────────────────────────────

# Consistent color palette for strategies
STRATEGY_COLORS = {
    "FedAvg": "#1f77b4",
    "fedavg": "#1f77b4",
    "Krum": "#e74c3c",
    "krum": "#e74c3c",
    "Trimmed Mean": "#2ecc71",
    "trimmed_mean": "#2ecc71",
    "Trust Weighted": "#9b59b6",
    "trust_weighted": "#9b59b6",
}

STRATEGY_MARKERS = {
    "FedAvg": "o",
    "fedavg": "o",
    "Krum": "s",
    "krum": "s",
    "Trimmed Mean": "^",
    "trimmed_mean": "^",
    "Trust Weighted": "D",
    "trust_weighted": "D",
}

# Publication plot defaults
_PLOT_DEFAULTS = {
    "dpi": 300,
    "figsize_cm": (10, 7),      # confusion matrix
    "figsize_line": (10, 5.5),  # line plots
    "font_family": "serif",
    "font_size": 11,
    "title_size": 13,
    "label_size": 11,
    "tick_size": 9,
    "legend_size": 9,
    "linewidth": 2.0,
    "marker_size": 7,
    "grid_alpha": 0.3,
}


def _apply_academic_style():
    """Apply consistent academic style to all plots."""
    plt.rcParams.update({
        "font.family": _PLOT_DEFAULTS["font_family"],
        "font.size": _PLOT_DEFAULTS["font_size"],
        "axes.titlesize": _PLOT_DEFAULTS["title_size"],
        "axes.labelsize": _PLOT_DEFAULTS["label_size"],
        "xtick.labelsize": _PLOT_DEFAULTS["tick_size"],
        "ytick.labelsize": _PLOT_DEFAULTS["tick_size"],
        "legend.fontsize": _PLOT_DEFAULTS["legend_size"],
        "figure.dpi": _PLOT_DEFAULTS["dpi"],
        "savefig.dpi": _PLOT_DEFAULTS["dpi"],
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": _PLOT_DEFAULTS["grid_alpha"],
        "grid.linestyle": "--",
    })


def _save_figure(fig, output_path: str) -> str:
    """Save figure with consistent settings and close."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(
        output_path,
        dpi=_PLOT_DEFAULTS["dpi"],
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    logger.info(f"Plot saved to {output_path}")
    return os.path.abspath(output_path)


def _get_strategy_display_name(strategy: str) -> str:
    """Convert internal strategy key to display name."""
    names = {
        "fedavg": "FedAvg",
        "krum": "Krum",
        "trimmed_mean": "Trimmed Mean",
        "trust_weighted": "Trust Weighted",
        "full": "Full Pipeline",
    }
    return names.get(strategy.lower(), strategy)


# ── Confusion Matrix Plots ───────────────────────────────────────────────────

def plot_confusion_matrix_raw(
    conf_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    output_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Generate a raw (count-based) confusion matrix heatmap.

    Args:
        conf_matrix: 2D numpy array (n_classes x n_classes).
        class_names: List of class label strings; auto-generated if None.
        title: Plot title.
        output_path: Full path for the output PNG. Auto-generated if None.
        timestamp: Timestamp string for filename; auto-generated if None.

    Returns:
        Absolute path of the saved PNG file.
    """
    _apply_academic_style()

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = f"out/plots/confusion_matrix/cm_raw_{timestamp}.png"

    n_classes = conf_matrix.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Scale figure size with number of classes
    size = max(6, n_classes * 0.7 + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="#e0e0e0",
        annot_kws={"size": max(7, 12 - n_classes // 3)},
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, pad=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    return _save_figure(fig, output_path)


def plot_confusion_matrix_normalized(
    conf_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Normalized Confusion Matrix",
    output_path: Optional[str] = None,
    timestamp: Optional[str] = None,
    accuracy: Optional[float] = None,
    f1_macro: Optional[float] = None,
    f1_weighted: Optional[float] = None,
) -> str:
    """
    Generate a row-normalized (recall-based) confusion matrix heatmap.

    Each row is normalized to sum to 1.0, showing the proportion of
    true labels predicted as each class.

    Args:
        conf_matrix: 2D numpy array (n_classes x n_classes), raw counts.
        class_names: List of class label strings; auto-generated if None.
        title: Plot title.
        output_path: Full path for the output PNG. Auto-generated if None.
        timestamp: Timestamp string for filename; auto-generated if None.

    Returns:
        Absolute path of the saved PNG file.
    """
    _apply_academic_style()

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = f"out/plots/confusion_matrix/cm_normalized_{timestamp}.png"

    n_classes = conf_matrix.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Row-normalize
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid division by zero
    cm_normalized = (conf_matrix.astype(float) / row_sums) * 100.0

    size = max(6, n_classes * 0.7 + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    # Create percentage labels for each cell
    labels = np.array([f"{val:.2f}%" for val in cm_normalized.flatten()]).reshape(
        cm_normalized.shape
    )

    sns.heatmap(
        cm_normalized,
        annot=labels,
        fmt="",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
        square=True,
        vmin=0.0,
        vmax=100.0,
        linewidths=0.5,
        linecolor="#e0e0e0",
        annot_kws={"size": max(7, 12 - n_classes // 3)},
    )

    if accuracy is not None and f1_macro is not None and f1_weighted is not None:
        title = (
            f"{title}\n"
            f"(Acc: {accuracy*100:.2f}%, F1-M: {f1_macro:.4f}, F1-W: {f1_weighted:.4f})"
        )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, pad=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    return _save_figure(fig, output_path)


# ── SNR Evaluation Plots ─────────────────────────────────────────────────────

def plot_accuracy_vs_snr(
    per_snr_accuracy: Dict[float, float],
    title: str = "Classification Accuracy vs. SNR",
    output_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Generate Accuracy vs SNR line plot.

    Args:
        per_snr_accuracy: Dict mapping SNR (dB) → accuracy (0.0–1.0).
        title: Plot title.
        output_path: Output PNG path; auto-generated if None.
        timestamp: Timestamp string; auto-generated if None.

    Returns:
        Absolute path of the saved PNG file.
    """
    _apply_academic_style()

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = f"out/plots/accuracy_vs_snr_{timestamp}.png"

    snrs = sorted(per_snr_accuracy.keys())
    accs = [per_snr_accuracy[s] * 100 for s in snrs]  # convert to percentage

    fig, ax = plt.subplots(figsize=_PLOT_DEFAULTS["figsize_line"])

    ax.plot(
        snrs, accs,
        marker="o",
        linewidth=_PLOT_DEFAULTS["linewidth"],
        markersize=_PLOT_DEFAULTS["marker_size"],
        color="#1f77b4",
        label="Accuracy",
    )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, pad=10)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    return _save_figure(fig, output_path)


def plot_f1_vs_snr(
    per_snr_f1_macro: Dict[float, float],
    per_snr_f1_weighted: Optional[Dict[float, float]] = None,
    title: str = "F1-Score vs. SNR",
    output_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Generate F1-Score (macro and optionally weighted) vs SNR line plot.

    Args:
        per_snr_f1_macro: Dict mapping SNR (dB) → F1 macro score (0.0–1.0).
        per_snr_f1_weighted: Optional dict mapping SNR → F1 weighted score.
        title: Plot title.
        output_path: Output PNG path; auto-generated if None.
        timestamp: Timestamp string; auto-generated if None.

    Returns:
        Absolute path of the saved PNG file.
    """
    _apply_academic_style()

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = f"out/plots/f1_vs_snr_{timestamp}.png"

    snrs = sorted(per_snr_f1_macro.keys())
    f1_macro = [per_snr_f1_macro[s] for s in snrs]

    fig, ax = plt.subplots(figsize=_PLOT_DEFAULTS["figsize_line"])

    ax.plot(
        snrs, f1_macro,
        marker="o",
        linewidth=_PLOT_DEFAULTS["linewidth"],
        markersize=_PLOT_DEFAULTS["marker_size"],
        color="#e74c3c",
        label="F1 Macro",
    )

    if per_snr_f1_weighted is not None:
        f1_weighted = [per_snr_f1_weighted.get(s, 0) for s in snrs]
        ax.plot(
            snrs, f1_weighted,
            marker="s",
            linewidth=_PLOT_DEFAULTS["linewidth"],
            markersize=_PLOT_DEFAULTS["marker_size"],
            color="#2ecc71",
            linestyle="--",
            label="F1 Weighted",
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("F1-Score")
    ax.set_title(title, pad=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    return _save_figure(fig, output_path)


# ── Byzantine Strategy Comparison Plots ──────────────────────────────────────

def plot_byzantine_accuracy_comparison(
    strategy_results: Dict[str, List[Dict[str, Any]]],
    metric_key: str = "knn_accuracy",
    title: str = "Accuracy vs. Federated Round",
    ylabel: str = "Accuracy (%)",
    scale_percent: bool = True,
    output_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Generate accuracy (or F1) vs federated round for multiple Byzantine strategies.

    Args:
        strategy_results: Dict mapping strategy name → list of per-round dicts.
            Each round dict must have 'round' and the metric_key.
        metric_key: Key to extract from each round dict (e.g., 'knn_accuracy', 'knn_f1').
        title: Plot title.
        ylabel: Y-axis label.
        scale_percent: If True, multiply metric values by 100.
        output_path: Output PNG path; auto-generated if None.
        timestamp: Timestamp string; auto-generated if None.

    Returns:
        Absolute path of the saved PNG file.
    """
    _apply_academic_style()

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        safe_key = metric_key.replace("_", "-")
        output_path = f"out/plots/byzantine/byzantine_{safe_key}_{timestamp}.png"

    fig, ax = plt.subplots(figsize=_PLOT_DEFAULTS["figsize_line"])

    for strategy, rounds_data in strategy_results.items():
        display_name = _get_strategy_display_name(strategy)
        color = STRATEGY_COLORS.get(strategy, STRATEGY_COLORS.get(strategy.lower(), "#333333"))
        marker = STRATEGY_MARKERS.get(strategy, STRATEGY_MARKERS.get(strategy.lower(), "o"))

        round_nums = [r["round"] for r in rounds_data]
        values = [r.get(metric_key, 0) for r in rounds_data]

        if scale_percent:
            values = [v * 100 for v in values]

        ax.plot(
            round_nums, values,
            marker=marker,
            linewidth=_PLOT_DEFAULTS["linewidth"],
            markersize=_PLOT_DEFAULTS["marker_size"],
            color=color,
            label=display_name,
        )

    ax.set_xlabel("Federated Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if scale_percent:
        ax.set_ylim(0, 105)
    else:
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return _save_figure(fig, output_path)


def plot_statistical_comparison(
    multi_run_results: Dict[str, List[List[Dict[str, Any]]]],
    metric_key: str = "knn_accuracy",
    title: str = "Statistical Comparison (Accuracy)",
    ylabel: str = "Accuracy (%)",
    scale_percent: bool = True,
    output_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Plot mean curves with shaded confidence bands (±1 std) for multiple strategies.

    Args:
        multi_run_results: Dict mapping strategy name → list of runs.
            Each run is a list of per-round dicts.
        metric_key: Key to extract from round dicts.
        title: Plot title.
        ylabel: Y-axis label.
        scale_percent: If True, multiply by 100.
        output_path: Output path.
        timestamp: Timestamp.
    """
    _apply_academic_style()

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        safe_key = metric_key.replace("_", "-")
        output_path = f"out/plots/byzantine/stats_{safe_key}_{timestamp}.png"

    fig, ax = plt.subplots(figsize=_PLOT_DEFAULTS["figsize_line"])

    for strategy, all_runs in multi_run_results.items():
        if not all_runs:
            continue

        color = STRATEGY_COLORS.get(
            strategy, STRATEGY_COLORS.get(strategy.lower(), "#333333")
        )
        display_name = _get_strategy_display_name(strategy)

        # Collect data per round across all runs
        n_rounds = len(all_runs[0])
        n_runs = len(all_runs)
        
        round_nums = [r["round"] for r in all_runs[0]]
        
        # Matrix: (n_runs, n_rounds)
        run_matrix = np.zeros((n_runs, n_rounds))
        for run_idx, rounds_data in enumerate(all_runs):
            for round_idx, r_data in enumerate(rounds_data):
                val = r_data.get(metric_key, 0)
                if scale_percent:
                    val *= 100
                run_matrix[run_idx, round_idx] = val

        means = np.mean(run_matrix, axis=0)
        stds = np.std(run_matrix, axis=0)

        # Plot mean line
        ax.plot(
            round_nums,
            means,
            label=f"{display_name} (n={n_runs})",
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
        )

        # Plot shaded confidence band
        ax.fill_between(
            round_nums,
            means - stds,
            means + stds,
            color=color,
            alpha=0.15,
            edgecolor="none",
        )

    ax.set_xlabel("Federated Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if scale_percent:
        ax.set_ylim(0, 105)
    else:
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return _save_figure(fig, output_path)


def plot_byzantine_f1_comparison(
    strategy_results: Dict[str, List[Dict[str, Any]]],
    metric_key: str = "knn_f1",
    title: str = "F1 Macro vs. Federated Round",
    output_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Generate F1-Score vs federated round for multiple Byzantine strategies.

    Convenience wrapper around plot_byzantine_accuracy_comparison
    with F1-appropriate defaults.
    """
    if output_path is None and timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = f"out/plots/byzantine/byzantine_f1_macro_{timestamp}.png"

    return plot_byzantine_accuracy_comparison(
        strategy_results,
        metric_key=metric_key,
        title=title,
        ylabel="F1 Macro Score",
        scale_percent=False,
        output_path=output_path,
        timestamp=timestamp,
    )


def plot_byzantine_client_acceptance(
    strategy_results: Dict[str, List[Dict[str, Any]]],
    title: str = "Client Acceptance per Round",
    output_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Generate bar chart showing accepted vs rejected clients per round,
    grouped by Byzantine strategy.

    Args:
        strategy_results: Dict mapping strategy name → list of per-round dicts.
            Each round dict must have 'round', 'n_accepted', 'n_rejected'.
        title: Plot title.
        output_path: Output PNG path; auto-generated if None.
        timestamp: Timestamp string; auto-generated if None.

    Returns:
        Absolute path of the saved PNG file.
    """
    _apply_academic_style()

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = f"out/plots/byzantine/byzantine_client_acceptance_{timestamp}.png"

    n_strategies = len(strategy_results)
    if n_strategies == 0:
        logger.warning("No strategy results to plot for client acceptance.")
        return ""

    # Find max rounds across strategies
    all_rounds = set()
    for rounds_data in strategy_results.values():
        for r in rounds_data:
            all_rounds.add(r["round"])
    rounds_sorted = sorted(all_rounds)
    n_rounds = len(rounds_sorted)

    fig, ax = plt.subplots(figsize=(max(8, n_rounds * 1.5), 5.5))

    bar_width = 0.8 / n_strategies
    x = np.arange(n_rounds)

    for idx, (strategy, rounds_data) in enumerate(strategy_results.items()):
        display_name = _get_strategy_display_name(strategy)
        color = STRATEGY_COLORS.get(strategy, STRATEGY_COLORS.get(strategy.lower(), "#333333"))

        # Build accepted/rejected arrays aligned to rounds_sorted
        round_lookup = {r["round"]: r for r in rounds_data}
        accepted = [round_lookup.get(rnd, {}).get("n_accepted", 0) for rnd in rounds_sorted]
        rejected = [round_lookup.get(rnd, {}).get("n_rejected", 0) for rnd in rounds_sorted]

        offset = (idx - n_strategies / 2 + 0.5) * bar_width
        bars_acc = ax.bar(
            x + offset, accepted, bar_width * 0.9,
            label=f"{display_name} (accepted)",
            color=color, alpha=0.8,
        )
        ax.bar(
            x + offset, rejected, bar_width * 0.9,
            bottom=accepted,
            label=f"{display_name} (rejected)",
            color=color, alpha=0.3, hatch="//",
        )

    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Number of Clients")
    ax.set_title(title, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rounds_sorted])
    ax.legend(loc="best", fontsize=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    return _save_figure(fig, output_path)


# ── Convenience: generate all evaluation plots from a single result ──────────

def generate_all_evaluation_plots(
    conf_matrix: np.ndarray,
    per_snr_accuracy: Dict[float, float],
    per_snr_f1_macro: Dict[float, float],
    per_snr_f1_weighted: Optional[Dict[float, float]] = None,
    class_names: Optional[List[str]] = None,
    timestamp: Optional[str] = None,
    accuracy: Optional[float] = None,
    f1_macro: Optional[float] = None,
    f1_weighted: Optional[float] = None,
) -> Dict[str, str]:
    """
    Generate all standard evaluation plots in one call.

    Args:
        conf_matrix: Raw confusion matrix (numpy 2D array).
        per_snr_accuracy: Dict SNR → accuracy.
        per_snr_f1_macro: Dict SNR → F1 macro.
        per_snr_f1_weighted: Optional dict SNR → F1 weighted.
        class_names: Class label names.
        timestamp: Shared timestamp for all filenames.

    Returns:
        Dict mapping plot name to absolute file path.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = {}

    paths["cm_raw"] = plot_confusion_matrix_raw(
        conf_matrix, class_names, timestamp=timestamp,
    )
    # Normalized CM (main CM for reports)
    paths["cm_normalized"] = plot_confusion_matrix_normalized(
        conf_matrix,
        class_names,
        timestamp=timestamp,
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
    )

    if per_snr_accuracy:
        paths["accuracy_vs_snr"] = plot_accuracy_vs_snr(
            per_snr_accuracy, timestamp=timestamp,
        )

    if per_snr_f1_macro:
        paths["f1_vs_snr"] = plot_f1_vs_snr(
            per_snr_f1_macro, per_snr_f1_weighted, timestamp=timestamp,
        )

    logger.info(f"Generated {len(paths)} evaluation plots")
    return paths
