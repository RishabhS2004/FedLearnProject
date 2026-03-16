"""
Central Dashboard for RadioFed - Production-Level Monitoring

Comprehensive Gradio dashboard with Byzantine fault tolerance monitoring,
model comparison, trust scores, and advanced visualizations.
"""

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.metrics import confusion_matrix

from central.state import (
    get_client_status,
    get_registry_stats,
    get_latest_aggregation_result,
    get_all_aggregation_results,
    get_accuracy_trends,
    get_latest_round_metrics,
    get_auto_aggregation_state
)
from central.byzantine import (
    get_all_trust_scores,
    get_trust_history,
    get_byzantine_aggregator
)

logger = logging.getLogger(__name__)


# ─── Custom CSS ───────────────────────────────────────────────────────────────

DASHBOARD_CSS = """
:root {
    --primary: #6366f1;
    --accent: #06b6d4;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

.gradio-container {
    max-width: 1600px !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

.dashboard-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 16px;
    border: 1px solid rgba(99, 102, 241, 0.3);
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.15);
}

.dashboard-header h1 {
    background: linear-gradient(135deg, #818cf8, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2em;
    margin: 0;
}

.dashboard-header p {
    color: #94a3b8;
    margin: 4px 0 0 0;
}

.stat-card {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.9), rgba(49, 46, 129, 0.5));
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    min-height: 100px;
}

.stat-card:hover {
    border-color: rgba(99, 102, 241, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
}

.stat-value {
    font-size: 2.2em;
    font-weight: 700;
    line-height: 1;
}

.stat-label {
    font-size: 0.85em;
    color: #94a3b8;
    margin-top: 8px;
}

.byzantine-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(245, 158, 11, 0.05));
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 12px;
    padding: 16px;
}

.trust-high { color: #10b981; }
.trust-medium { color: #f59e0b; }
.trust-low { color: #ef4444; }
"""

# ─── Plot Style Config ────────────────────────────────────────────────────────

PLOT_BG = '#0f0f1a'
PLOT_FACE = '#1e1b4b'
PLOT_TEXT = '#e2e8f0'
PLOT_GRID = '#6366f1'
PLOT_MUTED = '#94a3b8'


def _style_ax(ax):
    """Apply consistent dark style to axes."""
    ax.set_facecolor(PLOT_FACE)
    ax.tick_params(colors=PLOT_MUTED)
    ax.grid(True, alpha=0.15, color=PLOT_GRID)
    for spine in ax.spines.values():
        spine.set_color(PLOT_GRID)
        spine.set_alpha(0.3)


# ─── Dashboard State ──────────────────────────────────────────────────────────

class DashboardState:
    def __init__(self):
        self.current_round = 0
        self.server_running = True
        self.snr_levels = list(range(-20, 20, 2))
        self.modulation_classes = ['AM', 'FM']
        self.num_classes = 2

    def get_baseline_accuracy(self) -> float:
        return 100.0 / self.num_classes


dashboard_state = DashboardState()


# ─── Dashboard Update Functions ───────────────────────────────────────────────

def get_system_status_html():
    """Generate system status HTML cards."""
    stats = get_registry_stats()
    agg_state = get_auto_aggregation_state()
    knn_result = get_latest_aggregation_result('knn')

    n_clients = stats['total_clients']
    n_samples = stats['total_samples']
    current_round = agg_state.get('current_round', 0)
    pending = agg_state.get('pending_uploads', 0)
    threshold = agg_state.get('threshold', 2)

    # Get accuracy
    accuracy = "N/A"
    if knn_result and 'result' in knn_result:
        acc = knn_result['result'].get('accuracy', 0)
        accuracy = f"{acc*100:.1f}%" if acc else "N/A"

    # Trust scores summary
    trust_scores = get_all_trust_scores()
    avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 0
    low_trust = sum(1 for s in trust_scores.values() if s < 0.3)

    return f"""
    <div style="display:grid; grid-template-columns: repeat(6, 1fr); gap:12px; margin-bottom:16px;">
        <div class="stat-card">
            <div class="stat-value" style="color:#10b981">&#x25CF;</div>
            <div class="stat-label">Server Running</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#6366f1">{n_clients}</div>
            <div class="stat-label">Connected Clients</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#8b5cf6">{current_round}</div>
            <div class="stat-label">Aggregation Round</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#06b6d4">{pending}/{threshold}</div>
            <div class="stat-label">Pending Uploads</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#10b981">{accuracy}</div>
            <div class="stat-label">Global Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:{'#ef4444' if low_trust > 0 else '#10b981'}">{low_trust}</div>
            <div class="stat-label">Low Trust Clients</div>
        </div>
    </div>
    """


def get_client_table():
    """Get client monitoring table."""
    clients = get_client_status()
    trust_scores = get_all_trust_scores()

    if not clients:
        return pd.DataFrame(columns=["Client ID", "Status", "Trust Score",
                                      "Last Upload", "Samples"])
    data = []
    for client in clients:
        cid = client['client_id']
        status = client.get('status', 'unknown')

        if status == 'weights_uploaded':
            status_display = "Uploaded"
        elif status == 'training':
            status_display = "Training..."
        elif status == 'idle':
            status_display = "Idle"
        else:
            status_display = "Connected"

        trust = trust_scores.get(cid, 0.5)
        trust_display = f"{trust:.2f}"

        last_upload = client.get('last_upload', 'Never')
        if last_upload and last_upload != 'Never':
            try:
                dt = datetime.fromisoformat(last_upload)
                last_upload = dt.strftime("%H:%M:%S")
            except Exception:
                pass

        data.append([cid, status_display, trust_display,
                    last_upload, client.get('n_samples', 0)])

    return pd.DataFrame(data, columns=["Client ID", "Status", "Trust Score",
                                        "Last Upload", "Samples"])


def get_trust_scores_plot():
    """Generate trust score bar chart."""
    trust_scores = get_all_trust_scores()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(PLOT_BG)
    _style_ax(ax)

    if not trust_scores:
        ax.text(0.5, 0.5, 'No client data', ha='center', va='center',
               color=PLOT_TEXT, fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        return fig

    clients = list(trust_scores.keys())
    scores = list(trust_scores.values())
    colors = ['#10b981' if s >= 0.5 else '#f59e0b' if s >= 0.3 else '#ef4444'
              for s in scores]

    bars = ax.barh(clients, scores, color=colors, alpha=0.8, height=0.6)

    # Threshold line
    ax.axvline(x=0.3, color='#ef4444', linestyle='--', alpha=0.6, label='Rejection Threshold')
    ax.axvline(x=0.5, color='#f59e0b', linestyle='--', alpha=0.4, label='Warning Threshold')

    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Trust Score', color=PLOT_TEXT)
    ax.set_title('Client Trust Scores', color=PLOT_TEXT, fontweight='bold', fontsize=14)
    ax.legend(facecolor=PLOT_FACE, edgecolor=PLOT_GRID, labelcolor=PLOT_TEXT, fontsize=8)
    ax.tick_params(colors=PLOT_MUTED)

    # Value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f'{score:.2f}', va='center', color=PLOT_TEXT, fontsize=10)

    plt.tight_layout()
    return fig


def get_byzantine_report():
    """Get Byzantine defense report."""
    aggregator = get_byzantine_aggregator()
    log = aggregator.get_aggregation_log()

    if not log:
        return "No Byzantine filtering has occurred yet. Waiting for aggregation rounds."

    latest = log[-1]
    report = f"""### Byzantine Defense Report - {latest.get('timestamp', 'N/A')[:19]}

| Metric | Value |
|--------|-------|
| Strategy | `{latest.get('strategy', 'N/A')}` |
| Total Clients | {latest.get('total_clients', 0)} |
| Accepted | {latest.get('accepted_count', 0)} |
| Rejected | {latest.get('rejected_count', 0)} |

"""
    rejected = latest.get('rejected_clients', [])
    if rejected:
        report += "#### Rejected Clients\n\n"
        report += "| Client | Reason |\n|--------|--------|\n"
        for r in rejected:
            report += f"| {r.get('client_id', 'N/A')} | {r.get('reason', 'N/A')} |\n"

    actions = latest.get('defense_actions', [])
    if actions:
        report += f"\n#### Defense Actions ({len(actions)} total)\n\n"
        for a in actions[-5:]:
            report += f"- **{a.get('step', 'N/A')}**: {a.get('action', 'N/A')}"
            if 'client' in a:
                report += f" ({a['client']})"
            report += "\n"

    return report


def get_accuracy_trends_plot():
    """Generate accuracy trends over rounds."""
    trends = get_accuracy_trends()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PLOT_BG)
    _style_ax(ax)

    if not trends['rounds']:
        ax.text(0.5, 0.5, 'No training history available',
               ha='center', va='center', fontsize=14, color=PLOT_TEXT,
               transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        return fig

    rounds = trends['rounds']

    ax.plot(rounds, trends['knn_before'], 'o--', color='#6366f1',
           label='KNN (Before Agg)', alpha=0.6, linewidth=1.5, markersize=6)
    ax.plot(rounds, trends['knn_after'], 'o-', color='#6366f1',
           label='KNN (After Agg)', linewidth=2.5, markersize=7)

    # Fill between
    ax.fill_between(rounds, trends['knn_before'], trends['knn_after'],
                    alpha=0.1, color='#6366f1')

    ax.set_xlabel('Training Round', color=PLOT_TEXT)
    ax.set_ylabel('Accuracy', color=PLOT_TEXT)
    ax.set_title('Accuracy Trends Over Training Rounds', color=PLOT_TEXT,
                fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor=PLOT_FACE, edgecolor=PLOT_GRID, labelcolor=PLOT_TEXT)

    plt.tight_layout()
    return fig


def get_before_after_table():
    """Generate before/after comparison."""
    latest = get_latest_round_metrics()

    if not latest:
        return pd.DataFrame(
            [['No aggregation data', '-', '-', '-']],
            columns=['Metric', 'Before', 'After', 'Improvement']
        )

    knn_before = latest['before']['knn_accuracy']
    knn_after = latest['after']['knn_accuracy']
    knn_improvement = latest['improvement']['knn']

    data = [
        ['KNN Accuracy',
         f"{knn_before:.2%}", f"{knn_after:.2%}",
         f"+{knn_improvement:.2%}" if knn_improvement >= 0 else f"{knn_improvement:.2%}"]
    ]

    return pd.DataFrame(data, columns=['Metric', 'Before', 'After', 'Improvement'])


def get_snr_accuracy_table():
    """Generate per-SNR accuracy table."""
    baseline = dashboard_state.get_baseline_accuracy()
    knn_result = get_latest_aggregation_result('knn')

    knn_per_snr = {}
    if knn_result and 'result' in knn_result:
        knn_per_snr = knn_result['result'].get('per_snr_accuracy', {})

    data = []
    for snr in dashboard_state.snr_levels:
        knn_acc = knn_per_snr.get(snr, knn_per_snr.get(float(snr),
                  knn_per_snr.get(str(snr), 0.0)))
        if isinstance(knn_acc, (int, float)) and 0 < knn_acc <= 1.0:
            knn_acc = knn_acc * 100
        data.append([snr, round(baseline, 2), round(knn_acc, 2)])

    return pd.DataFrame(data, columns=["SNR (dB)", "Baseline (%)", "KNN (%)"])


def get_accuracy_vs_snr_plot():
    """Generate accuracy vs SNR plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PLOT_BG)
    _style_ax(ax)

    snr_values = dashboard_state.snr_levels
    baseline_acc = dashboard_state.get_baseline_accuracy()
    baseline = [baseline_acc] * len(snr_values)

    knn_result = get_latest_aggregation_result('knn')
    knn_accuracy = []

    for snr in snr_values:
        if knn_result and 'result' in knn_result:
            per_snr = knn_result['result'].get('per_snr_accuracy', {})
            acc = per_snr.get(snr, per_snr.get(float(snr), per_snr.get(str(snr), 0.0)))
            if isinstance(acc, (int, float)) and 0 < acc <= 1.0:
                acc = acc * 100
            knn_accuracy.append(acc)
        else:
            knn_accuracy.append(0.0)

    ax.plot(snr_values, baseline, 'o--', color='#94a3b8', label='Baseline (Random)',
           alpha=0.7, linewidth=1.5, markersize=5)
    ax.plot(snr_values, knn_accuracy, 's-', color='#6366f1', label='KNN',
           linewidth=2.5, markersize=6)

    ax.fill_between(snr_values, baseline, knn_accuracy, alpha=0.1, color='#6366f1')

    ax.set_xlabel('SNR (dB)', color=PLOT_TEXT)
    ax.set_ylabel('Accuracy (%)', color=PLOT_TEXT)
    ax.set_ylim(0, 105)
    ax.set_title('Model Accuracy vs Signal-to-Noise Ratio', color=PLOT_TEXT,
                fontweight='bold', fontsize=14)
    ax.legend(facecolor=PLOT_FACE, edgecolor=PLOT_GRID, labelcolor=PLOT_TEXT)

    plt.tight_layout()
    return fig


def get_confusion_matrix_plot():
    """Generate confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_FACE)

    result = get_latest_aggregation_result('knn')
    if result and 'result' in result:
        cm_data = result['result'].get('confusion_matrix')
        if cm_data is not None:
            cm = np.array(cm_data) if isinstance(cm_data, list) else cm_data
        else:
            cm = np.array([[0, 0], [0, 0]])
    else:
        cm = np.array([[0, 0], [0, 0]])

    classes = dashboard_state.modulation_classes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=classes, yticklabels=classes, ax=ax,
               cbar=True, linewidths=0.5, linecolor=PLOT_GRID)

    ax.set_xlabel('Predicted', color=PLOT_TEXT, fontsize=12)
    ax.set_ylabel('Actual', color=PLOT_TEXT, fontsize=12)

    agg_state = get_auto_aggregation_state()
    round_num = agg_state.get('current_round', 0)
    title = f'KNN Confusion Matrix - Round {round_num}' if round_num > 0 else 'KNN Confusion Matrix'
    ax.set_title(title, color=PLOT_TEXT, fontweight='bold', fontsize=14)
    ax.tick_params(colors=PLOT_TEXT)

    plt.tight_layout()
    return fig


def get_complexity_table():
    """Generate complexity comparison table."""
    knn_result = get_latest_aggregation_result('knn')

    knn_time = 0.0
    knn_inference = 0.0

    if knn_result and 'result' in knn_result:
        knn_time = float(knn_result['result'].get('training_time', 0.0))
        knn_inference = float(knn_result['result'].get('inference_time_ms_per_sample', 0.0))

    data = [
        ["K-Nearest Neighbors", f"{knn_time:.3f}", f"{knn_inference:.3f}"],
    ]

    return pd.DataFrame(data, columns=["Method", "Training Time (s)",
                                        "Inference Time (ms/sample)"])


# ─── Master Update Function ──────────────────────────────────────────────────

def update_dashboard():
    """Update all dashboard components."""
    return (
        get_system_status_html(),
        get_client_table(),
        get_trust_scores_plot(),
        get_byzantine_report(),
        get_accuracy_trends_plot(),
        get_before_after_table(),
        get_snr_accuracy_table(),
        get_confusion_matrix_plot(),
        get_accuracy_vs_snr_plot(),
        get_complexity_table()
    )


# ─── Build Dashboard ─────────────────────────────────────────────────────────

def create_dashboard_interface() -> gr.Blocks:
    plt.style.use('default')

    with gr.Blocks(
        title="RadioFed Dashboard",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=DASHBOARD_CSS
    ) as dashboard:

        # ── Header ──
        gr.HTML("""
        <div class="dashboard-header">
            <h1>RadioFed Dashboard</h1>
            <p>Byzantine-Resilient Federated Learning &mdash; Real-Time Monitoring</p>
        </div>
        """)

        # Auto-refresh
        timer = gr.Timer(value=2, active=True)

        # ── Status Cards ──
        status_html = gr.HTML(value=get_system_status_html())

        with gr.Tabs():

            # ═══ Tab 1: Overview ═══
            with gr.TabItem("Overview"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Accuracy Trends")
                        accuracy_trends_plot = gr.Plot(label="Accuracy Over Rounds")
                    with gr.Column(scale=1):
                        gr.Markdown("### Before/After Aggregation")
                        before_after_table = gr.Dataframe(
                            headers=["Metric", "Before", "After", "Improvement"],
                            label="Latest Round"
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Confusion Matrix")
                        confusion_plot = gr.Plot(label="KNN Confusion Matrix")
                    with gr.Column():
                        gr.Markdown("### Accuracy vs SNR")
                        snr_plot = gr.Plot(label="Performance Across SNR Levels")

            # ═══ Tab 2: Clients ═══
            with gr.TabItem("Client Monitoring"):
                gr.Markdown("### Connected Clients")
                client_table = gr.Dataframe(
                    headers=["Client ID", "Status", "Trust Score", "Last Upload", "Samples"],
                    label="Client Registry"
                )
                gr.Markdown("### Trust Scores")
                trust_plot = gr.Plot(label="Client Trust Scores")

            # ═══ Tab 3: Byzantine Defense ═══
            with gr.TabItem("Byzantine Defense"):
                gr.Markdown("### Byzantine Fault Tolerance")
                gr.Markdown("""
                The system employs multiple defense mechanisms against Byzantine (malicious/faulty) clients:
                - **Krum Selection**: Selects updates closest to the geometric median
                - **Trimmed Mean**: Removes statistical outliers before aggregation
                - **Trust Scoring**: Maintains per-client reputation over rounds
                - **Anomaly Detection**: Statistical checks on feature distributions
                - **Cosine Filtering**: Rejects updates with low similarity to median
                """)
                byzantine_report = gr.Markdown("")
                gr.Markdown("### Trust Score Visualization")
                trust_viz_plot = gr.Plot(label="Trust Scores")

            # ═══ Tab 4: Metrics ═══
            with gr.TabItem("Detailed Metrics"):
                gr.Markdown("### Per-SNR Accuracy")
                snr_table = gr.Dataframe(
                    headers=["SNR (dB)", "Baseline (%)", "KNN (%)"],
                    label="Accuracy by SNR Level"
                )
                gr.Markdown("### Computation Complexity")
                complexity_table = gr.Dataframe(
                    headers=["Method", "Training Time (s)", "Inference Time (ms/sample)"],
                    label="Model Complexity"
                )

        # Wire up auto-refresh
        timer.tick(
            fn=update_dashboard,
            outputs=[
                status_html,
                client_table,
                trust_plot,
                byzantine_report,
                accuracy_trends_plot,
                before_after_table,
                snr_table,
                confusion_plot,
                snr_plot,
                complexity_table
            ]
        )

    return dashboard
