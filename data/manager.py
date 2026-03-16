"""
RadioFed Data Manager — Standalone Service (port 7862)

Separate site for downloading, inspecting, and partitioning RadioML datasets.
Runs independently from the central server and client.

Usage:
    python data/manager.py                  # http://localhost:7862
    python data/manager.py --port 7870      # custom port
"""

import os, sys, io, base64, json, logging, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fasthtml.common import *
from data.datasets import (
    DATASETS, ANALOG_FILTERS, get_dataset_status, download_dataset,
    load_dataset, partition_dataset, list_partitions,
)

logger = logging.getLogger(__name__)

# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = Style("""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}html{font-size:15px}
body{font-family:'Inter',system-ui,sans-serif;background:#09090b;color:#e4e4e7;min-height:100vh;line-height:1.6}
a{color:#818cf8;text-decoration:none}
.wrap{max-width:1100px;margin:0 auto;padding:20px 24px}
.hdr{padding:32px 0 24px;border-bottom:1px solid #27272a;margin-bottom:24px}
.hdr h1{font-size:1.75rem;font-weight:700;color:#f4f4f5;letter-spacing:-.02em}
.hdr p{color:#71717a;font-size:.9rem;margin-top:2px}
.cd{background:#18181b;border:1px solid #27272a;border-radius:10px;padding:20px;margin-bottom:16px}
.cd h3{font-size:.8rem;font-weight:600;color:#a1a1aa;margin-bottom:12px;text-transform:uppercase;letter-spacing:.4px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:768px){.g2{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th{text-align:left;padding:8px 12px;color:#71717a;font-weight:500;font-size:.72rem;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #27272a}
td{padding:8px 12px;border-bottom:1px solid #18181b}
tr:hover td{background:#1c1c1f}
input,select{background:#18181b;border:1px solid #27272a;border-radius:8px;padding:10px 14px;color:#e4e4e7;font-size:.85rem;width:100%;font-family:inherit}
input:focus,select:focus{outline:none;border-color:#818cf8;box-shadow:0 0 0 2px rgba(129,140,248,.15)}
label{display:block;color:#71717a;font-size:.78rem;margin-bottom:3px;font-weight:500}
.fg{margin-bottom:12px}.fr{display:flex;gap:12px}.fr>*{flex:1}
.btn{display:inline-flex;align-items:center;gap:6px;padding:10px 22px;border-radius:8px;font-weight:600;font-size:.875rem;border:none;cursor:pointer;transition:all .15s;font-family:inherit}
.btn-p{background:#6366f1;color:#fff}.btn-p:hover{background:#818cf8}
.btn-s{background:#27272a;color:#e4e4e7;border:1px solid #3f3f46}.btn-s:hover{border-color:#818cf8}
.btn-d{background:#ef4444;color:#fff}.btn-d:hover{background:#f87171}
.bg{display:inline-block;padding:3px 10px;border-radius:99px;font-size:.75rem;font-weight:500}
.bg-g{background:#052e16;color:#4ade80}.bg-r{background:#450a0a;color:#f87171}.bg-b{background:#172554;color:#60a5fa}.bg-y{background:#422006;color:#fbbf24}
.tg{color:#4ade80}.tr{color:#f87171}.ty{color:#fbbf24}.tm{color:#71717a}.tp{color:#818cf8}
.mt{margin-top:14px}
code{background:#27272a;padding:2px 6px;border-radius:4px;font-size:.82rem;color:#a78bfa}
.htmx-indicator{display:none}.htmx-request .htmx-indicator,.htmx-request.htmx-indicator{display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}
.sp{display:inline-block;width:14px;height:14px;border:2px solid #27272a;border-top-color:#818cf8;border-radius:50%;animation:spin .5s linear infinite;vertical-align:middle;margin-right:6px}
.fi{animation:fi .3s ease}@keyframes fi{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.section{margin-bottom:24px}
.mod-list{font-size:.75rem;color:#71717a;line-height:1.5;margin-top:8px;padding:8px 12px;background:#111113;border-radius:6px;border:1px solid #1f1f2e}
""")

# ── Helpers ──────────────────────────────────────────────────────────────────

PBG = "#09090b"; PFC = "#18181b"; PTX = "#e4e4e7"; PMU = "#71717a"

def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=PBG, edgecolor="none")
    plt.close(fig); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _plot_partition_balance(meta):
    """Bar chart of samples per modulation in a partition set."""
    mods = meta.get("modulations", [])
    if not mods:
        return None
    # Just show the modulation count as a simple bar
    fig, ax = plt.subplots(figsize=(7, max(2, len(mods) * 0.3)))
    fig.patch.set_facecolor(PBG); ax.set_facecolor(PFC)
    ax.tick_params(colors=PMU, labelsize=7)
    ax.grid(True, alpha=.08, color="#3f3f46")
    for s in ax.spines.values():
        s.set_color("#27272a")

    y = range(len(mods))
    # Even distribution assumption
    total = meta.get("total_samples", 0)
    per_mod = total // max(len(mods), 1)
    bars = ax.barh(list(y), [per_mod] * len(mods), color="#818cf8", alpha=0.7, height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(mods, fontsize=7)
    ax.set_xlabel("Samples (approx)", color=PMU, fontsize=8)
    for b in bars:
        ax.text(b.get_width() + per_mod * 0.02, b.get_y() + b.get_height() / 2,
                f"{per_mod:,}", va="center", color=PTX, fontsize=7)
    plt.tight_layout()
    return _b64(fig)


# ── Page Builders ────────────────────────────────────────────────────────────

def _build_page():
    status = get_dataset_status()
    parts = list_partitions()

    # ── Dataset Cards ──
    ds_cards = []
    for key, st in status.items():
        is_dl = st["downloaded"]
        badge = Span("Downloaded", cls="bg bg-g") if is_dl else Span("Not downloaded", cls="bg bg-r")
        size = f" ({st['size_mb']} MB)" if is_dl else ""
        n_mods = len(st["modulations"])
        snr_lo, snr_hi = st["snr_range"]

        analog_mods = list(ANALOG_FILTERS.get(key, {}).keys())
        analog_str = ", ".join(analog_mods) if analog_mods else "N/A"

        ds_cards.append(Div(
            # Header
            Div(Div(st["name"], style="font-weight:600;color:#e4e4e7;font-size:1.05rem"),
                badge,
                style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px"),
            # Info table
            Table(Tbody(
                Tr(Td("Classes", cls="tm"), Td(f"{n_mods} modulations")),
                Tr(Td("SNR", cls="tm"), Td(f"{snr_lo} to {snr_hi} dB, step {st['snr_step']}")),
                Tr(Td("Samples/key", cls="tm"), Td(f"{st['samples_per_key']:,}")),
                Tr(Td("Sample length", cls="tm"), Td(f"{st['sample_length']} I/Q points")),
                Tr(Td("Format", cls="tm"), Td(Code(st["format"]))),
                Tr(Td("File", cls="tm"), Td(
                    Span(f"{st['filename']}{size}", cls="tg") if is_dl else Code(st["filename"]))),
                Tr(Td("Partitions", cls="tm"), Td(
                    Span(f"{st['num_partitions']} created", cls="tg") if st["num_partitions"] else
                    Span("None", cls="tm"))),
                Tr(Td("Analog subset", cls="tm"), Td(analog_str, style="font-size:.8rem")),
                Tr(Td("Kaggle", cls="tm"), Td(Code(st["kaggle_slug"]), style="font-size:.75rem")),
            )),
            # Full modulation list
            Div(", ".join(st["modulations"]), cls="mod-list"),
            # Download button
            Div(
                Button("Download via Kaggle" if not is_dl else "Already downloaded",
                       cls="btn btn-p" if not is_dl else "btn btn-s",
                       hx_post="/a/download", hx_target=f"#dl-{key}",
                       hx_vals=json.dumps({"ds": key}),
                       hx_indicator=f"#dli-{key}",
                       **({"disabled": True} if is_dl else {})),
                Span(Span(cls="sp"), "Downloading...", id=f"dli-{key}",
                     cls="htmx-indicator tm", style="margin-left:8px"),
                Div(id=f"dl-{key}", cls="mt"),
                cls="mt"),
            cls="cd"))

    # ── Partition Tool ──
    ds_options = [Option(DATASETS[k]["name"], value=k) for k in DATASETS]
    partition_tool = Div(
        H3("Create Partitions"),
        P("Split a downloaded dataset into non-overlapping client partitions for federated simulation.",
          cls="tm", style="margin-bottom:14px;font-size:.85rem"),
        Div(
            Div(Label("Dataset"), Select(*ds_options, name="ds", id="pt-ds"), cls="fg"),
            Div(Label("Clients"),
                Input(name="n_clients", type="number", value="4", min="2", max="50",
                      id="pt-n", style="max-width:100px"), cls="fg"),
            Div(Label("Filter"),
                Select(Option("All modulations", value="all"),
                       Option("Analog only", value="analog"),
                       name="fmode", id="pt-fm"), cls="fg"),
            Div(Label("Distribution"),
                Select(Option("IID (uniform)", value="iid"),
                       Option("Non-IID (Dirichlet)", value="noniid"),
                       name="dist", id="pt-dist"), cls="fg"),
            cls="fr"),
        Div(Label("Dirichlet alpha (lower = more skewed, only for non-IID)"),
            Input(name="alpha", type="number", value="0.5", min="0.01", max="100", step="0.1",
                  id="pt-alpha", style="max-width:120px"),
            cls="fg"),
        Button("Partition", cls="btn btn-p",
               hx_post="/a/partition", hx_target="#pt-result",
               hx_include="#pt-ds,#pt-n,#pt-fm,#pt-dist,#pt-alpha",
               hx_indicator="#pt-spin"),
        Span(Span(cls="sp"), "Partitioning...", id="pt-spin",
             cls="htmx-indicator tm", style="margin-left:8px"),
        Div(id="pt-result", cls="mt"),
        cls="cd")

    # ── Existing Partitions ──
    part_section = _partition_table(parts)

    return Div(
        Div(H3("Available Datasets"),
            P("RadioML datasets for automatic modulation classification.",
              cls="tm", style="margin-bottom:14px;font-size:.85rem"),
            Div(*ds_cards, cls="g2"),
            cls="section"),
        partition_tool,
        part_section,
        cls="fi")


def _partition_table(parts=None):
    if parts is None:
        parts = list_partitions()
    if not parts:
        return Div(H3("Existing Partitions"),
                   P("No partitions yet. Download a dataset and create partitions above.", cls="tm"),
                   cls="cd", id="part-table")

    rows = []
    for key, info in parts.items():
        meta = info.get("meta", {})
        mods = meta.get("modulations", [])
        filt = meta.get("filter_mode", "—")
        total = meta.get("total_samples", 0)
        total_str = f"{total:,}" if isinstance(total, int) and total > 0 else "—"

        rows.append(Tr(
            Td(Code(key)),
            Td(str(info["count"])),
            Td(Span(filt, cls="bg bg-b" if filt == "all" else "bg bg-y")),
            Td(total_str),
            Td(f"{len(mods)}" if mods else "—"),
            Td(", ".join(mods[:6]) + ("..." if len(mods) > 6 else "") if mods else "—",
               style="font-size:.75rem;color:#71717a"),
        ))

    return Div(
        H3("Existing Partitions"),
        P(f"{sum(v['count'] for v in parts.values())} total partition files across "
          f"{len(parts)} dataset(s).", cls="tm", style="margin-bottom:10px;font-size:.85rem"),
        Table(
            Thead(Tr(Th("Dataset"), Th("Clients"), Th("Filter"), Th("Samples"),
                     Th("Classes"), Th("Modulations"))),
            Tbody(*rows)),
        cls="cd", id="part-table")


# ── App ──────────────────────────────────────────────────────────────────────

def create_data_manager_app(port=7862):
    app = FastHTML(debug=False, hdrs=[CSS])

    @app.get("/")
    def index():
        return Title("RadioFed Data Manager"), Div(Div(
            Div(H1("RadioFed Data Manager"),
                P("Download, inspect, and partition RadioML datasets for federated learning simulation"),
                cls="hdr"),
            _build_page(),
            cls="wrap"))

    @app.post("/a/download")
    def a_download(ds: str = ""):
        if not ds:
            return P("Select a dataset.", cls="ty")
        ok, msg = download_dataset(ds)
        cls = "bg bg-g" if ok else "bg bg-r"
        return Div(Span(msg, cls=cls, style="font-size:.85rem;padding:6px 14px"))

    @app.post("/a/partition")
    def a_partition(ds: str = "", n_clients: int = 4, fmode: str = "all",
                    dist: str = "iid", alpha: float = 0.5):
        if not ds:
            return P("Select a dataset.", cls="ty")
        ok, msg = partition_dataset(ds, num_clients=n_clients, filter_mode=fmode,
                                    distribution=dist, dirichlet_alpha=alpha)
        if ok:
            # Get meta for balance plot
            parts = list_partitions(ds)
            meta = parts.get(ds, {}).get("meta", {}) if parts else {}
            plot = _plot_partition_balance(meta)
            return Div(
                Span(msg, cls="bg bg-g", style="font-size:.85rem;padding:6px 14px"),
                Div(H3("Partition Balance"), Img(src=plot), cls="cd mt") if plot else "",
                _partition_table(), cls="mt")
        return Div(Span(msg, cls="bg bg-r", style="font-size:.85rem;padding:6px 14px"))

    return app


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RadioFed Data Manager")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()

    import uvicorn
    app = create_data_manager_app(port=args.port)
    print(f"RadioFed Data Manager running at http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
