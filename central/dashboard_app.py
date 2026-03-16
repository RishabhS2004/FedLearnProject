"""
RadioFed Dashboard — FastHTML + HTMX
Clean, minimal, production monitoring.
"""

import os, sys, io, base64, logging
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from fasthtml.common import *

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from central.state import (
    get_client_status, get_registry_stats,
    get_latest_aggregation_result, get_accuracy_trends,
    get_latest_round_metrics, get_auto_aggregation_state,
)
from central.byzantine import get_all_trust_scores, get_byzantine_aggregator

logger = logging.getLogger(__name__)

# ── Embedded CSS ─────────────────────────────────────────────────────────────

CSS = Style("""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html{font-size:15px}
body{font-family:'Inter',system-ui,sans-serif;background:#09090b;color:#e4e4e7;min-height:100vh;line-height:1.6}
a{color:#818cf8;text-decoration:none}
.wrap{max-width:1280px;margin:0 auto;padding:20px 24px}

/* Header */
.hdr{padding:32px 0 24px;border-bottom:1px solid #27272a;margin-bottom:24px}
.hdr h1{font-size:1.75rem;font-weight:700;color:#f4f4f5;letter-spacing:-.02em}
.hdr p{color:#71717a;font-size:.9rem;margin-top:2px}

/* Stats row */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:28px}
.st{background:#18181b;border:1px solid #27272a;border-radius:10px;padding:16px 18px;text-align:center}
.st:hover{border-color:#3f3f46}
.st .v{font-size:1.65rem;font-weight:700;line-height:1.1}
.st .l{font-size:.75rem;color:#71717a;margin-top:4px;text-transform:uppercase;letter-spacing:.5px}

/* Tabs */
.tabs{display:flex;gap:2px;margin-bottom:24px;border-bottom:1px solid #27272a;padding-bottom:0}
.tb{padding:10px 18px;color:#a1a1aa;font-size:.875rem;font-weight:500;cursor:pointer;border:none;background:none;border-bottom:2px solid transparent;transition:all .15s}
.tb:hover{color:#f4f4f5}
.tb.on{color:#818cf8;border-bottom-color:#818cf8}

/* Cards */
.cd{background:#18181b;border:1px solid #27272a;border-radius:10px;padding:20px;margin-bottom:16px}
.cd h3{font-size:.95rem;font-weight:600;color:#a1a1aa;margin-bottom:12px;text-transform:uppercase;letter-spacing:.4px;font-size:.8rem}
.cd img{width:100%;border-radius:6px;display:block}

/* Grid */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:768px){.g2{grid-template-columns:1fr}}

/* Tables */
table{width:100%;border-collapse:collapse;font-size:.875rem}
th{text-align:left;padding:8px 12px;color:#71717a;font-weight:500;font-size:.75rem;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #27272a}
td{padding:8px 12px;border-bottom:1px solid #18181b}
tr:hover td{background:#1c1c1f}

/* Badges */
.bg{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.75rem;font-weight:500}
.bg-g{background:#052e16;color:#4ade80}
.bg-y{background:#422006;color:#fbbf24}
.bg-r{background:#450a0a;color:#f87171}
.bg-b{background:#172554;color:#60a5fa}

/* Text colors */
.tg{color:#4ade80}.tr{color:#f87171}.ty{color:#fbbf24}.tb2{color:#60a5fa}.tm{color:#71717a}

/* Utilities */
.mt{margin-top:16px}
.mb{margin-bottom:16px}
code{background:#27272a;padding:2px 6px;border-radius:4px;font-size:.85rem;color:#a78bfa}
pre{background:#18181b;border:1px solid #27272a;border-radius:8px;padding:14px;overflow-x:auto;font-size:.82rem;color:#a1a1aa}

/* HTMX */
.htmx-indicator{display:none}.htmx-request .htmx-indicator,.htmx-request.htmx-indicator{display:inline}
@keyframes spin{to{transform:rotate(360deg)}}
.sp{display:inline-block;width:14px;height:14px;border:2px solid #27272a;border-top-color:#818cf8;border-radius:50%;animation:spin .5s linear infinite}

/* Animation */
.fi{animation:fi .3s ease}
@keyframes fi{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
""")

# ── Plot Config ──────────────────────────────────────────────────────────────

PBG = "#09090b"
PFC = "#18181b"
PTX = "#e4e4e7"
PGR = "#27272a"
PMU = "#71717a"
SNR = list(range(-20, 20, 2))

def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=PBG, edgecolor="none")
    plt.close(fig); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def _ax(ax):
    ax.set_facecolor(PFC)
    ax.tick_params(colors=PMU, labelsize=7)
    ax.grid(True, alpha=.08, color="#3f3f46")
    for s in ax.spines.values(): s.set_color("#27272a")

def _plot_trends():
    t = get_accuracy_trends()
    fig, ax = plt.subplots(figsize=(7, 3.2)); fig.patch.set_facecolor(PBG); _ax(ax)
    if not t["rounds"]:
        ax.text(.5, .5, "No data yet", ha="center", va="center",
                color=PMU, fontsize=12, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return _b64(fig)
    # De-duplicate: keep only the last entry per round, drop zeros
    seen = {}
    for r, b, a in zip(t["rounds"], t["knn_before"], t["knn_after"]):
        if a == 0 and b == 0:
            continue
        seen[r] = (b, a)
    if not seen:
        ax.text(.5, .5, "No data yet", ha="center", va="center",
                color=PMU, fontsize=12, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return _b64(fig)
    rounds = sorted(seen.keys())
    bef = [seen[r][0] for r in rounds]
    aft = [seen[r][1] for r in rounds]
    # Replace 0.0 before-values with after-values (first round has no prior model)
    bef = [a if b == 0 else b for b, a in zip(bef, aft)]
    # Use sequential x-axis (1, 2, 3...) for clean plotting
    x = list(range(1, len(rounds) + 1))
    ax.plot(x, bef, "o--", color="#f97316", lw=1.5, ms=5, alpha=.8, label="Before Agg")
    ax.plot(x, aft, "o-", color="#4ade80", lw=2.2, ms=6, label="After Agg")
    ax.fill_between(x, bef, aft, alpha=.08, color="#4ade80")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", color=PMU, fontsize=8)
    ax.set_xlabel("Round", color=PMU, fontsize=8)
    ax.set_xticks(x)
    ax.legend(fontsize=7, facecolor=PFC, edgecolor=PGR, labelcolor=PTX)
    return _b64(fig)

def _plot_snr():
    knn=get_latest_aggregation_result("knn"); fig,ax=plt.subplots(figsize=(7,3.2)); fig.patch.set_facecolor(PBG); _ax(ax)
    bl=[50.0]*len(SNR); ka=[]
    for s in SNR:
        a=0.0
        if knn and "result" in knn:
            ps=knn["result"].get("per_snr_accuracy",{})
            a=ps.get(s,ps.get(float(s),ps.get(str(s),0.0)))
            if isinstance(a,(int,float)) and 0<a<=1: a*=100
        ka.append(a)
    ax.plot(SNR,bl,"--",color="#3f3f46",lw=1,alpha=.7,label="Baseline")
    ax.plot(SNR,ka,"s-",color="#818cf8",lw=2,ms=4,label="KNN")
    ax.fill_between(SNR,bl,ka,alpha=.05,color="#818cf8")
    ax.set_ylim(0,105); ax.set_xlabel("SNR (dB)",color=PMU,fontsize=8); ax.set_ylabel("Accuracy %",color=PMU,fontsize=8)
    ax.legend(fontsize=7,facecolor=PFC,edgecolor=PGR,labelcolor=PTX)
    return _b64(fig)

def _plot_cm():
    knn=get_latest_aggregation_result("knn"); fig,ax=plt.subplots(figsize=(4.5,3.5)); fig.patch.set_facecolor(PBG); ax.set_facecolor(PFC)
    cm=np.array([[0,0],[0,0]])
    if knn and "result" in knn:
        d=knn["result"].get("confusion_matrix")
        if d: cm=np.array(d)
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["AM","FM"],yticklabels=["AM","FM"],ax=ax,linewidths=.5,linecolor=PGR,annot_kws={"size":14})
    ax.set_xlabel("Predicted",color=PMU,fontsize=9); ax.set_ylabel("Actual",color=PMU,fontsize=9); ax.tick_params(colors=PTX)
    return _b64(fig)

def _plot_trust():
    sc=get_all_trust_scores(); fig,ax=plt.subplots(figsize=(7,max(2,len(sc)*.55+.8))); fig.patch.set_facecolor(PBG); _ax(ax)
    if not sc:
        ax.text(.5,.5,"No clients",ha="center",va="center",color=PMU,fontsize=12,transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([])
        return _b64(fig)
    ids=list(sc.keys()); vals=list(sc.values())
    cs=["#4ade80" if v>=.5 else "#fbbf24" if v>=.3 else "#f87171" for v in vals]
    bars=ax.barh(ids,vals,color=cs,alpha=.85,height=.5)
    ax.axvline(.3,color="#f87171",ls="--",alpha=.4,lw=1)
    ax.set_xlim(0,1.05)
    for b,v in zip(bars,vals): ax.text(b.get_width()+.02,b.get_y()+b.get_height()/2,f"{v:.2f}",va="center",color=PTX,fontsize=8)
    return _b64(fig)

# ── Components ───────────────────────────────────────────────────────────────

def _stat(val, label, color="#818cf8"):
    return Div(Div(str(val),cls="v",style=f"color:{color}"),Div(label,cls="l"),cls="st")

def _client_rows():
    clients=get_client_status(); trust=get_all_trust_scores()
    if not clients: return [Tr(Td("No clients connected",colspan="4",cls="tm",style="text-align:center"))]
    rows=[]
    for c in clients:
        cid=c["client_id"]; st=c.get("status","connected"); t=trust.get(cid,.5)
        if st=="weights_uploaded": badge=Span("Uploaded",cls="bg bg-g")
        elif st=="training": badge=Span("Training",cls="bg bg-y")
        else: badge=Span("Idle",cls="bg bg-b")
        tcls="tg" if t>=.5 else "ty" if t>=.3 else "tr"
        rows.append(Tr(Td(cid),Td(badge),Td(Span(f"{t:.2f}",cls=tcls)),Td(str(c.get("n_samples",0)))))
    return rows

# ── Partials ─────────────────────────────────────────────────────────────────

def _stats_partial():
    s=get_registry_stats(); a=get_auto_aggregation_state()
    knn=get_latest_aggregation_result("knn"); acc="—"
    if knn and "result" in knn:
        v=knn["result"].get("accuracy",0)
        if v: acc=f"{v*100:.1f}%"
    ts=get_all_trust_scores(); low=sum(1 for v in ts.values() if v<.3)
    avg_t=f"{np.mean(list(ts.values())):.2f}" if ts else "—"
    return Div(
        _stat("●","Server","#4ade80"),
        _stat(s["total_clients"],"Clients"),
        _stat(f'{s["total_samples"]:,}',"Samples","#38bdf8"),
        _stat(a.get("current_round",0),"Round","#a78bfa"),
        _stat(f'{a.get("pending_uploads",0)}/{a.get("threshold",2)}',"Uploads","#fbbf24"),
        _stat(acc,"Accuracy","#4ade80"),
        _stat(avg_t,"Avg Trust"),
        _stat(low,"Flagged","#f87171" if low else "#4ade80"),
        cls="stats",id="stats"
    )

def _overview():
    ba = get_latest_round_metrics()
    if ba:
        kb = ba["before"]["knn_accuracy"]; ka = ba["after"]["knn_accuracy"]
        imp = ba["improvement"]["knn"]
        ic = "tg" if imp >= 0 else "tr"
        istr = f"+{imp:.2%}" if imp >= 0 else f"{imp:.2%}"
        ba_table = Table(
            Thead(Tr(Th("Metric"), Th("Before"), Th("After"), Th("Change"))),
            Tbody(Tr(Td("KNN Accuracy"), Td(f"{kb:.2%}"), Td(f"{ka:.2%}"),
                     Td(Span(istr, cls=ic)))))
    else:
        ba_table = P("Waiting for first aggregation.", cls="tm")

    # Get latest aggregation result for model details
    knn = get_latest_aggregation_result("knn")
    r = knn["result"] if knn and "result" in knn else {}

    # Model configuration table
    n_neighbors = r.get("n_neighbors", 5)
    n_clients = r.get("num_clients", 0)
    total_samples = r.get("total_samples", 0)
    feat_dim = r.get("feature_dim", 0)
    n_test = r.get("n_test_samples", 0)
    acc = r.get("accuracy", 0)
    train_t = r.get("training_time", 0)
    inf_t = r.get("inference_time_ms_per_sample", 0)

    model_table = Table(
        Thead(Tr(Th("Parameter"), Th("Value"))),
        Tbody(
            Tr(Td("Algorithm"), Td("K-Nearest Neighbors")),
            Tr(Td("k (neighbors)"), Td(str(n_neighbors))),
            Tr(Td("Distance metric"), Td("Minkowski (p=2)")),
            Tr(Td("Weight function"), Td("uniform")),
            Tr(Td("Feature dimensions"), Td(f"{feat_dim}D")),
            Tr(Td("Training samples"), Td(f"{total_samples:,}")),
            Tr(Td("Test samples"), Td(f"{n_test:,}" if n_test else "—")),
            Tr(Td("Participating clients"), Td(str(n_clients))),
            Tr(Td("Test accuracy"), Td(Span(f"{acc:.4f}", cls="tg" if acc > 0.7 else "ty") if acc else "—")),
            Tr(Td("Training time"), Td(f"{train_t:.4f} s")),
            Tr(Td("Inference latency"), Td(f"{inf_t:.4f} ms/sample")),
        ))

    # Confusion matrix breakdown
    cm = r.get("confusion_matrix")
    cm_detail = P("No confusion data.", cls="tm")
    if cm and len(cm) == 2:
        cm = np.array(cm)
        tp_am = cm[0][0]; fn_am = cm[0][1]; fp_am = cm[1][0]; tn_am = cm[1][1]
        prec_am = tp_am / (tp_am + fp_am + 1e-9)
        rec_am = tp_am / (tp_am + fn_am + 1e-9)
        f1_am = 2 * prec_am * rec_am / (prec_am + rec_am + 1e-9)
        prec_fm = tn_am / (tn_am + fn_am + 1e-9)
        rec_fm = tn_am / (tn_am + fp_am + 1e-9)
        f1_fm = 2 * prec_fm * rec_fm / (prec_fm + rec_fm + 1e-9)
        total = cm.sum()
        cm_detail = Table(
            Thead(Tr(Th("Class"), Th("Precision"), Th("Recall"), Th("F1"), Th("Support"))),
            Tbody(
                Tr(Td("AM"), Td(f"{prec_am:.4f}"), Td(f"{rec_am:.4f}"),
                   Td(f"{f1_am:.4f}"), Td(str(tp_am + fn_am))),
                Tr(Td("FM"), Td(f"{prec_fm:.4f}"), Td(f"{rec_fm:.4f}"),
                   Td(f"{f1_fm:.4f}"), Td(str(fp_am + tn_am))),
                Tr(Td(B("Weighted Avg")),
                   Td(f"{(prec_am*(tp_am+fn_am)+prec_fm*(fp_am+tn_am))/total:.4f}"),
                   Td(f"{(rec_am*(tp_am+fn_am)+rec_fm*(fp_am+tn_am))/total:.4f}"),
                   Td(f"{(f1_am*(tp_am+fn_am)+f1_fm*(fp_am+tn_am))/total:.4f}"),
                   Td(str(total))),
            ))

    # Available classifiers info
    classifiers_info = Table(
        Thead(Tr(Th("Classifier"), Th("Type"), Th("Key Parameters"))),
        Tbody(
            Tr(Td("KNN"), Td("Instance-based"), Td("k, weights, metric")),
            Tr(Td("Decision Tree"), Td("Tree-based"), Td("max_depth, min_samples_split")),
            Tr(Td("Random Forest"), Td("Ensemble (bagging)"), Td("n_estimators, max_depth")),
            Tr(Td("Gradient Boosting"), Td("Ensemble (boosting)"), Td("n_estimators, learning_rate, max_depth")),
            Tr(Td("SVM"), Td("Kernel-based"), Td("kernel (rbf/linear/poly), C")),
            Tr(Td("Logistic Regression"), Td("Linear"), Td("C, max_iter")),
            Tr(Td("Naive Bayes"), Td("Probabilistic"), Td("Gaussian prior")),
            Tr(Td("MLP Neural Net"), Td("Neural network"), Td("hidden_layers, max_iter")),
        ))

    return Div(
        # Row 1: plots
        Div(Div(H3("Accuracy Trends"), Img(src=_plot_trends()), cls="cd"),
            Div(H3("Accuracy vs SNR"), Img(src=_plot_snr()), cls="cd"),
            cls="g2"),
        # Row 2: confusion + before/after
        Div(Div(H3("Confusion Matrix"), Img(src=_plot_cm()), cls="cd"),
            Div(H3("Aggregation Results"), ba_table, cls="cd"),
            cls="g2"),
        # Row 3: model details + per-class metrics
        Div(Div(H3("Global Model Configuration"), model_table, cls="cd"),
            Div(H3("Per-Class Metrics"), cm_detail, cls="cd"),
            cls="g2"),
        # Row 4: available classifiers
        Div(H3("Available Classifiers (Client-Side)"), classifiers_info, cls="cd"),
        id="panel", cls="fi"
    )

def _clients():
    return Div(
        Div(H3("Client Registry"),
            Table(Thead(Tr(Th("ID"),Th("Status"),Th("Trust"),Th("Samples"))),Tbody(*_client_rows())),cls="cd"),
        Div(H3("Trust Scores"),Img(src=_plot_trust()),cls="cd"),
        id="panel",cls="fi"
    )

def _byzantine():
    agg=get_byzantine_aggregator(); log=agg.get_aggregation_log()
    if not log:
        body=P("No Byzantine filtering yet. Waiting for aggregation.",cls="tm")
    else:
        lt=log[-1]; rej=lt.get("rejected_clients",[])
        rr=[Tr(Td(r["client_id"]),Td(r["reason"])) for r in rej] if rej else [Tr(Td("None",colspan="2",cls="tm"))]
        body=Div(
            Table(Thead(Tr(Th("Property"),Th("Value"))),Tbody(
                Tr(Td("Strategy"),Td(Code(lt.get("strategy","—")))),
                Tr(Td("Accepted"),Td(Span(str(lt.get("accepted_count",0)),cls="tg"))),
                Tr(Td("Rejected"),Td(Span(str(lt.get("rejected_count",0)),cls="tr"))),
            )),
            H3("Rejected Clients",cls="mt"),
            Table(Thead(Tr(Th("Client"),Th("Reason"))),Tbody(*rr)),
        )
    return Div(
        Div(H3("Byzantine Defense"),
            P("Krum · Trimmed Mean · Trust Scoring · Anomaly Detection · Cosine Filtering",cls="tm",style="margin-bottom:12px"),
            body,cls="cd"),
        Div(H3("Trust Distribution"),Img(src=_plot_trust()),cls="cd"),
        id="panel",cls="fi"
    )

def _metrics():
    knn=get_latest_aggregation_result("knn"); bl=50.0
    rows=[]
    for s in SNR:
        a=0.0
        if knn and "result" in knn:
            ps=knn["result"].get("per_snr_accuracy",{})
            a=ps.get(s,ps.get(float(s),ps.get(str(s),0.0)))
            if isinstance(a,(int,float)) and 0<a<=1: a*=100
        rows.append(Tr(Td(str(s)),Td(f"{bl:.0f}"),Td(f"{a:.1f}")))
    kt=ki=0.0
    if knn and "result" in knn: kt=float(knn["result"].get("training_time",0)); ki=float(knn["result"].get("inference_time_ms_per_sample",0))
    return Div(Div(
        Div(H3("Per-SNR Accuracy"),
            Div(Table(Thead(Tr(Th("SNR (dB)"),Th("Baseline"),Th("KNN"))),Tbody(*rows)),style="max-height:360px;overflow-y:auto"),cls="cd"),
        Div(H3("Complexity"),
            Table(Thead(Tr(Th("Method"),Th("Train (s)"),Th("Inference (ms)"))),
                  Tbody(Tr(Td("KNN"),Td(f"{kt:.3f}"),Td(f"{ki:.3f}")))),cls="cd"),
        cls="g2"),id="panel",cls="fi")

def _plot_flow():
    """Render clean system flow diagram — vertical pipeline, no overlaps."""
    fig, ax = plt.subplots(figsize=(9, 10))
    fig.patch.set_facecolor(PBG); ax.set_facecolor(PBG)
    ax.set_xlim(0, 12); ax.set_ylim(0, 13); ax.axis("off")

    # Box styles
    def _box(ec="#818cf8", lw=1.5):
        return dict(boxstyle="round,pad=.45", facecolor="#18181b", edgecolor=ec, linewidth=lw)
    _pill = dict(boxstyle="round,pad=.25", facecolor="#27272a", edgecolor="#3f3f46", linewidth=.8)
    _ar = dict(arrowstyle="->,head_width=.25,head_length=.15", color="#4f4f6f", lw=1.2)
    _ar_g = dict(arrowstyle="->,head_width=.25,head_length=.15", color="#4ade80", lw=1.5)
    _ar_r = dict(arrowstyle="->,head_width=.25,head_length=.15", color="#f87171", lw=1.2, ls="--")

    # Row positions (top to bottom)
    Y = [12, 10.3, 8.6, 6.9, 5.0, 3.2, 1.5]
    CX = [2.5, 5, 7.5, 10]  # client x positions

    # ── Row 0: Title ──
    ax.text(6, Y[0], "RadioFed System Pipeline", ha="center", fontsize=15,
            fontweight="bold", color="#f4f4f5", family="sans-serif")

    # ── Row 1: Clients ──
    ax.text(.6, Y[1], "1", fontsize=10, fontweight="bold", color="#818cf8",
            bbox=dict(boxstyle="circle,pad=.2", fc="#1e1b4b", ec="#818cf8", lw=1.2))
    ax.text(1.3, Y[1], "Data Partitioning", fontsize=9, color="#a1a1aa", va="center")
    for i, x in enumerate(CX):
        ec = "#f87171" if i == 3 else "#818cf8"
        lbl = f"Client {i+1}" if i < 3 else "Client 4"
        ax.text(x, Y[1] - .7, lbl, ha="center", fontsize=8.5, color=PTX, bbox=_box(ec))
    ax.text(10, Y[1] - 1.15, "(Byzantine)", ha="center", fontsize=7, color="#f87171")

    # ── Row 2: Feature Extraction ──
    ax.text(.6, Y[2], "2", fontsize=10, fontweight="bold", color="#818cf8",
            bbox=dict(boxstyle="circle,pad=.2", fc="#1e1b4b", ec="#818cf8", lw=1.2))
    ax.text(1.3, Y[2], "Feature Extraction", fontsize=9, color="#a1a1aa", va="center")
    for x in CX:
        ax.annotate("", xy=(x, Y[2] - .3), xytext=(x, Y[1] - 1.05), arrowprops=_ar)
        ax.text(x, Y[2] - .55, "16D features", ha="center", fontsize=7, color=PMU, bbox=_pill)

    # ── Row 3: Local Training ──
    ax.text(.6, Y[3], "3", fontsize=10, fontweight="bold", color="#818cf8",
            bbox=dict(boxstyle="circle,pad=.2", fc="#1e1b4b", ec="#818cf8", lw=1.2))
    ax.text(1.3, Y[3], "Local Training", fontsize=9, color="#a1a1aa", va="center")
    for x in CX:
        ax.annotate("", xy=(x, Y[3] - .2), xytext=(x, Y[2] - .85), arrowprops=_ar)
    for x in CX[:3]:
        ax.text(x, Y[3] - .5, "KNN  DT  RF  GB\nSVM  LR  NB  MLP", ha="center",
                fontsize=6.5, color="#e4e4e7", bbox=_pill, linespacing=1.5)
    ax.text(CX[3], Y[3] - .5, "Poisoned\nmodel", ha="center", fontsize=6.5,
            color="#f87171", bbox=dict(boxstyle="round,pad=.25", fc="#1a0a0a", ec="#f87171", lw=.8))

    # ── Row 4: Upload to Server ──
    ax.text(.6, Y[4], "4", fontsize=10, fontweight="bold", color="#818cf8",
            bbox=dict(boxstyle="circle,pad=.2", fc="#1e1b4b", ec="#818cf8", lw=1.2))
    ax.text(1.3, Y[4], "Upload to Server", fontsize=9, color="#a1a1aa", va="center")
    for x in CX:
        ax.annotate("", xy=(6, Y[4] - .5), xytext=(x, Y[3] - 1.0), arrowprops=_ar)
    # Server box
    ax.text(6, Y[4] - .7, "Central Server", ha="center", fontsize=11, fontweight="bold",
            color="#818cf8", bbox=_box("#818cf8", 2.5))

    # ── Row 5: Byzantine Defense ──
    ax.text(.6, Y[5], "5", fontsize=10, fontweight="bold", color="#f87171",
            bbox=dict(boxstyle="circle,pad=.2", fc="#1a0505", ec="#f87171", lw=1.2))
    ax.text(1.3, Y[5], "Byzantine Defense", fontsize=9, color="#a1a1aa", va="center")
    ax.annotate("", xy=(6, Y[5]), xytext=(6, Y[4] - 1.05), arrowprops=_ar)
    steps = ["Krum", "Trust", "Anomaly", "Cosine"]
    for j, (s, sx) in enumerate(zip(steps, [3.5, 5, 6.8, 8.5])):
        ax.text(sx, Y[5] - .55, s, ha="center", fontsize=7.5, color="#fbbf24",
                bbox=dict(boxstyle="round,pad=.2", fc="#1a1500", ec="#fbbf24", lw=.6))
        if j < 3:
            ax.annotate("", xy=(sx + .7, Y[5] - .55), xytext=(sx + .3, Y[5] - .55),
                        arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=.8))
    ax.text(10.2, Y[5] - .55, "Client 4\nrejected", ha="center", fontsize=7,
            color="#f87171", fontweight="600")

    # ── Row 6: Aggregation & Distribution ──
    ax.text(.6, Y[6], "6", fontsize=10, fontweight="bold", color="#4ade80",
            bbox=dict(boxstyle="circle,pad=.2", fc="#021a0a", ec="#4ade80", lw=1.2))
    ax.text(1.3, Y[6], "Aggregate & Distribute", fontsize=9, color="#a1a1aa", va="center")
    ax.text(6, Y[6] - .1, "Global Model", ha="center", fontsize=10, fontweight="bold",
            color="#4ade80", bbox=_box("#4ade80", 2))
    for x in CX[:3]:
        ax.annotate("", xy=(x, Y[6] - .5), xytext=(6, Y[6] - .4), arrowprops=_ar_g)
    ax.annotate("", xy=(CX[3], Y[6] - .3), xytext=(6, Y[6] - .3), arrowprops=_ar_r)
    ax.text(CX[3], Y[6] - .55, "blocked", ha="center", fontsize=7, color="#f87171")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return _b64(fig)


def _howitworks():
    """How It Works tab with flow diagram and explanation."""
    return Div(
        Div(H3("System Architecture"),Img(src=_plot_flow()),cls="cd"),
        Div(H3("How RadioFed Works"),
            Div(
                Div(
                    H3("1. Data Partitioning"),
                    P("RadioML 2016.10a dataset is split into non-overlapping partitions. "
                      "Each client receives a unique subset for training.",cls="tm"),cls="cd"),
                Div(
                    H3("2. Feature Extraction"),
                    P("16-dimensional features are extracted from I/Q samples: "
                      "statistical moments (I/Q channels), spectral features (FFT), "
                      "and temporal features (zero-crossing rate, energy).",cls="tm"),cls="cd"),
                Div(
                    H3("3. Local Training"),
                    P("Clients train one or more classifiers — KNN, Decision Tree, Random Forest, "
                      "SVM, Logistic Regression, Naive Bayes — on their local data.",cls="tm"),cls="cd"),
                Div(
                    H3("4. Model Upload"),
                    P("Trained models, features, and labels are uploaded to the central server "
                      "via REST API. Trust scores are initialized for new clients.",cls="tm"),cls="cd"),
                Div(
                    H3("5. Byzantine Defense"),
                    P("Multi-layer protection filters malicious updates: "
                      "Krum selection (geometric median), trimmed mean, statistical anomaly detection, "
                      "cosine similarity filtering, and per-client trust scoring.",cls="tm"),cls="cd"),
                Div(
                    H3("6. Secure Aggregation"),
                    P("Accepted client data is merged and a global model is retrained. "
                      "Trust scores are updated — honest clients gain reputation, "
                      "malicious ones are progressively excluded.",cls="tm"),cls="cd"),
                cls="g2"
            ),
            cls="cd"),
        Div(H3("Manim Animations"),
            _render_videos(),
            cls="cd"),
        id="panel",cls="fi"
    )


def _render_videos():
    vd=os.path.join(os.path.dirname(os.path.dirname(__file__)),"static","videos")
    vids=sorted([f for f in os.listdir(vd) if f.endswith(".mp4")]) if os.path.isdir(vd) else []
    if not vids:
        return Div(P("No animations found.",cls="tm"))
    # Ordered nicely
    order=["FederatedLearningFlow","SignalClassification","ByzantineDetection","AggregationProcess","TrustEvolution"]
    ordered=[v for o in order for v in vids if o in v]
    for v in vids:
        if v not in ordered: ordered.append(v)
    titles={
        "FederatedLearningFlow":"Federated Learning Round",
        "SignalClassification":"Signal Classification Pipeline",
        "ByzantineDetection":"Byzantine Fault Detection",
        "AggregationProcess":"Data-Centric Aggregation",
        "TrustEvolution":"Trust Score Evolution",
    }
    vid_css = ("width:100%;border-radius:8px;display:block;"
               "object-fit:contain;background:#1c1c2e")

    def _vcard(v, t, full=False):
        return Div(
            P(t, style="font-size:.75rem;font-weight:600;color:#a1a1aa;"
              "text-transform:uppercase;letter-spacing:.4px;margin-bottom:8px"),
            Video(Source(src=f"/static/videos/{v}", type="video/mp4"),
                  autoplay=True, loop=True, muted=True, playsinline=True,
                  style=vid_css),
            cls="cd", style="padding:14px;overflow:hidden")

    cards = []
    for v in ordered:
        name = v.replace(".mp4", "")
        t = titles.get(name, name.replace("_", " ").title())
        cards.append((v, t))

    parts = []
    # First video — full width hero
    if len(cards) >= 1:
        parts.append(_vcard(*cards[0], full=True))
    # Rest in 2-col grid, each with constrained max-height so they stay sharp
    for i in range(1, len(cards), 2):
        pair = [_vcard(*cards[j], full=False) for j in range(i, min(i+2, len(cards)))]
        parts.append(Div(*pair, cls="g2"))
    return Div(*parts)

# ── App ──────────────────────────────────────────────────────────────────────

def create_dashboard_app(port=7860):
    app = FastHTML(debug=False, hdrs=[CSS], static_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),"static"))

    TAB_JS = Script("""
    document.addEventListener('click', e => {
        if(e.target.classList.contains('tb')){
            document.querySelectorAll('.tb').forEach(t=>t.classList.remove('on'));
            e.target.classList.add('on');
        }
    });
    """)

    @app.get("/")
    def index():
        return Title("RadioFed"), Div(
            Div(
                Div(H1("RadioFed"),P("Byzantine-Resilient Federated Learning — Monitoring Dashboard"),cls="hdr"),
                _stats_partial(),
                Div(
                    A("Overview",cls="tb on",hx_get="/p/overview",hx_target="#panel",hx_swap="outerHTML"),
                    A("Clients",cls="tb",hx_get="/p/clients",hx_target="#panel",hx_swap="outerHTML"),
                    A("Byzantine",cls="tb",hx_get="/p/byzantine",hx_target="#panel",hx_swap="outerHTML"),
                    A("Metrics",cls="tb",hx_get="/p/metrics",hx_target="#panel",hx_swap="outerHTML"),
                    A("How It Works",cls="tb",hx_get="/p/howitworks",hx_target="#panel",hx_swap="outerHTML"),
                    cls="tabs"),
                TAB_JS,
                _overview(),
                cls="wrap",
            ),
            Div(hx_get="/x/stats",hx_trigger="every 3s",hx_target="#stats",hx_swap="outerHTML",style="display:none"),
        )

    @app.get("/p/overview")
    def p_overview(): return _overview()
    @app.get("/p/clients")
    def p_clients(): return _clients()
    @app.get("/p/byzantine")
    def p_byzantine(): return _byzantine()
    @app.get("/p/metrics")
    def p_metrics(): return _metrics()
    @app.get("/p/howitworks")
    def p_howitworks(): return _howitworks()
    @app.get("/x/stats")
    def x_stats(): return _stats_partial()

    # Serve video files
    @app.get("/static/videos/{fname:path}")
    async def serve_video(fname: str):
        from starlette.responses import FileResponse
        vpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "videos", fname)
        if os.path.exists(vpath):
            return FileResponse(vpath, media_type="video/mp4")
        return FileResponse(vpath)

    return app
