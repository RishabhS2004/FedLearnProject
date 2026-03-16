"""
RadioFed Client — FastHTML + HTMX
8 ML models with hyperparameter tuning, full metrics.
"""

import os, sys, io, base64, logging, pickle
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from fasthtml.common import *

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.dataset_loader import load_radioml_dataset, get_dataset_info, flatten_dataset
from client.feature_extract import normalize_features, extract_features_from_iq, extract_features
from client.train import (
    train_knn_model, train_dt_model, train_rf_model, train_gb_model,
    train_svm_model, train_lr_model, train_nb_model, train_mlp_model,
    _save_model,
)
from client.sync import check_server_status, download_global_model
from client.state import load_config, save_config, save_metrics

logger = logging.getLogger(__name__)

MODELS = {
    "knn":  "KNN",
    "dt":   "Decision Tree",
    "rf":   "Random Forest",
    "gb":   "Gradient Boosting",
    "svm":  "SVM",
    "lr":   "Logistic Regression",
    "nb":   "Naive Bayes",
    "mlp":  "MLP Neural Net",
}

CSS = Style("""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}html{font-size:15px}
body{font-family:'Inter',system-ui,sans-serif;background:#09090b;color:#e4e4e7;min-height:100vh;line-height:1.6}
a{color:#818cf8;text-decoration:none}.wrap{max-width:1060px;margin:0 auto;padding:20px 24px}
.hdr{padding:32px 0 24px;border-bottom:1px solid #27272a;margin-bottom:24px}
.hdr h1{font-size:1.75rem;font-weight:700;color:#f4f4f5;letter-spacing:-.02em}
.hdr p{color:#71717a;font-size:.9rem;margin-top:2px}
.tabs{display:flex;gap:2px;margin-bottom:24px;border-bottom:1px solid #27272a}
.tb{padding:10px 18px;color:#a1a1aa;font-size:.875rem;font-weight:500;cursor:pointer;border:none;background:none;border-bottom:2px solid transparent;transition:all .15s}
.tb:hover{color:#f4f4f5}.tb.on{color:#818cf8;border-bottom-color:#818cf8}
.cd{background:#18181b;border:1px solid #27272a;border-radius:10px;padding:20px;margin-bottom:16px}
.cd h3{font-size:.8rem;font-weight:600;color:#a1a1aa;margin-bottom:12px;text-transform:uppercase;letter-spacing:.4px}
.cd img{width:100%;border-radius:6px;display:block}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.g4{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px}
@media(max-width:768px){.g2,.g3,.g4{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th{text-align:left;padding:7px 10px;color:#71717a;font-weight:500;font-size:.72rem;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #27272a}
td{padding:7px 10px;border-bottom:1px solid #18181b}
tr:hover td{background:#1c1c1f}
input,select{background:#18181b;border:1px solid #27272a;border-radius:8px;padding:9px 12px;color:#e4e4e7;font-size:.85rem;width:100%;font-family:inherit}
input:focus,select:focus{outline:none;border-color:#818cf8;box-shadow:0 0 0 2px rgba(129,140,248,.15)}
input:disabled,select:disabled{opacity:.3;cursor:not-allowed}
label{display:block;color:#71717a;font-size:.78rem;margin-bottom:3px;font-weight:500}
.fg{margin-bottom:12px}.fr{display:flex;gap:12px}.fr>*{flex:1}
.btn{display:inline-flex;align-items:center;gap:6px;padding:10px 22px;border-radius:8px;font-weight:600;font-size:.875rem;border:none;cursor:pointer;transition:all .15s;font-family:inherit}
.btn-p{background:#6366f1;color:#fff}.btn-p:hover{background:#818cf8}
.btn-s{background:#27272a;color:#e4e4e7;border:1px solid #3f3f46}.btn-s:hover{border-color:#818cf8}
.bg{display:inline-block;padding:3px 10px;border-radius:99px;font-size:.75rem;font-weight:500}
.bg-g{background:#052e16;color:#4ade80}.bg-r{background:#450a0a;color:#f87171}.bg-b{background:#172554;color:#60a5fa}.bg-y{background:#422006;color:#fbbf24}
.tg{color:#4ade80}.tr{color:#f87171}.ty{color:#fbbf24}.tm{color:#71717a}.tp{color:#818cf8}
.mt{margin-top:14px}
code{background:#27272a;padding:2px 6px;border-radius:4px;font-size:.82rem;color:#a78bfa}
.htmx-indicator{display:none}.htmx-request .htmx-indicator,.htmx-request.htmx-indicator{display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}
.sp{display:inline-block;width:14px;height:14px;border:2px solid #27272a;border-top-color:#818cf8;border-radius:50%;animation:spin .5s linear infinite;vertical-align:middle;margin-right:6px}
.fi{animation:fi .3s ease}@keyframes fi{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.chk{display:flex;flex-wrap:wrap;gap:6px}
.chk label{display:flex;align-items:center;gap:6px;padding:5px 12px;border:1px solid #27272a;border-radius:8px;cursor:pointer;font-size:.82rem;color:#a1a1aa;transition:all .15s;font-weight:500}
.chk label:hover{border-color:#3f3f46;color:#e4e4e7}
.chk label:has(input:checked){border-color:#818cf8;background:rgba(99,102,241,.08);color:#818cf8}
.chk input{display:none}
.hp{background:#111113;border:1px solid #1f1f2e;border-radius:8px;padding:14px;margin-top:10px}
.hp h4{font-size:.75rem;color:#71717a;margin-bottom:8px;text-transform:uppercase;letter-spacing:.4px}
.sm{font-size:.78rem}
""")

# ── State & Plot helpers ─────────────────────────────────────────────────────

class _S:
    dataset=None; dataset_info=None; features=None; labels=None
    raw_samples=None; raw_labels=None; config=None; results={}
S=_S()

PBG="#09090b"; PFC="#18181b"; PTX="#e4e4e7"; PMU="#71717a"
COLS=["#818cf8","#38bdf8","#a78bfa","#4ade80","#fbbf24","#f87171","#e879f9","#fb923c"]

def _b64(fig):
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=140,bbox_inches="tight",facecolor=PBG,edgecolor="none")
    plt.close(fig); buf.seek(0); return "data:image/png;base64,"+base64.b64encode(buf.read()).decode()

def _ax(ax):
    ax.set_facecolor(PFC); ax.tick_params(colors=PMU,labelsize=7); ax.grid(True,alpha=.08,color="#3f3f46")
    for s in ax.spines.values(): s.set_color("#27272a")

def _plot_waves():
    if S.raw_samples is None: return None
    fig,axes=plt.subplots(1,2,figsize=(9,3)); fig.patch.set_facecolor(PBG)
    mods={i:n for i,n in enumerate(sorted(S.dataset_info["modulations"]))}
    for idx,lab in enumerate(np.unique(S.raw_labels)[:2]):
        ax=axes[idx]; _ax(ax); si=np.where(S.raw_labels==lab)[0][0]; sa=S.raw_samples[si]
        ax.plot(sa[0],color="#818cf8",lw=1.2,alpha=.9,label="I")
        ax.plot(sa[1],color="#38bdf8",lw=1.2,alpha=.7,ls="--",label="Q")
        ax.set_title(mods.get(lab,str(lab)),color=PTX,fontsize=10,fontweight="600")
        ax.legend(fontsize=7,facecolor=PFC,edgecolor="#27272a",labelcolor=PTX)
    plt.tight_layout(); return _b64(fig)

def _plot_feat():
    if S.features is None: return None
    nm=["Amp μ","Amp σ²","Amp Skew","Amp Kurt","Freq μ","Freq σ²","Freq Skew","Freq Kurt"]
    n=min(S.features.shape[1],8); fig,axes=plt.subplots(2,4,figsize=(11,4.5)); fig.patch.set_facecolor(PBG)
    for i in range(n):
        ax=axes[i//4][i%4]; _ax(ax)
        for lab in np.unique(S.labels):
            d=S.features[S.labels==lab,i]
            if len(d)>400: d=np.random.choice(d,400,replace=False)
            ax.hist(d,bins=25,alpha=.5,density=True,color=COLS[0] if lab==0 else COLS[1],label="AM" if lab==0 else "FM")
        ax.set_title(nm[i] if i<len(nm) else f"F{i}",color=PTX,fontsize=8)
        ax.legend(fontsize=6,facecolor=PFC,edgecolor="#27272a",labelcolor=PTX)
    for i in range(n,8): axes[i//4][i%4].set_visible(False)
    plt.tight_layout(); return _b64(fig)

def _plot_comp():
    if not S.results: return None
    codes=list(S.results.keys()); n=len(codes)
    fig,axes=plt.subplots(1,4,figsize=(14,max(2.8,n*.45+1.2))); fig.patch.set_facecolor(PBG)
    for j,(ax,title,key) in enumerate([
        (axes[0],"Accuracy %","test_accuracy"),
        (axes[1],"F1 Score","f1_score"),
        (axes[2],"Train Time (s)","training_time"),
        (axes[3],"Inference (ms)","inference_time_ms_per_sample"),
    ]):
        _ax(ax); y=np.arange(n)
        vals=[S.results[c][key]*(100 if key=="test_accuracy" else 1) for c in codes]
        bars=ax.barh(y,vals,color=[COLS[i%len(COLS)] for i in range(n)],alpha=.85,height=.5)
        ax.set_yticks(y); ax.set_yticklabels([MODELS[c] for c in codes],fontsize=7)
        ax.set_title(title,color=PTX,fontsize=9,fontweight="600")
        for b,v in zip(bars,vals):
            fmt=f"{v:.1f}%" if key=="test_accuracy" else f"{v:.3f}"
            ax.text(b.get_width()+max(vals+[.01])*.02,b.get_y()+b.get_height()/2,fmt,va="center",color=PTX,fontsize=7)
    plt.tight_layout(); return _b64(fig)

def _plot_cm():
    best=max(S.results.items(),key=lambda x:x[1]["test_accuracy"],default=(None,None)) if S.results else (None,None)
    if not best[1]: return None
    fig,ax=plt.subplots(figsize=(4,3.3)); fig.patch.set_facecolor(PBG); ax.set_facecolor(PFC)
    sns.heatmap(best[1]["confusion_matrix"],annot=True,fmt="d",cmap="Blues",xticklabels=["AM","FM"],yticklabels=["AM","FM"],ax=ax,linewidths=.5,linecolor="#27272a",annot_kws={"size":13})
    ax.set_xlabel("Predicted",color=PMU,fontsize=8); ax.set_ylabel("Actual",color=PMU,fontsize=8)
    ax.set_title(f"{best[1].get('model_name',MODELS[best[0]])}",color=PTX,fontsize=10,fontweight="600"); ax.tick_params(colors=PTX)
    plt.tight_layout(); return _b64(fig)

def _plot_feature_importance():
    """Plot feature importance from tree-based models (RF, GB, DT)."""
    # Find first model with feature_importance
    fi = None; fi_name = None
    for code in ('rf', 'gb', 'dt'):
        if code in S.results and S.results[code].get('feature_importance'):
            fi = S.results[code]['feature_importance']
            fi_name = MODELS[code]
            break
    if fi is None: return None
    n = len(fi)
    names = [f"F{i}" for i in range(n)]
    # Sort by importance
    idx = np.argsort(fi)[::-1][:min(n, 16)]  # top 16
    fig, ax = plt.subplots(figsize=(8, max(2.5, len(idx) * 0.3)))
    fig.patch.set_facecolor(PBG); _ax(ax)
    y = range(len(idx))
    vals = [fi[i] for i in idx]
    ax.barh(list(y), vals, color=COLS[:len(idx)], alpha=0.8, height=0.5)
    ax.set_yticks(list(y)); ax.set_yticklabels([names[i] for i in idx], fontsize=7)
    ax.set_title(f"{fi_name} — Feature Importance", color=PTX, fontsize=10, fontweight="600")
    for b, v in zip(ax.patches, vals):
        ax.text(b.get_width() + max(vals) * 0.02, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", color=PTX, fontsize=7)
    ax.invert_yaxis()
    plt.tight_layout(); return _b64(fig)


def _load_cfg():
    try: S.config=load_config()
    except Exception:
        import random,string
        rid="".join(random.choices(string.ascii_lowercase+string.digits,k=6))
        S.config={"client_id":f"client_{rid}","server_url":"http://localhost:8000","dataset_path":"./data/RML2016.10a_dict.pkl","local_model_path":"./client/local/local_model.pth","training":{"epochs":10,"batch_size":32,"learning_rate":0.001},"partition_id":0}

# ── App ──────────────────────────────────────────────────────────────────────

def create_client_app(port=7861):
    _load_cfg()
    app=FastHTML(debug=False,hdrs=[CSS])
    TAB_JS=Script("document.addEventListener('click',e=>{if(e.target.classList.contains('tb')){document.querySelectorAll('.tb').forEach(t=>t.classList.remove('on'));e.target.classList.add('on')}});")

    @app.get("/")
    def index():
        return Title("RadioFed Client"),Div(Div(
            Div(H1("RadioFed Client"),P("Byzantine-Resilient Federated Learning for Automatic Modulation Classification"),cls="hdr"),
            Div(A("Config",cls="tb on",hx_get="/p/config",hx_target="#pn",hx_swap="outerHTML"),
                A("Dataset",cls="tb",hx_get="/p/dataset",hx_target="#pn",hx_swap="outerHTML"),
                A("Features",cls="tb",hx_get="/p/features",hx_target="#pn",hx_swap="outerHTML"),
                A("Training",cls="tb",hx_get="/p/training",hx_target="#pn",hx_swap="outerHTML"),
                A("Sync",cls="tb",hx_get="/p/sync",hx_target="#pn",hx_swap="outerHTML"),cls="tabs"),
            TAB_JS,_p_config(),cls="wrap"))

    # ═══ Config ═══
    def _p_config():
        return Div(Div(H3("Configuration"),
            Div(Div(Label("Client ID"),Input(name="client_id",value=S.config.get("client_id",""),id="ci"),cls="fg"),
                Div(Label("Server URL"),Input(name="server_url",value=S.config.get("server_url",""),id="su"),cls="fg"),cls="fr"),
            Button("Save & Connect",cls="btn btn-p",hx_post="/a/cfg",hx_target="#cfgr",hx_include="[id=ci],[id=su]"),
            Div(id="cfgr",cls="mt"),cls="cd"),id="pn",cls="fi")
    @app.get("/p/config")
    def pc(): return _p_config()
    @app.post("/a/cfg")
    def ac(client_id:str="",server_url:str=""):
        S.config["client_id"]=client_id.strip() or S.config["client_id"]
        S.config["server_url"]=server_url.strip() or S.config["server_url"]
        try: save_config(S.config)
        except Exception: pass
        try:
            import requests; requests.post(f'{S.config["server_url"].rstrip("/")}/register/{S.config["client_id"]}',timeout=3)
            st=check_server_status(S.config["server_url"],timeout=3)
            return Span(f"Connected — {st.get('total_clients',0)} clients",cls="bg bg-g",style="font-size:.85rem;padding:6px 14px")
        except Exception: return Span("Cannot reach server",cls="bg bg-r",style="font-size:.85rem;padding:6px 14px")

    # ═══ Dataset ═══
    @app.get("/p/dataset")
    def pd():
        # Scan available partitions
        from data.datasets import list_partitions, DATASETS
        parts = list_partitions()
        # Build source options
        options = []
        for key, info in parts.items():
            for f in info["files"]:
                label_prefix = DATASETS.get(key, {}).get("name", key) if key != "legacy" else "RML 2016.10a"
                cid = f.replace("client_", "").replace(".pkl", "")
                options.append(Option(f"{label_prefix} — Client {cid}", value=f"{key}/{f}"))
        if not options:
            # Fallback to legacy numbering
            for i in range(4):
                p = f"./data/partitions/client_{i}.pkl"
                if os.path.exists(p):
                    options.append(Option(f"Partition {i}", value=f"legacy/client_{i}.pkl"))

        return Div(Div(H3("Load Dataset Partition"),
            P("Select a partition created via the Data Manager (Dashboard → Data Manager tab).",
              cls="tm sm", style="margin-bottom:10px"),
            Div(Label("Partition Source"),
                Select(*options, name="src", id="ds-src") if options else
                P("No partitions found. Use Data Manager to create them.", cls="ty"),
                cls="fg", style="max-width:500px"),
            Button("Load", cls="btn btn-p", hx_post="/a/load", hx_target="#dsr",
                   hx_include="#ds-src", hx_indicator="#dsi") if options else "",
            Span(Span(cls="sp"), "Loading...", id="dsi",
                 cls="htmx-indicator tm", style="margin-left:10px"),
            Div(id="dsr", cls="mt"), cls="cd"), id="pn", cls="fi")

    @app.post("/a/load")
    def al(src: str = "", pid: int = 0):
        # Resolve path from src (format: "dataset_key/filename.pkl") or legacy pid
        if src:
            parts = src.split("/", 1)
            if len(parts) == 2:
                from data.datasets import PARTITIONS_DIR
                if parts[0] == "legacy":
                    path = os.path.join(PARTITIONS_DIR, parts[1])
                else:
                    path = os.path.join(PARTITIONS_DIR, parts[0], parts[1])
            else:
                path = f"./data/partitions/client_{pid}.pkl"
        else:
            path = f"./data/partitions/client_{pid}.pkl"

        if not os.path.exists(path):
            return P(f"Not found: {path}", cls="tr")
        try:
            S.dataset=load_radioml_dataset(path); S.dataset_info=get_dataset_info(S.dataset)
            S.raw_samples,S.raw_labels=flatten_dataset(S.dataset)
            S.config["partition_id"]=pid
            try: save_config(S.config)
            except Exception: pass
            i=S.dataset_info; wf=_plot_waves()
            return Div(Table(Thead(Tr(Th("Property"),Th("Value"))),Tbody(
                Tr(Td("Modulations"),Td(", ".join(i["modulations"]))),
                Tr(Td("SNR Range"),Td(f'{min(i["snrs"])} → {max(i["snrs"])} dB')),
                Tr(Td("Samples"),Td(f'{i["sample_count"]:,}')),
                *[Tr(Td(m),Td(f"{c:,}")) for m,c in i["samples_per_mod"].items()])),
                Div(H3("I/Q Waveforms"),Img(src=wf),cls="cd mt") if wf else "")
        except Exception as e: return P(f"Error: {e}",cls="tr")

    # ═══ Features ═══
    @app.get("/p/features")
    def pf():
        return Div(Div(H3("Feature Extraction"),
            Div(
                Div(Label("Extraction Mode"),
                    Select(Option("16D Traditional — I/Q stats, FFT, ZCR, energy", value="16d"),
                           Option("8D Analog — amplitude & frequency statistics", value="8d"),
                           Option("24D Extended — 16D + HOC + envelope + phase", value="24d"),
                           name="fmode", id="fmode"),
                    cls="fg"),
                style="max-width:480px"
            ),
            P("8D: instantaneous amp/freq moments. 16D: + FFT spectral + temporal. "
              "24D: + cumulants C20/C21/C40/C42 + crest factor + phase entropy.",
              cls="tm sm", style="margin-bottom:10px"),
            Button("Extract Features", cls="btn btn-p",
                   hx_post="/a/feat", hx_target="#ftr", hx_indicator="#fti",
                   hx_include="#fmode"),
            Span(Span(cls="sp"), "Extracting...", id="fti",
                 cls="htmx-indicator tm", style="margin-left:10px"),
            Div(id="ftr", cls="mt"), cls="cd"), id="pn", cls="fi")

    @app.post("/a/feat")
    def af(fmode: str = "16d"):
        if S.dataset is None: return P("Load dataset first.", cls="tr")
        try:
            sa, la = flatten_dataset(S.dataset)
            dim_map = {"8d": 8, "16d": 16, "24d": 24}
            dim = dim_map.get(fmode, 16)
            fl = []
            for i in range(sa.shape[0]):
                try:
                    fl.append(extract_features(sa[i], mode=fmode))
                except Exception:
                    fl.append(np.zeros(dim, dtype=np.float32))
            fe = np.array(fl, dtype=np.float32)
            nm, _, _ = normalize_features(fe)
            S.features = nm; S.labels = la
            return Div(
                Span(f"{fe.shape[1]}D features x {fe.shape[0]:,} samples",
                     cls="bg bg-g", style="font-size:.85rem;padding:6px 14px"),
                Div(H3("Distributions"), Img(src=_plot_feat()), cls="cd mt"))
        except Exception as e:
            return P(f"Error: {e}", cls="tr")

    # ═══ Training ═══
    @app.get("/p/training")
    def pt():
        return Div(Div(
            H3("Model Training"),
            P("Select models and tune hyperparameters.",cls="tm sm",style="margin-bottom:12px"),
            # Model checkboxes
            Div(*[Label(Input(type="checkbox",name="models",value=c,checked=(c in("knn","dt","rf"))),Span(n))
                  for c,n in MODELS.items()],cls="chk"),
            # Hyperparameters panel
            Div(
                H4("Hyperparameters"),
                Div(
                    Div(Label("KNN k"),Input(name="knn_k",type="number",value="5",min="1",max="25",id="hp-knn-k",style="max-width:80px"),cls="fg"),
                    Div(Label("KNN weights"),Select(Option("uniform"),Option("distance"),name="knn_w",id="hp-knn-w"),cls="fg"),
                    Div(Label("DT max depth"),Input(name="dt_depth",type="number",value="",placeholder="∞",min="1",max="50",id="hp-dt-d",style="max-width:80px"),cls="fg"),
                    Div(Label("RF n_estimators"),Input(name="rf_n",type="number",value="100",min="10",max="500",id="hp-rf-n",style="max-width:80px"),cls="fg"),
                    Div(Label("GB learning rate"),Input(name="gb_lr",type="number",value="0.1",min="0.01",max="1",step="0.01",id="hp-gb-lr",style="max-width:80px"),cls="fg"),
                    Div(Label("GB n_estimators"),Input(name="gb_n",type="number",value="100",min="10",max="500",id="hp-gb-n",style="max-width:80px"),cls="fg"),
                    Div(Label("SVM kernel"),Select(Option("rbf"),Option("linear"),Option("poly"),name="svm_k",id="hp-svm-k"),cls="fg"),
                    Div(Label("SVM C"),Input(name="svm_c",type="number",value="1.0",min="0.01",max="100",step="0.1",id="hp-svm-c",style="max-width:80px"),cls="fg"),
                    Div(Label("LR C"),Input(name="lr_c",type="number",value="1.0",min="0.01",max="100",step="0.1",id="hp-lr-c",style="max-width:80px"),cls="fg"),
                    Div(Label("MLP layers"),Input(name="mlp_h",value="64,32",id="hp-mlp-h",style="max-width:100px",placeholder="64,32"),cls="fg"),
                    cls="g4"),
                # JS: grey out irrelevant params
                Script("""
                function upHP(){
                    const sel = new Set([...document.querySelectorAll('[name=models]:checked')].map(e=>e.value));
                    document.querySelectorAll('[id^=hp-]').forEach(el=>{
                        const pre=el.id.split('-')[1]; // knn, dt, rf, etc.
                        el.disabled=!sel.has(pre);
                        el.style.opacity=sel.has(pre)?'1':'.25';
                    });
                }
                document.addEventListener('change',e=>{if(e.target.name==='models')upHP()});
                setTimeout(upHP,50);
                """),
                cls="hp"),
            # Advanced options
            Div(
                H4("Advanced"),
                Div(
                    Div(Label("Differential Privacy (epsilon)"),
                        Input(name="dp_eps",type="number",value="",placeholder="disabled",
                              min="0.01",max="100",step="0.1",id="dp-eps",style="max-width:100px"),
                        P("Lower = more private. Leave empty to disable.",cls="tm",style="font-size:.7rem"),
                        cls="fg"),
                    Div(Label(Input(type="checkbox",name="run_cv",value="1",id="run-cv",style="width:auto;margin-right:6px"),
                              "Run 5-fold cross-validation"),cls="fg",style="padding-top:20px"),
                    cls="fr"),
                cls="hp mt"),
            Button("Train Selected",cls="btn btn-p mt",hx_post="/a/train",hx_target="#trr",hx_indicator="#tri",
                   hx_include="[name=models],[name^=knn_],[name^=dt_],[name^=rf_],[name^=gb_],[name^=svm_],[name^=lr_],[name^=mlp_],[name=dp_eps],[name=run_cv]"),
            Span(Span(cls="sp"),"Training...",id="tri",cls="htmx-indicator tm",style="margin-left:10px"),
            Div(id="trr",cls="mt"),cls="cd"),id="pn",cls="fi")

    @app.post("/a/train")
    def at(models:list[str]=None,knn_k:int=5,knn_w:str="uniform",dt_depth:str="",
           rf_n:int=100,gb_lr:float=0.1,gb_n:int=100,svm_k:str="rbf",svm_c:float=1.0,
           lr_c:float=1.0,mlp_h:str="64,32",dp_eps:str="",run_cv:str=""):
        if S.features is None: return P("Extract features first.",cls="tr")
        if not models: return P("Select at least one model.",cls="ty")
        if isinstance(models,str): models=[models]
        try:
            # Apply DP if requested
            features_to_use = S.features
            dp_info = None
            if dp_eps and dp_eps.strip():
                from client.train import apply_differential_privacy
                features_to_use, dp_info = apply_differential_privacy(S.features, epsilon=float(dp_eps))

            S.results={}
            dt_d=int(dt_depth) if dt_depth and dt_depth.strip() else None
            mlp_layers=tuple(int(x.strip()) for x in mlp_h.split(",") if x.strip().isdigit()) or (64,32)

            trainers={
                "knn": lambda: train_knn_model(features_to_use,S.labels,n_neighbors=knn_k,weights=knn_w),
                "dt":  lambda: train_dt_model(features_to_use,S.labels,max_depth=dt_d),
                "rf":  lambda: train_rf_model(features_to_use,S.labels,n_estimators=rf_n),
                "gb":  lambda: train_gb_model(features_to_use,S.labels,n_estimators=gb_n,learning_rate=gb_lr),
                "svm": lambda: train_svm_model(features_to_use,S.labels,kernel=svm_k,C=svm_c),
                "lr":  lambda: train_lr_model(features_to_use,S.labels,C=lr_c),
                "nb":  lambda: train_nb_model(features_to_use,S.labels),
                "mlp": lambda: train_mlp_model(features_to_use,S.labels,hidden_layers=mlp_layers),
            }
            for c in models:
                if c in trainers:
                    S.results[c]=trainers[c]()
                    _save_art(S.results[c],c)

            # Results table — full metrics
            hd=["Metric"]+[MODELS[c] for c in S.results]
            def _r(lb,fn): return Tr(Td(lb,style="font-weight:500"),*[Td(fn(S.results[c])) for c in S.results])
            rows=[
                _r("Test Accuracy",  lambda r:Span(f'{r["test_accuracy"]*100:.2f}%',cls="tg" if r["test_accuracy"]>.7 else "")),
                _r("Train Accuracy", lambda r:f'{r["train_accuracy"]*100:.2f}%'),
                _r("Precision",      lambda r:f'{r["precision"]:.4f}'),
                _r("Recall",         lambda r:f'{r["recall"]:.4f}'),
                _r("F1 Score",       lambda r:f'{r["f1_score"]:.4f}'),
                _r("Cohen's κ",      lambda r:f'{r["cohen_kappa"]:.4f}'),
                _r("Train Time",     lambda r:f'{r["training_time"]:.3f}s'),
                _r("Inference",      lambda r:f'{r["inference_time_ms_per_sample"]:.3f}ms'),
                _r("Train Samples",  lambda r:f'{r["n_samples"]:,}'),
                _r("Test Samples",   lambda r:f'{r["n_test_samples"]:,}'),
                _r("Features",       lambda r:str(r["n_features"])),
            ]
            best_c=max(S.results,key=lambda c:S.results[c]["test_accuracy"])
            best_f1=max(S.results,key=lambda c:S.results[c]["f1_score"])

            parts=[
                Div(
                    Span(f"Best Accuracy: {MODELS[best_c]} — {S.results[best_c]['test_accuracy']*100:.1f}%",cls="bg bg-g",style="font-size:.85rem;padding:6px 14px"),
                    Span(f"Best F1: {MODELS[best_f1]} — {S.results[best_f1]['f1_score']:.4f}",cls="bg bg-b",style="font-size:.85rem;padding:6px 14px;margin-left:8px"),
                    cls="mb"),
                Div(Table(Thead(Tr(*[Th(h) for h in hd])),Tbody(*rows)),style="overflow-x:auto"),
            ]
            # Auto-upload KNN
            if "knn" in S.results:
                try:
                    um=_upload("knn")
                    if um: parts.append(P(um,cls="bg bg-b mt",style="font-size:.82rem;padding:5px 12px"))
                except Exception: pass

            # DP info badge
            if dp_info:
                parts.append(P(f"DP: epsilon={dp_info['epsilon']}, sigma={dp_info['sigma']:.4f}, SNR={dp_info['snr_db']:.1f}dB",
                               cls="bg bg-y mt",style="font-size:.82rem;padding:5px 12px"))

            cp=_plot_comp(); cm=_plot_cm()
            if cp: parts.append(Div(H3("Model Comparison"),Img(src=cp),cls="cd mt"))
            if cm: parts.append(Div(H3("Best Model — Confusion Matrix"),Img(src=cm),cls="cd mt"))

            # Feature importance (for tree-based models)
            fi_plot = _plot_feature_importance()
            if fi_plot: parts.append(Div(H3("Feature Importance"),Img(src=fi_plot),cls="cd mt"))

            # Cross-validation
            if run_cv == "1" and S.results:
                from client.train import cross_validate, MODEL_FACTORIES
                cv_rows = []
                for code in S.results:
                    factory = MODEL_FACTORIES.get(code)
                    if factory:
                        cv = cross_validate(factory, features_to_use, S.labels, n_folds=5)
                        cv_rows.append(Tr(Td(MODELS[code]),
                            Td(f"{cv['accuracy_mean']:.4f} +/- {cv['accuracy_std']:.4f}"),
                            Td(f"{cv['f1_mean']:.4f} +/- {cv['f1_std']:.4f}")))
                if cv_rows:
                    parts.append(Div(H3("5-Fold Cross-Validation"),
                        Table(Thead(Tr(Th("Model"),Th("Accuracy"),Th("F1"))),Tbody(*cv_rows)),cls="cd mt"))

            return Div(*parts)
        except Exception as e:
            import traceback; traceback.print_exc()
            return P(f"Error: {e}",cls="tr")

    def _save_art(r,code):
        mp=S.config.get("local_model_path","./client/local/local_model.pth").replace(".pth",f"_{code}.pkl")
        _save_model(r["model"],mp)
        fp=mp.replace(f"_{code}.pkl",f"_{code}_features.pkl"); lp=mp.replace(f"_{code}.pkl",f"_{code}_labels.pkl")
        os.makedirs(os.path.dirname(fp),exist_ok=True)
        with open(fp,"wb") as f: pickle.dump(S.features,f)
        with open(lp,"wb") as f: pickle.dump(S.labels,f)
        r["model_path"]=mp; r["features_path"]=fp; r["labels_path"]=lp
        save_metrics({"timestamp":datetime.now().isoformat(),"model_type":code,
            "train_accuracy":float(r["train_accuracy"]),"test_accuracy":float(r["test_accuracy"]),
            "precision":float(r["precision"]),"recall":float(r["recall"]),"f1_score":float(r["f1_score"]),
            "cohen_kappa":float(r["cohen_kappa"]),
            "training_time":float(r["training_time"]),"inference_time_ms_per_sample":float(r["inference_time_ms_per_sample"]),
            "n_samples":int(r["n_samples"])})

    def _upload(code="knn"):
        r=S.results.get(code)
        if not r or not r.get("model_path"): return ""
        import requests as rq
        url=f'{S.config["server_url"].rstrip("/")}/upload_model/{S.config["client_id"]}'
        files={"model_file":open(r["model_path"],"rb"),"features_file":open(r["features_path"],"rb"),"labels_file":open(r["labels_path"],"rb")}
        resp=rq.post(url,params={"n_samples":r["n_samples"]},files=files,timeout=30)
        for f in files.values(): f.close()
        if resp.status_code==200:
            d=resp.json().get("upload_status",{})
            return f'Uploaded ({d.get("pending_uploads",0)}/{d.get("threshold",2)})'
        return ""

    # ═══ Sync ═══
    @app.get("/p/sync")
    def ps():
        return Div(Div(
            Div(Div(H3("Upload Model"),P("Send to server for aggregation.",cls="tm sm"),
                    Button("Upload",cls="btn btn-p",hx_post="/a/up",hx_target="#sr"),cls="cd"),
                Div(H3("Download Global"),P("Get aggregated model from server.",cls="tm sm"),
                    Button("Download",cls="btn btn-s",hx_post="/a/dl",hx_target="#sr"),cls="cd"),cls="g2"),
            Div(id="sr",cls="mt")),id="pn",cls="fi")
    @app.post("/a/up")
    def aup():
        try:
            m=_upload("knn")
            if not m: return P("Train KNN first.",cls="tr")
            return Span(m,cls="bg bg-g",style="font-size:.85rem;padding:6px 14px")
        except Exception as e: return P(f"Error: {e}",cls="tr")
    @app.post("/a/dl")
    def adl():
        try:
            ok=download_global_model(S.config["server_url"],"./client/local/global_knn_model.pkl",timeout=30)
            return Span("Downloaded",cls="bg bg-g",style="font-size:.85rem;padding:6px 14px") if ok else P("Failed",cls="tr")
        except Exception as e: return P(f"Error: {e}",cls="tr")

    return app
