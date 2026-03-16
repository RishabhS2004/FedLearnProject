# RadioFed

**Byzantine-Resilient Federated Learning for Automatic Modulation Classification**

A production-grade federated learning system for classifying radio signal modulations using RadioML datasets. Features 8 ML classifiers, 3 feature extraction modes (8D/16D/24D), Byzantine fault tolerance, Manim animations, and 4 independent web services.

---

## Quick Start

```bash
# 1. Setup
git clone <repo-url> && cd RadioFed
uv venv .venv && uv pip install -e .

# 2. Download & partition dataset
python data/manager.py              # opens http://localhost:7862
# → Click "Download via Kaggle" on RadioML 2016.10a
# → Set clients=4, click "Partition"

# 3. Start server + dashboard
python central/main.py
# → API: http://localhost:8000  |  Dashboard: http://localhost:7860

# 4. Start client (separate terminal)
python client/main.py --port 7861
# → Client UI: http://localhost:7861
```

---

## Services

| Service | Port | Command | Purpose |
|---------|------|---------|---------|
| **Data Manager** | 7862 | `python data/manager.py` | Download datasets, create partitions |
| **Central Server** | 8000 + 7860 | `python central/main.py` | REST API + monitoring dashboard |
| **Client** | 7861 | `python client/main.py` | Train models, extract features, sync |

Run order: Data Manager first (one-time setup), then Central Server, then one or more Clients.

---

## Dataset Setup

### Supported Datasets

| Dataset | Modulations | SNR Range | Format | Samples/key |
|---------|------------|-----------|--------|-------------|
| **RadioML 2016.10a** | 11 (8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM) | -20 to +18 dB | pickle | 1,000 |
| **RadioML 2018.01A** | 24 (OOK, ASK4, ASK8, BPSK, QPSK, PSK8, PSK16, PSK32, APSK16, APSK32, APSK64, APSK128, QAM16, QAM32, QAM64, QAM128, QAM256, AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM, GMSK, OQPSK) | -20 to +30 dB | HDF5 | 4,096 |

### Download & Partition (via Data Manager UI)

1. `python data/manager.py` — opens http://localhost:7862
2. Click **Download via Kaggle** on the dataset you want
3. Choose number of clients and filter mode (all / analog only)
4. Click **Partition** — creates `data/partitions/{dataset}/client_N.pkl`

### Download & Partition (CLI)

```bash
# Kaggle CLI setup (one-time)
pip install kaggle
# Place API key at ~/.kaggle/kaggle.json

# Download
kaggle datasets download -d nolasthitnotomorrow/radioml2016-deepsigcom -p data/ --unzip

# Partition
python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 4
```

### Download & Partition (Python)

```python
from data.datasets import download_dataset, partition_dataset

download_dataset("rml2016.10a")
partition_dataset("rml2016.10a", num_clients=4, filter_mode="all")
partition_dataset("rml2016.10a", num_clients=6, filter_mode="analog")
```

---

## Architecture

```
                    ┌──────────────────────┐
                    │  Data Manager (:7862) │
                    │  Download & partition │
                    └──────────┬───────────┘
                               │ creates data/partitions/
        ┌──────────┬───────────┼───────────┬──────────┐
        ▼          ▼           ▼           ▼          ▼
   ┌─────────┐┌─────────┐┌─────────┐┌─────────┐
   │Client 1 ││Client 2 ││Client 3 ││Client 4 │
   │ :7861   ││ :7863   ││ :7865   ││(Byzant.)│
   └────┬────┘└────┬────┘└────┬────┘└────┬────┘
        │          │          │          │
        └──────────┴────┬─────┴──────────┘
                        ▼
   ┌─────────────────────────────────────────────┐
   │         Central Server (:8000)               │
   │  Byzantine: Krum → Trust → Anomaly → Cosine │
   │  Aggregate honest → Global Model             │
   ├─────────────────────────────────────────────┤
   │         Dashboard (:7860)                    │
   │  Overview · Clients · Byzantine · Metrics    │
   │  How It Works (Manim animations)             │
   └─────────────────────────────────────────────┘
```

## Feature Extraction

Three modes (selectable in Client UI → Features tab):

| Mode | Dim | Features |
|------|-----|----------|
| **8D** Analog | 8 | Instantaneous amplitude & frequency statistics |
| **16D** Traditional | 16 | I/Q stats + FFT spectral + zero-crossing rate + energy |
| **24D** Extended | 24 | 16D + higher-order cumulants (C20, C21, C40, C42) + crest factor + max/min ratio + phase std + phase entropy |

## ML Models

8 classifiers with hyperparameter tuning (Client UI → Training tab):

| Model | Type | Key Parameters |
|-------|------|---------------|
| KNN | Instance-based | k, weights (uniform/distance) |
| Decision Tree | Tree-based | max_depth, min_samples_split |
| Random Forest | Ensemble (bagging) | n_estimators, max_depth |
| Gradient Boosting | Ensemble (boosting) | n_estimators, learning_rate, max_depth |
| SVM | Kernel-based | kernel (rbf/linear/poly), C |
| Logistic Regression | Linear | C, max_iter |
| Naive Bayes | Probabilistic | Gaussian prior |
| MLP Neural Network | Neural network | hidden_layers, max_iter |

Metrics: Accuracy, Precision, Recall, F1, Cohen's Kappa, training time, inference time.

## Byzantine Fault Tolerance

Multi-layer defense (configured in `central/config.json`):

1. **Trust filtering** — exclude clients below trust threshold (0.3)
2. **Anomaly detection** — z-score on feature/label distributions
3. **Krum selection** — geometric median filtering
4. **Cosine filtering** — reject updates dissimilar to median
5. **Trust updates** — honest clients gain trust, malicious lose it

## Project Structure

```
RadioFed/
├── central/                 # Server + dashboard
│   ├── main.py             # Launcher (API :8000 + Dashboard :7860)
│   ├── server.py           # FastAPI REST endpoints
│   ├── dashboard_app.py    # FastHTML dashboard
│   ├── aggregator.py       # KNN/DT aggregation + Byzantine
│   ├── byzantine.py        # Trust, Krum, anomaly detection
│   ├── state.py            # Thread-safe state management
│   └── config.json
├── client/                  # Client UI
│   ├── main.py             # Launcher (:7861)
│   ├── app.py              # FastHTML client interface
│   ├── train.py            # 8 ML trainers
│   ├── feature_extract.py  # 8D/16D/24D extraction
│   ├── dataset_loader.py   # RadioML loader
│   ├── sync.py             # Server communication
│   └── state.py            # Config/metrics persistence
├── data/                    # Data management
│   ├── manager.py          # Standalone Data Manager (:7862)
│   ├── datasets.py         # Download, load, partition logic
│   └── partition_dataset.py # CLI partitioning tool
├── animations/              # Manim scenes
│   ├── fl_scenes.py        # 1080p hero animations
│   └── fl_scenes_small.py  # 360p card animations
├── static/videos/           # Rendered manim videos
├── tests/                   # 70+ pytest tests
├── amc-rml2016a-updated.ipynb  # Research notebook
├── pyproject.toml
└── readme.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/register/{client_id}` | POST | Register client + initialize trust |
| `/upload_model/{client_id}` | POST | Upload model + features + labels |
| `/aggregate` | POST | Manual aggregation trigger |
| `/global_model` | GET | Download global model |
| `/trust_scores` | GET | All client trust scores |
| `/trust_history/{client_id}` | GET | Trust history for a client |
| `/byzantine_report` | GET | Latest defense report |
| `/byzantine_strategy` | POST | Change defense strategy |
| `/status` | GET | Server status + registry |
| `/health` | GET | Health check |
| `/aggregation_results` | GET | Latest aggregation metrics |

## Tests

```bash
python -m pytest tests/ -v    # 70 tests
```

Covers: Byzantine defense, feature extraction, dataset loading, integration, dashboard metrics.

## Manim Animations

5 pre-rendered animations displayed in Dashboard → How It Works:

```bash
# Regenerate hero (1080p)
manim -qh animations/fl_scenes.py FederatedLearningFlow --media_dir static/manim_out
cp static/manim_out/videos/fl_scenes/1080p60/*.mp4 static/videos/

# Regenerate cards (360p, crisp at small size)
for s in SignalClassification ByzantineDetection AggregationProcess TrustEvolution; do
  manim -qh animations/fl_scenes_small.py $s --media_dir static/manim_out_sm
done
cp static/manim_out_sm/videos/fl_scenes_small/1080p60/*.mp4 static/videos/
```

## Tech Stack

FastHTML + HTMX | FastAPI + Uvicorn | scikit-learn | Manim | Matplotlib + Seaborn | uv
