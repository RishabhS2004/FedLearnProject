# RadioFed

**Byzantine-Resilient Federated Learning for Automatic Modulation Classification**

A federated learning system for classifying radio signal modulations using RadioML datasets. 8 ML classifiers, 3 feature extraction modes (8D/16D/24D), Byzantine fault tolerance, and 4 independent web services.

## Quick Start

```bash
git clone <repo-url> && cd RadioFed
uv sync

uv run python data/manager.py              # Data Manager  → localhost:7862
uv run python central/main.py              # Server + Dashboard → localhost:8000, localhost:7860
uv run python client/main.py --port 7861   # Client UI → localhost:7861
```

## Services

| Service | Port | Command |
|---------|------|---------|
| Data Manager | 7862 | `python data/manager.py` |
| Central Server | 8000 + 7860 | `python central/main.py` |
| Client | 7861+ | `python client/main.py --port <port>` |

## Dataset

Supports RadioML 2016.10a (11 modulations, pickle) and 2018.01A (24 modulations, HDF5).

Download and partition via Data Manager UI, CLI, or Python:

```bash
python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 4
```

```python
from data.datasets import download_dataset, partition_dataset
download_dataset("rml2016.10a")
partition_dataset("rml2016.10a", num_clients=4, filter_mode="analog")
```

## Architecture

```
                    ┌──────────────────────┐
                    │  Data Manager (:7862) │
                    └──────────┬───────────┘
        ┌──────────┬───────────┼───────────┬──────────┐
        ▼          ▼           ▼           ▼          ▼
   ┌─────────┐┌─────────┐┌─────────┐┌─────────┐
   │Client 1 ││Client 2 ││Client 3 ││Client N │
   └────┬────┘└────┬────┘└────┬────┘└────┬────┘
        └──────────┴────┬─────┴──────────┘
                        ▼
   ┌─────────────────────────────────────────────┐
   │         Central Server (:8000)               │
   │  Byzantine: Krum → Trust → Anomaly → Cosine │
   ├─────────────────────────────────────────────┤
   │         Dashboard (:7860)                    │
   └─────────────────────────────────────────────┘
```

## Feature Extraction

| Mode | Dim | Features |
|------|-----|----------|
| 8D | 8 | Instantaneous amplitude and frequency statistics |
| 16D | 16 | I/Q stats, FFT spectral, zero-crossing rate, energy |
| 24D | 24 | 16D + higher-order cumulants, crest factor, phase entropy |

## ML Models

| Model | Key Parameters |
|-------|---------------|
| KNN | k, weights |
| Decision Tree | max_depth |
| Random Forest | n_estimators |
| Gradient Boosting | n_estimators, learning_rate |
| SVM | kernel, C |
| Logistic Regression | C |
| Naive Bayes | Gaussian prior |
| MLP | hidden_layers |

## Byzantine Fault Tolerance

1. Trust filtering — exclude clients below threshold
2. Anomaly detection — z-score on feature/label distributions
3. Krum selection — geometric median filtering
4. Cosine filtering — reject dissimilar updates
5. Trust updates — reward honest, penalize malicious

## Project Structure

```
RadioFed/
├── central/              # Server + dashboard
│   ├── main.py           # API :8000 + Dashboard :7860
│   ├── server.py         # FastAPI endpoints
│   ├── dashboard_app.py  # FastHTML dashboard
│   ├── aggregator.py     # Model aggregation
│   ├── byzantine.py      # Byzantine defense
│   └── state.py          # State management
├── client/               # Client UI
│   ├── main.py           # Client launcher
│   ├── app.py            # FastHTML interface
│   ├── train.py          # 8 ML trainers
│   ├── feature_extract.py
│   ├── dataset_loader.py
│   └── sync.py
├── data/                 # Data management
│   ├── manager.py        # Data Manager :7862
│   ├── datasets.py       # Download/load/partition
│   └── partition_dataset.py
├── animations/           # Manim scenes
├── static/videos/        # Rendered animations
├── tests/                # 87 tests
└── pyproject.toml
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/register/{client_id}` | POST | Register client |
| `/upload_model/{client_id}` | POST | Upload model |
| `/aggregate` | POST | Trigger aggregation |
| `/global_model` | GET | Download global model |
| `/trust_scores` | GET | Client trust scores |
| `/byzantine_report` | GET | Defense report |
| `/status` | GET | Server status |
| `/health` | GET | Health check |

## Docker

```bash
docker compose up --build
```

Starts all services: server (:8000), dashboard (:7860), data manager (:7862), and 2 clients (:7861, :7863). Clients wait for the server health check before starting.

Run individual services:

```bash
docker build -t radiofed .
docker run -p 8000:8000 -p 7860:7860 radiofed uv run python central/main.py
docker run -p 7862:7862 radiofed uv run python data/manager.py
docker run -p 7861:7861 radiofed uv run python client/main.py --port 7861 --client-id client_0
```

## Tests

```bash
uv run pytest tests/ -v
```

## Tech Stack

FastHTML, HTMX, FastAPI, scikit-learn, Manim, Matplotlib, Docker, uv
