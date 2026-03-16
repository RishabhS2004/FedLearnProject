"""
RadioML Dataset Manager

Handles downloading, loading, and partitioning of multiple RadioML datasets:
- RadioML 2016.10a: 11 modulations, 20 SNR levels, pickle format
- RadioML 2018.01A: 24 modulations, 26 SNR levels, HDF5 format

Modulations in 2018.01A / 2017.01A:
  OOK, ASK4, ASK8, BPSK, QPSK, PSK8, PSK16, PSK32, APSK16, APSK32,
  APSK64, APSK128, QAM16, QAM32, QAM64, QAM128, QAM256, AM_SSB_WC,
  AM_SSB_SC, AM_DSB_WC, AM_DSB_SC, FM, GMSK, OQPSK
"""

import os
import pickle
import json
import logging
import subprocess
import shutil
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__))
PARTITIONS_DIR = os.path.join(DATA_DIR, "partitions")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.json")

# ── Dataset Catalog ──────────────────────────────────────────────────────────

DATASETS = {
    "rml2016.10a": {
        "name": "RadioML 2016.10a",
        "filename": "RML2016.10a_dict.pkl",
        "format": "pickle",
        "modulations": ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
                        "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"],
        "snr_range": [-20, 18],
        "snr_step": 2,
        "samples_per_key": 1000,
        "sample_length": 128,
        "kaggle_slug": "nolasthitnotomorrow/radioml2016-deepsigcom",
        "description": "11 modulations, 20 SNR levels (-20 to +18 dB), pickle format",
    },
    "rml2018.01a": {
        "name": "RadioML 2018.01A",
        "filename": "GOLD_XYZ_OSC.0001_1024.hdf5",
        "format": "hdf5",
        "modulations": ["OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK",
                        "16PSK", "32PSK", "16APSK", "32APSK", "64APSK",
                        "128APSK", "16QAM", "32QAM", "64QAM", "128QAM",
                        "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC",
                        "AM-DSB-SC", "FM", "GMSK", "OQPSK"],
        "snr_range": [-20, 30],
        "snr_step": 2,
        "samples_per_key": 4096,
        "sample_length": 1024,
        "kaggle_slug": "pinxau1000/radioml2018",
        "description": "24 modulations, 26 SNR levels (-20 to +30 dB), HDF5 format",
    },
}

# Analog-only filter maps per dataset
ANALOG_FILTERS = {
    "rml2016.10a": {"AM-DSB": "AM", "AM-SSB": "AM", "WBFM": "FM"},
    "rml2018.01a": {
        "AM-SSB-WC": "AM-SSB", "AM-SSB-SC": "AM-SSB",
        "AM-DSB-WC": "AM-DSB", "AM-DSB-SC": "AM-DSB", "FM": "FM",
    },
}


# ── Status Tracking ──────────────────────────────────────────────────────────

def _load_catalog() -> Dict:
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH, "r") as f:
            return json.load(f)
    return {"downloaded": {}, "partitioned": {}}


def _save_catalog(catalog: Dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2)


def get_dataset_status() -> Dict:
    """Get status of all datasets (downloaded, partitioned, paths)."""
    catalog = _load_catalog()
    status = {}
    for key, meta in DATASETS.items():
        fpath = os.path.join(DATA_DIR, meta["filename"])
        downloaded = os.path.exists(fpath)
        size_mb = round(os.path.getsize(fpath) / 1e6, 1) if downloaded else 0
        # Check partitions
        parts_dir = os.path.join(PARTITIONS_DIR, key)
        parts = []
        if os.path.isdir(parts_dir):
            parts = sorted([f for f in os.listdir(parts_dir) if f.endswith(".pkl")])
        status[key] = {
            **meta,
            "key": key,
            "downloaded": downloaded,
            "file_path": fpath if downloaded else None,
            "size_mb": size_mb,
            "partitions": parts,
            "num_partitions": len(parts),
        }
    return status


# ── Download ─────────────────────────────────────────────────────────────────

def download_dataset(dataset_key: str) -> Tuple[bool, str]:
    """Download a RadioML dataset via kaggle CLI."""
    if dataset_key not in DATASETS:
        return False, f"Unknown dataset: {dataset_key}"

    meta = DATASETS[dataset_key]
    slug = meta["kaggle_slug"]
    dest = DATA_DIR

    # Check if already downloaded
    fpath = os.path.join(dest, meta["filename"])
    if os.path.exists(fpath):
        return True, f"Already downloaded: {fpath}"

    # Try kaggle CLI
    try:
        logger.info(f"Downloading {meta['name']} via kaggle...")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", dest, "--unzip"],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0 and os.path.exists(fpath):
            catalog = _load_catalog()
            catalog["downloaded"][dataset_key] = fpath
            _save_catalog(catalog)
            return True, f"Downloaded: {meta['name']}"
        else:
            err = result.stderr.strip() or result.stdout.strip()
            return False, f"kaggle CLI failed: {err}"
    except FileNotFoundError:
        return False, ("kaggle CLI not found. Install with: pip install kaggle\n"
                       "Then set up API key: https://www.kaggle.com/docs/api")
    except subprocess.TimeoutExpired:
        return False, "Download timed out (10 min limit)"
    except Exception as e:
        return False, f"Download error: {e}"


# ── Loading ──────────────────────────────────────────────────────────────────

def load_dataset(dataset_key: str, filter_mode: str = "all") -> Tuple[Dict, Dict]:
    """
    Load a RadioML dataset.

    Args:
        dataset_key: 'rml2016.10a' or 'rml2018.01a'
        filter_mode: 'all' (all modulations), 'analog' (analog only)

    Returns:
        (data_dict, info_dict)
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    meta = DATASETS[dataset_key]
    fpath = os.path.join(DATA_DIR, meta["filename"])

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Dataset not found: {fpath}. Download it first.")

    if meta["format"] == "pickle":
        return _load_pickle_dataset(fpath, dataset_key, filter_mode)
    elif meta["format"] == "hdf5":
        return _load_hdf5_dataset(fpath, dataset_key, filter_mode)
    else:
        raise ValueError(f"Unsupported format: {meta['format']}")


def _load_pickle_dataset(fpath, dataset_key, filter_mode):
    with open(fpath, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    analog_map = ANALOG_FILTERS.get(dataset_key, {})
    data = {}
    for (mod, snr), samples in raw.items():
        if filter_mode == "analog" and mod not in analog_map:
            continue
        label = analog_map.get(mod, mod) if filter_mode == "analog" else mod
        key = (label, snr)
        if key not in data:
            data[key] = []
        data[key].append(samples)

    # Stack arrays
    for k in data:
        data[k] = np.concatenate(data[k], axis=0)

    mods = sorted(set(k[0] for k in data))
    snrs = sorted(set(k[1] for k in data))
    total = sum(v.shape[0] for v in data.values())

    info = {
        "dataset": dataset_key,
        "modulations": mods,
        "snrs": snrs,
        "total_samples": total,
        "sample_shape": data[list(data.keys())[0]].shape[1:] if data else None,
        "filter_mode": filter_mode,
    }
    return data, info


def _load_hdf5_dataset(fpath, dataset_key, filter_mode):
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 datasets: pip install h5py")

    analog_map = ANALOG_FILTERS.get(dataset_key, {})
    meta = DATASETS[dataset_key]
    mod_list = meta["modulations"]

    with h5py.File(fpath, "r") as f:
        X = f["X"][:]   # (N, 1024, 2) — I/Q
        Y = f["Y"][:]   # (N, 24) one-hot
        Z = f["Z"][:]   # (N,) SNR

    data = {}
    for i in range(len(X)):
        mod_idx = np.argmax(Y[i])
        mod = mod_list[mod_idx]
        snr = int(Z[i])

        if filter_mode == "analog" and mod not in analog_map:
            continue
        label = analog_map.get(mod, mod) if filter_mode == "analog" else mod

        # Transpose to (2, 1024) to match our format
        iq = X[i].T  # from (1024, 2) to (2, 1024)
        key = (label, snr)
        if key not in data:
            data[key] = []
        data[key].append(iq[np.newaxis, :, :])

    for k in data:
        data[k] = np.concatenate(data[k], axis=0)

    mods = sorted(set(k[0] for k in data))
    snrs = sorted(set(k[1] for k in data))
    total = sum(v.shape[0] for v in data.values())

    info = {
        "dataset": dataset_key,
        "modulations": mods,
        "snrs": snrs,
        "total_samples": total,
        "sample_shape": data[list(data.keys())[0]].shape[1:] if data else None,
        "filter_mode": filter_mode,
    }
    return data, info


# ── Partitioning ─────────────────────────────────────────────────────────────

def partition_dataset(
    dataset_key: str,
    num_clients: int = 4,
    filter_mode: str = "all",
    distribution: str = "iid",
    dirichlet_alpha: float = 0.5,
    random_seed: int = 42
) -> Tuple[bool, str]:
    """
    Load dataset, partition into num_clients non-overlapping parts, save to disk.

    Args:
        distribution: 'iid' (uniform random) or 'noniid' (Dirichlet-based skew)
        dirichlet_alpha: Concentration parameter for non-IID. Lower = more skewed.
            alpha=100 ≈ IID, alpha=0.5 = moderate skew, alpha=0.1 = extreme skew

    Saves to: data/partitions/{dataset_key}/client_{i}.pkl
    """
    try:
        data, info = load_dataset(dataset_key, filter_mode)
    except Exception as e:
        return False, f"Failed to load: {e}"

    out_dir = os.path.join(PARTITIONS_DIR, dataset_key)
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(random_seed)

    if distribution == "noniid":
        _partition_noniid(data, num_clients, dirichlet_alpha, out_dir)
    else:
        _partition_iid(data, num_clients, out_dir)

    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "dataset": dataset_key, "num_clients": num_clients,
            "filter_mode": filter_mode, "distribution": distribution,
            "dirichlet_alpha": dirichlet_alpha if distribution == "noniid" else None,
            "modulations": info["modulations"], "snrs": info["snrs"],
            "total_samples": info["total_samples"], "seed": random_seed,
        }, f, indent=2)

    catalog = _load_catalog()
    catalog["partitioned"][dataset_key] = {
        "num_clients": num_clients, "path": out_dir,
        "filter_mode": filter_mode, "distribution": distribution,
    }
    _save_catalog(catalog)
    return True, f"Created {num_clients} {distribution} partitions in {out_dir}"


def _partition_iid(data, num_clients, out_dir):
    """Uniform random (IID) partitioning."""
    for i in range(num_clients):
        part = {}
        for key, samples in data.items():
            n = samples.shape[0]
            indices = np.random.permutation(n)
            chunk = n // num_clients
            start = i * chunk
            end = n if i == num_clients - 1 else start + chunk
            part[key] = samples[indices[start:end]]
        with open(os.path.join(out_dir, f"client_{i}.pkl"), "wb") as f:
            pickle.dump(part, f)


def _partition_noniid(data, num_clients, alpha, out_dir):
    """
    Dirichlet-based non-IID partitioning.
    Each client gets a skewed distribution of modulation classes.
    alpha controls skew: lower = more heterogeneous.
    """
    # Group all samples by modulation class
    mods = sorted(set(k[0] for k in data))
    snrs = sorted(set(k[1] for k in data))

    # For each modulation, collect all samples across SNRs
    mod_samples = {m: [] for m in mods}
    mod_keys = {m: [] for m in mods}
    for (mod, snr), samples in data.items():
        mod_samples[mod].append(samples)
        mod_keys[mod].append((mod, snr))

    # Generate Dirichlet proportions per class
    # Shape: (num_classes, num_clients) — how much of each class goes to each client
    proportions = np.random.dirichlet([alpha] * num_clients, size=len(mods))

    # Initialize empty partitions
    partitions = [{} for _ in range(num_clients)]

    for class_idx, mod in enumerate(mods):
        props = proportions[class_idx]  # proportion for each client
        for (m, snr), samples in data.items():
            if m != mod:
                continue
            n = samples.shape[0]
            indices = np.random.permutation(n)
            # Split according to Dirichlet proportions
            splits = (props * n).astype(int)
            # Fix rounding: give remainder to last client
            splits[-1] = n - splits[:-1].sum()
            splits = np.maximum(splits, 0)

            offset = 0
            for i in range(num_clients):
                count = splits[i]
                if count > 0:
                    key = (m, snr)
                    chunk = samples[indices[offset:offset + count]]
                    if key in partitions[i]:
                        partitions[i][key] = np.concatenate([partitions[i][key], chunk])
                    else:
                        partitions[i][key] = chunk
                offset += count

    for i, part in enumerate(partitions):
        with open(os.path.join(out_dir, f"client_{i}.pkl"), "wb") as f:
            pickle.dump(part, f)


def list_partitions(dataset_key: str = None) -> Dict:
    """List all available partitions."""
    result = {}
    if dataset_key:
        keys = [dataset_key]
    else:
        keys = list(DATASETS.keys())

    for key in keys:
        d = os.path.join(PARTITIONS_DIR, key)
        meta_path = os.path.join(d, "meta.json")
        if os.path.isdir(d):
            parts = sorted([f for f in os.listdir(d) if f.endswith(".pkl")])
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            result[key] = {"files": parts, "count": len(parts), "path": d, "meta": meta}

    # Also check legacy partitions (data/partitions/client_*.pkl)
    legacy = sorted([f for f in os.listdir(PARTITIONS_DIR) if f.startswith("client_") and f.endswith(".pkl")])
    if legacy:
        result["legacy"] = {"files": legacy, "count": len(legacy), "path": PARTITIONS_DIR, "meta": {"dataset": "rml2016.10a (legacy)"}}

    return result
