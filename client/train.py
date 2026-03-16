"""
Local Training Module

8 classifiers, full ML metrics, cross-validation, feature importance,
differential privacy, per-SNR model selection.
"""

import numpy as np
import os, pickle, time, logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score,
    f1_score, cohen_kappa_score,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _train(model, name, features, labels, test_split=0.3, random_state=42, verbose=True):
    """Shared training logic with comprehensive ML metrics."""
    if verbose:
        logger.info(f"Training {name} — {len(features)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_split, random_state=random_state, stratify=labels)

    t0 = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - t0

    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)

    t1 = time.time()
    test_preds = model.predict(X_test)
    inf_time = ((time.time() - t1) / max(len(X_test), 1)) * 1000

    test_acc = accuracy_score(y_test, test_preds)
    cm = confusion_matrix(y_test, test_preds)
    avg = 'weighted'
    precision = precision_score(y_test, test_preds, average=avg, zero_division=0)
    recall = recall_score(y_test, test_preds, average=avg, zero_division=0)
    f1 = f1_score(y_test, test_preds, average=avg, zero_division=0)
    kappa = cohen_kappa_score(y_test, test_preds)

    # Per-class
    unique_labels = sorted(np.unique(np.concatenate([y_train, y_test])))
    per_class = {}
    for lbl in unique_labels:
        mask = y_test == lbl
        if mask.sum() > 0:
            per_class[int(lbl)] = {
                'accuracy': float(accuracy_score(y_test[mask], test_preds[mask])),
                'count': int(mask.sum()),
            }

    # Feature importance (for tree-based models)
    feat_importance = None
    if hasattr(model, 'feature_importances_'):
        feat_importance = model.feature_importances_.tolist()

    if verbose:
        logger.info(f"{name}: acc={test_acc*100:.2f}% F1={f1:.4f} kappa={kappa:.4f} time={training_time:.3f}s")

    return {
        'model': model, 'model_name': name,
        'train_accuracy': train_acc, 'test_accuracy': test_acc,
        'precision': precision, 'recall': recall, 'f1_score': f1,
        'cohen_kappa': kappa, 'training_time': training_time,
        'inference_time_ms_per_sample': inf_time,
        'n_samples': len(X_train), 'n_test_samples': len(X_test),
        'n_features': features.shape[1], 'confusion_matrix': cm,
        'per_class': per_class, 'feature_importance': feat_importance,
    }


# ── Cross-Validation ─────────────────────────────────────────────────────────

def cross_validate(model_fn, features, labels, n_folds=5, random_state=42):
    """
    Run stratified k-fold cross-validation.
    model_fn: callable that returns a fresh sklearn model instance.
    Returns dict with per-fold scores and summary stats.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

    for fold, (train_idx, test_idx) in enumerate(cv.split(features, labels)):
        model = model_fn()
        model.fit(features[train_idx], labels[train_idx])
        preds = model.predict(features[test_idx])
        y_true = labels[test_idx]
        metrics['accuracy'].append(float(accuracy_score(y_true, preds)))
        metrics['f1'].append(float(f1_score(y_true, preds, average='weighted', zero_division=0)))
        metrics['precision'].append(float(precision_score(y_true, preds, average='weighted', zero_division=0)))
        metrics['recall'].append(float(recall_score(y_true, preds, average='weighted', zero_division=0)))

    result = {}
    for key, vals in metrics.items():
        result[key] = vals
        result[f'{key}_mean'] = float(np.mean(vals))
        result[f'{key}_std'] = float(np.std(vals))

    result['n_folds'] = n_folds
    return result


# ── Differential Privacy ──────────────────────────────────────────────────────

def apply_differential_privacy(features, epsilon=1.0, sensitivity=None):
    """
    Add calibrated Gaussian noise for epsilon-differential privacy.
    Lower epsilon = more privacy, more noise.
    """
    if sensitivity is None:
        sensitivity = np.max(np.abs(features)) * 0.01  # conservative
    sigma = sensitivity * np.sqrt(2 * np.log(1.25)) / max(epsilon, 0.01)
    noise = np.random.normal(0, sigma, features.shape)
    noisy = features + noise
    snr_db = 10 * np.log10(np.var(features) / (np.var(noise) + 1e-12))
    logger.info(f"DP: epsilon={epsilon}, sigma={sigma:.4f}, SNR={snr_db:.1f}dB")
    return noisy.astype(np.float32), {'epsilon': epsilon, 'sigma': float(sigma), 'snr_db': float(snr_db)}


# ── Per-SNR Model Selection ──────────────────────────────────────────────────

def per_snr_best_model(results_dict, features, labels, snrs):
    """
    Given multiple trained models, find which is best at each SNR band.
    results_dict: {code: training_result_dict}
    Returns: {snr: (best_model_code, accuracy)}
    """
    snr_best = {}
    unique_snrs = sorted(np.unique(snrs))
    for snr in unique_snrs:
        mask = snrs == snr
        if mask.sum() == 0:
            continue
        X_snr = features[mask]
        y_snr = labels[mask]
        best_code, best_acc = None, -1
        for code, res in results_dict.items():
            model = res['model']
            try:
                preds = model.predict(X_snr)
                acc = accuracy_score(y_snr, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_code = code
            except Exception:
                continue
        if best_code:
            snr_best[int(snr)] = {'model': best_code, 'accuracy': float(best_acc)}
    return snr_best


# ── Model Trainers ────────────────────────────────────────────────────────────

def train_knn_model(features, labels, test_split=0.3, n_neighbors=5,
                    weights='uniform', random_state=42, verbose=True):
    return _train(KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights),
                  f"KNN (k={n_neighbors}, {weights})", features, labels, test_split, random_state, verbose)

def train_dt_model(features, labels, test_split=0.3, max_depth=None,
                   min_samples_split=2, random_state=42, verbose=True):
    return _train(DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                  random_state=random_state),
                  f"Decision Tree (depth={'all' if max_depth is None else max_depth})",
                  features, labels, test_split, random_state, verbose)

def train_rf_model(features, labels, test_split=0.3, n_estimators=100,
                   max_depth=None, random_state=42, verbose=True):
    return _train(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                  random_state=random_state),
                  f"Random Forest (n={n_estimators})", features, labels, test_split, random_state, verbose)

def train_gb_model(features, labels, test_split=0.3, n_estimators=100,
                   learning_rate=0.1, max_depth=3, random_state=42, verbose=True):
    return _train(GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                  max_depth=max_depth, random_state=random_state),
                  f"Gradient Boosting (n={n_estimators}, lr={learning_rate})",
                  features, labels, test_split, random_state, verbose)

def train_svm_model(features, labels, test_split=0.3, kernel='rbf',
                    C=1.0, random_state=42, verbose=True):
    return _train(SVC(kernel=kernel, C=C, random_state=random_state),
                  f"SVM ({kernel}, C={C})", features, labels, test_split, random_state, verbose)

def train_lr_model(features, labels, test_split=0.3, C=1.0,
                   max_iter=1000, random_state=42, verbose=True):
    return _train(LogisticRegression(C=C, max_iter=max_iter, random_state=random_state),
                  f"Logistic Regression (C={C})", features, labels, test_split, random_state, verbose)

def train_nb_model(features, labels, test_split=0.3, verbose=True):
    return _train(GaussianNB(), "Gaussian Naive Bayes", features, labels, test_split, 42, verbose)

def train_mlp_model(features, labels, test_split=0.3, hidden_layers=(64, 32),
                    max_iter=300, random_state=42, verbose=True):
    return _train(MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=max_iter,
                  random_state=random_state),
                  f"MLP {hidden_layers}", features, labels, test_split, random_state, verbose)


# ── Model factory (for CV) ───────────────────────────────────────────────────

MODEL_FACTORIES = {
    'knn': lambda **kw: KNeighborsClassifier(n_neighbors=kw.get('k', 5), weights=kw.get('weights', 'uniform')),
    'dt':  lambda **kw: DecisionTreeClassifier(max_depth=kw.get('max_depth'), random_state=42),
    'rf':  lambda **kw: RandomForestClassifier(n_estimators=kw.get('n_estimators', 100), random_state=42),
    'gb':  lambda **kw: GradientBoostingClassifier(n_estimators=kw.get('n_estimators', 100),
                         learning_rate=kw.get('learning_rate', 0.1), random_state=42),
    'svm': lambda **kw: SVC(kernel=kw.get('kernel', 'rbf'), C=kw.get('C', 1.0), random_state=42),
    'lr':  lambda **kw: LogisticRegression(C=kw.get('C', 1.0), max_iter=1000, random_state=42),
    'nb':  lambda **kw: GaussianNB(),
    'mlp': lambda **kw: MLPClassifier(hidden_layer_sizes=kw.get('hidden_layers', (64, 32)),
                         max_iter=300, random_state=42),
}


# ── Save / Load ──────────────────────────────────────────────────────────────

def _save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f: pickle.dump(model, f)

def _load_model(path):
    if not os.path.exists(path): raise FileNotFoundError(f"Not found: {path}")
    with open(path, 'rb') as f: return pickle.load(f)

save_knn_model = save_dt_model = save_rf_model = _save_model
save_svm_model = save_lr_model = save_nb_model = _save_model
save_gb_model = save_mlp_model = _save_model
load_knn_model = load_dt_model = _load_model
