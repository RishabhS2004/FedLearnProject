"""
RadioFed Multi-Client Simulation Runner

Programmatically simulates a full federated learning experiment:
- N honest clients + M Byzantine clients
- Configurable feature mode, models, partitioning
- Generates results table and summary

Usage:
    python simulate.py                           # defaults
    python simulate.py --clients 6 --byzantine 2 --rounds 5 --features 24d
    python simulate.py --noniid --alpha 0.3      # non-IID Dirichlet
"""

import sys, os, argparse, json, time, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client.dataset_loader import load_radioml_dataset, get_dataset_info, flatten_dataset
from client.feature_extract import extract_features, normalize_features
from client.train import (
    train_knn_model, train_dt_model, train_rf_model, train_gb_model,
    train_svm_model, train_lr_model, train_nb_model, train_mlp_model,
    cross_validate, apply_differential_privacy, per_snr_best_model,
    MODEL_FACTORIES,
)
from central.aggregator import aggregate_knn_models, evaluate_global_model, generate_synthetic_snr_values
from central.byzantine import (
    ByzantineResilientAggregator, reset_trust_scores, initialize_trust,
    get_all_trust_scores, advance_adaptive_threshold,
)
from data.datasets import partition_dataset, list_partitions, load_dataset, PARTITIONS_DIR


def simulate(
    dataset_key="rml2016.10a",
    filter_mode="analog",
    num_clients=4,
    num_byzantine=1,
    feature_mode="16d",
    models_to_train=("knn", "rf"),
    n_rounds=3,
    distribution="iid",
    dirichlet_alpha=0.5,
    dp_epsilon=None,
    byzantine_strategy="krum",
    verbose=True,
):
    """Run a full FL simulation and return results DataFrame."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"  RadioFed Simulation")
        print(f"  {num_clients} clients ({num_byzantine} Byzantine), {n_rounds} rounds")
        print(f"  Features: {feature_mode}, Distribution: {distribution}")
        print(f"  Byzantine defense: {byzantine_strategy}")
        print(f"{'='*60}\n")

    # 1. Partition
    ok, msg = partition_dataset(dataset_key, num_clients, filter_mode, distribution, dirichlet_alpha)
    if not ok:
        print(f"Partition failed: {msg}")
        return None
    if verbose:
        print(f"[1/5] Partitioned: {msg}")

    # 2. Load partitions + extract features
    parts_dir = os.path.join(PARTITIONS_DIR, dataset_key)
    client_features = []
    client_labels = []

    for i in range(num_clients):
        path = os.path.join(parts_dir, f"client_{i}.pkl")
        ds = load_radioml_dataset(path)
        samples, labels = flatten_dataset(ds)

        # Extract features
        fl = []
        dim = {"8d": 8, "16d": 16, "24d": 24}.get(feature_mode, 16)
        for j in range(samples.shape[0]):
            try:
                fl.append(extract_features(samples[j], mode=feature_mode))
            except Exception:
                fl.append(np.zeros(dim, dtype=np.float32))
        feats = np.array(fl, dtype=np.float32)
        feats, _, _ = normalize_features(feats)

        # Differential privacy
        if dp_epsilon is not None:
            feats, _ = apply_differential_privacy(feats, epsilon=dp_epsilon)

        # Byzantine: poison the last num_byzantine clients
        if i >= num_clients - num_byzantine:
            if verbose:
                print(f"  Client {i}: BYZANTINE (poisoning features)")
            feats = feats * np.random.uniform(-5, 5, feats.shape).astype(np.float32)
            np.random.shuffle(labels)  # scramble labels too

        client_features.append(feats)
        client_labels.append(labels)

    if verbose:
        print(f"[2/5] Extracted {feature_mode} features for {num_clients} clients")

    # 3. Simulate rounds
    reset_trust_scores()
    all_round_results = []

    for round_num in range(n_rounds):
        if verbose:
            print(f"\n--- Round {round_num + 1}/{n_rounds} ---")

        # Initialize trust
        client_ids = [f"client_{i}" for i in range(num_clients)]
        for cid in client_ids:
            initialize_trust(cid)

        # Byzantine filtering
        aggregator = ByzantineResilientAggregator(strategy=byzantine_strategy)
        result = aggregator.filter_and_aggregate(
            client_features, client_labels, client_ids
        )

        merged_features = result['features']
        merged_labels = result['labels']
        accepted = result['accepted_clients']
        rejected = result['rejected_clients']

        if verbose:
            print(f"  Byzantine filter: {len(accepted)} accepted, {len(rejected)} rejected")
            for cid, reason in rejected:
                print(f"    Rejected {cid}: {reason}")

        # Train models on merged data
        round_metrics = {'round': round_num + 1}
        for model_code in models_to_train:
            trainer = {
                'knn': lambda f, l: train_knn_model(f, l, verbose=False),
                'dt':  lambda f, l: train_dt_model(f, l, verbose=False),
                'rf':  lambda f, l: train_rf_model(f, l, verbose=False),
                'gb':  lambda f, l: train_gb_model(f, l, verbose=False),
                'svm': lambda f, l: train_svm_model(f, l, verbose=False),
                'lr':  lambda f, l: train_lr_model(f, l, verbose=False),
                'nb':  lambda f, l: train_nb_model(f, l, verbose=False),
                'mlp': lambda f, l: train_mlp_model(f, l, verbose=False),
            }.get(model_code)

            if trainer:
                res = trainer(merged_features, merged_labels)
                round_metrics[f'{model_code}_accuracy'] = res['test_accuracy']
                round_metrics[f'{model_code}_f1'] = res['f1_score']
                round_metrics[f'{model_code}_kappa'] = res['cohen_kappa']
                round_metrics[f'{model_code}_train_time'] = res['training_time']
                if verbose:
                    print(f"  {model_code.upper()}: acc={res['test_accuracy']*100:.2f}% F1={res['f1_score']:.4f}")

        round_metrics['n_accepted'] = len(accepted)
        round_metrics['n_rejected'] = len(rejected)
        round_metrics['total_samples'] = len(merged_labels)

        # Advance adaptive threshold
        advance_adaptive_threshold(len(accepted), len(rejected))
        trust_scores = get_all_trust_scores()
        round_metrics['avg_trust'] = np.mean(list(trust_scores.values())) if trust_scores else 0

        all_round_results.append(round_metrics)

    # 4. Build results DataFrame
    df = pd.DataFrame(all_round_results)

    if verbose:
        print(f"\n{'='*60}")
        print("  SIMULATION RESULTS")
        print(f"{'='*60}")
        print(df.to_markdown(index=False, floatfmt='.4f'))

    # 5. Cross-validation on final merged data
    if verbose:
        print(f"\n--- 5-Fold Cross-Validation (final merged data) ---")
        for model_code in models_to_train:
            factory = MODEL_FACTORIES.get(model_code)
            if factory:
                cv = cross_validate(factory, merged_features, merged_labels, n_folds=5)
                print(f"  {model_code.upper()}: {cv['accuracy_mean']:.4f} +/- {cv['accuracy_std']:.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"results/sim_{ts}.csv"
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"\nResults saved to {out_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="RadioFed FL Simulation")
    parser.add_argument("--dataset", default="rml2016.10a")
    parser.add_argument("--filter", default="analog", choices=["all", "analog"])
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--byzantine", type=int, default=1)
    parser.add_argument("--features", default="16d", choices=["8d", "16d", "24d"])
    parser.add_argument("--models", default="knn,rf", help="Comma-separated model codes")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--noniid", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for non-IID")
    parser.add_argument("--dp-epsilon", type=float, default=None, help="DP epsilon (None=disabled)")
    parser.add_argument("--defense", default="krum", choices=["krum", "trimmed_mean", "trust_weighted", "full"])
    args = parser.parse_args()

    simulate(
        dataset_key=args.dataset,
        filter_mode=args.filter,
        num_clients=args.clients,
        num_byzantine=args.byzantine,
        feature_mode=args.features,
        models_to_train=args.models.split(","),
        n_rounds=args.rounds,
        distribution="noniid" if args.noniid else "iid",
        dirichlet_alpha=args.alpha,
        dp_epsilon=args.dp_epsilon,
        byzantine_strategy=args.defense,
    )


if __name__ == "__main__":
    main()
