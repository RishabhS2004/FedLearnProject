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
    python simulate.py --compare-strategies      # Compare FedAvg, Krum, Trimmed Mean
"""

import sys, os, argparse, json, time, pickle
import numpy as np
import pandas as pd
import contextlib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.dataset_loader import load_radioml_dataset, get_dataset_info, flatten_dataset
from client.feature_extract import extract_features, normalize_features
from client.train import (
    train_knn_model, train_dt_model, train_rf_model, train_gb_model,
    train_svm_model, train_lr_model, train_nb_model, train_mlp_model,
    train_mlp_incremental, cross_validate, apply_differential_privacy, per_snr_best_model,
    MODEL_FACTORIES,
)
from central.aggregator import aggregate_knn_models, evaluate_global_model, generate_synthetic_snr_values, aggregate_mlp_fedavg
from central.byzantine import (
    ByzantineResilientAggregator, reset_trust_scores, initialize_trust,
    get_all_trust_scores, advance_adaptive_threshold,
)
from data.datasets import partition_dataset, list_partitions, load_dataset, PARTITIONS_DIR
from tests.experiment_tracker import ExperimentTracker
from central.state import (
    register_client_connection,
    register_client_upload,
    store_aggregation_result,
    store_aggregation_round,
    save_auto_aggregation_state,
    reset_aggregation_state,
    initialize_metrics_history,
    clear_client_registry,
    clear_aggregation_results,
    set_round,
    track_client_upload,
)


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
    dropout_rate=0.0,
    verbose=True,
    tracker=None,
    random_seed=42,
    smote=False,
):
    """Run a full FL simulation and return results DataFrame."""
    
    # Determine training mode
    is_true_fedavg = "mlp" in models_to_train
    training_mode = "true_fedavg" if is_true_fedavg else "data_accumulation_simulation"
    
    # Setup tracker if not provided
    own_tracker = False
    if tracker is None:
        tracker = ExperimentTracker(
            strategy=byzantine_strategy,
            model=models_to_train[0] if models_to_train else "unknown",
            feature_mode=feature_mode,
            dataset=dataset_key,
            num_clients=num_clients,
            num_byzantine=num_byzantine,
            n_rounds=n_rounds,
            distribution=distribution,
            training_mode=training_mode,
            random_seed=random_seed,
        )
        own_tracker = True
    else:
        tracker.training_mode = training_mode

    with tracker if own_tracker else contextlib.nullcontext():
        # ── Dashboard State: Initialization ──
        try:
            initialize_metrics_history()
            clear_client_registry()
            clear_aggregation_results()
            set_round(0)
            save_auto_aggregation_state({
                'enabled': True,
                'threshold': num_clients,
                'pending_uploads': 0,
                'current_round': 0,
                'clients_uploaded_this_round': [],
                'last_aggregation_time': None,
            })
        except Exception:
            pass  # Dashboard sync is best-effort; don't break simulation

        if verbose:
            print(f"\n{'='*60}")
            print(f"  RadioFed Simulation ({training_mode})")
            print(f"  {num_clients} clients ({num_byzantine} Byzantine), {n_rounds} rounds")
            print(f"  Features: {feature_mode}, Distribution: {distribution}")
            print(f"  Byzantine defense: {byzantine_strategy}")
            print(f"{'='*60}\n")

        # 1. Partition
        ok, msg = partition_dataset(dataset_key, num_clients, filter_mode, distribution, dirichlet_alpha, random_seed=random_seed)
        if not ok:
            print(f"Partition failed: {msg}")
            return None
        if verbose:
            print(f"[1/5] Partitioned: {msg}")

        # 2. Load partitions + extract features
        parts_dir = os.path.join(PARTITIONS_DIR, dataset_key)
        client_features = []
        client_labels = []
        global_test_features_list = []
        global_test_labels_list = []
        from sklearn.model_selection import train_test_split

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

            # Split train and test before DP and Byzantine poisoning
            X_train, X_test, y_train, y_test = train_test_split(
                feats, labels, test_size=0.3, random_state=42, stratify=labels)

            if i < num_clients - num_byzantine or num_byzantine == 0:
                global_test_features_list.append(X_test)
                global_test_labels_list.append(y_test)

            # Differential privacy
            if dp_epsilon is not None:
                X_train, _ = apply_differential_privacy(X_train, epsilon=dp_epsilon)

            # Byzantine: poison the last num_byzantine clients
            if i >= num_clients - num_byzantine and num_byzantine > 0:
                if verbose:
                    print(f"  Client {i}: BYZANTINE (poisoning features)")
                X_train = X_train * np.random.uniform(-5, 5, X_train.shape).astype(np.float32)
                np.random.shuffle(y_train)  # scramble labels too

            client_features.append(X_train)
            client_labels.append(y_train)
            
        global_test_features = np.concatenate(global_test_features_list) if global_test_features_list else np.array([])
        global_test_labels = np.concatenate(global_test_labels_list) if global_test_labels_list else np.array([])

        if verbose:
            print(f"[2/5] Extracted {feature_mode} features for {num_clients} clients")

        # 3. Simulate rounds
        reset_trust_scores()
        all_round_results = []
        
        # Initialize global state for true FedAvg
        global_models = {m: None for m in models_to_train if m == "mlp"}

        for round_num in range(n_rounds):
            if verbose:
                print(f"\n--- Round {round_num + 1}/{n_rounds} ---")

            # Initialize trust
            client_ids = [f"client_{i}" for i in range(num_clients)]
            for cid in client_ids:
                initialize_trust(cid)

            # Dropout simulation
            active_indices = []
            dropped_clients = []
            for i, cid in enumerate(client_ids):
                if np.random.rand() >= dropout_rate:
                    active_indices.append(i)
                else:
                    dropped_clients.append(cid)

            if dropped_clients and verbose:
                print(f"  Dropout simulation: {len(dropped_clients)} clients disconnected ({', '.join(dropped_clients)})")

            active_features = []
            active_labels = []
            active_ids = [client_ids[i] for i in active_indices]

            # ── Dashboard State: Register active clients ──
            try:
                for cid in active_ids:
                    register_client_connection(cid)
            except Exception:
                pass

            # Progressive Data Accumulation (for KNN/RF) or Full Data (for MLP)
            if is_true_fedavg:
                # MLP uses all available local data per round for incremental partial_fit
                round_fraction = 1.0
            else:
                # Accumulate data over rounds for classical ML
                round_fraction = (round_num + 1) / n_rounds

            for i in active_indices:
                feats = client_features[i]
                lbls = client_labels[i]
                n_samples = max(1, int(len(feats) * round_fraction))
                active_features.append(feats[:n_samples])
                active_labels.append(lbls[:n_samples])

                # ── Dashboard State: Register client upload ──
                try:
                    register_client_upload(
                        client_id=client_ids[i],
                        n_samples=n_samples,
                        weights_path='simulation',
                        model_type='knn',
                    )
                    track_client_upload(client_ids[i])
                except Exception:
                    pass

            # Byzantine filtering
            aggregator = ByzantineResilientAggregator(strategy=byzantine_strategy)
            result = aggregator.filter_and_aggregate(
                active_features, active_labels, active_ids
            )

            merged_features = result['features']
            merged_labels = result['labels']
            accepted = result['accepted_clients']
            rejected = result['rejected_clients']

            if verbose:
                print(f"  Byzantine filter: {len(accepted)} accepted, {len(rejected)} rejected")
                for cid, reason in rejected:
                    print(f"    Rejected {cid}: {reason}")

            # Train models
            round_metrics = {'round': round_num + 1}
            primary_eval_metrics = {}
            _primary = models_to_train[0] if models_to_train else 'knn'
            
            for model_code in models_to_train:
                if model_code == 'mlp':
                    # TRUE FEDAVG WORKFLOW
                    client_models_this_round = []
                    n_samples_per_client = []
                    
                    # Extract features of accepted clients
                    accepted_idx = [active_ids.index(cid) for cid in accepted]
                    
                    if not accepted_idx:
                        continue
                        
                    for idx in accepted_idx:
                        feats = active_features[idx]
                        lbls = active_labels[idx]
                        
                        # Local Incremental Training
                        res = train_mlp_incremental(
                            features=feats, labels=lbls,
                            global_model=global_models["mlp"],
                            verbose=False,
                            smote=smote
                        )
                        client_models_this_round.append(res['model'])
                        n_samples_per_client.append(res['n_samples'])
                    
                    # Server Aggregation via existing aggregator function
                    agg_res = aggregate_mlp_fedavg(
                        client_models=client_models_this_round,
                        n_samples_per_client=n_samples_per_client
                    )
                    global_models["mlp"] = agg_res['global_model']
                    
                    # Evaluate global model
                    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
                    from sklearn.model_selection import train_test_split
                    # Evaluate against merged accepted data to get metrics
                    X_train, X_test, y_train, y_test = train_test_split(
                        merged_features, merged_labels, test_size=0.3, random_state=random_seed, stratify=merged_labels)
                    
                    if smote:
                        try:
                            from imblearn.over_sampling import SMOTE
                            sm = SMOTE(random_state=random_seed)
                            X_train, y_train = sm.fit_resample(X_train, y_train)
                        except ImportError:
                            pass

                    preds = global_models["mlp"].predict(X_test)
                    round_metrics['mlp_accuracy'] = accuracy_score(y_test, preds)
                    round_metrics['mlp_f1'] = f1_score(y_test, preds, average='weighted', zero_division=0)
                    round_metrics['mlp_kappa'] = cohen_kappa_score(y_test, preds)
                    round_metrics['mlp_train_time'] = 0.0 # Time logged in client
                    
                    if model_code == _primary:
                        from central.metrics import compute_full_metrics, compute_snr_metrics
                        from central.aggregator import generate_synthetic_snr_values
                        from sklearn.metrics import confusion_matrix
                        conf_matrix = confusion_matrix(y_test, preds)
                        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
                        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)
                        full_metrics = compute_full_metrics(y_test, preds)
                        snr_test = generate_synthetic_snr_values(len(X_test))
                        snr_metrics = compute_snr_metrics(y_test, preds, snr_test, min_samples=1)
                        
                        unique_labels = sorted(np.unique(np.concatenate([y_test, preds])))
                        class_names = [f"Class {lbl}" for lbl in unique_labels]

                        primary_eval_metrics = {
                            'accuracy': full_metrics['accuracy'],
                            'per_snr_accuracy': snr_metrics['per_snr_accuracy'],
                            'confusion_matrix': conf_matrix.tolist(),
                            'normalized_confusion_matrix': conf_matrix_norm.tolist(),
                            'n_samples': len(X_test),
                            'f1_macro': full_metrics['f1_macro'],
                            'f1_weighted': full_metrics['f1_weighted'],
                            'precision_macro': full_metrics['precision_macro'],
                            'recall_macro': full_metrics['recall_macro'],
                            'training_time': 0.0,
                            'classification_report': full_metrics.get('classification_report', {}),
                            'class_metrics': full_metrics.get('per_class', {}),
                            'labels': class_names
                        }

                    if verbose:
                        print(f"  MLP (FedAvg): acc={round_metrics['mlp_accuracy']*100:.2f}% F1={round_metrics['mlp_f1']:.4f}")
                        
                else:
                    # DATA ACCUMULATION SIMULATION WORKFLOW
                    trainer = {
                        'knn': lambda f, l: train_knn_model(f, l, verbose=False, smote=smote),
                        'dt':  lambda f, l: train_dt_model(f, l, verbose=False, smote=smote),
                        'rf':  lambda f, l: train_rf_model(f, l, verbose=False, smote=smote),
                        'gb':  lambda f, l: train_gb_model(f, l, verbose=False, smote=smote),
                        'svm': lambda f, l: train_svm_model(f, l, verbose=False, smote=smote),
                        'lr':  lambda f, l: train_lr_model(f, l, verbose=False, smote=smote),
                        'nb':  lambda f, l: train_nb_model(f, l, verbose=False, smote=smote),
                    }.get(model_code)

                    if trainer and len(merged_features) > 0:
                        res = trainer(merged_features, merged_labels)
                        round_metrics[f'{model_code}_accuracy'] = res['test_accuracy']
                        round_metrics[f'{model_code}_f1'] = res['f1_score']
                        round_metrics[f'{model_code}_kappa'] = res['cohen_kappa']
                        round_metrics[f'{model_code}_train_time'] = res['training_time']
                        
                        if model_code == _primary:
                            from sklearn.model_selection import train_test_split
                            from central.metrics import compute_full_metrics, compute_snr_metrics
                            from central.aggregator import generate_synthetic_snr_values
                            from sklearn.metrics import confusion_matrix
                            
                            X_train, X_test, y_train, y_test = train_test_split(
                                merged_features, merged_labels, test_size=0.3, random_state=random_seed, stratify=merged_labels)
                            
                            preds = res['model'].predict(X_test)
                            conf_matrix = confusion_matrix(y_test, preds)
                            conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
                            conf_matrix_norm = np.nan_to_num(conf_matrix_norm)
                            full_metrics = compute_full_metrics(y_test, preds)
                            snr_test = generate_synthetic_snr_values(len(X_test))
                            snr_metrics = compute_snr_metrics(y_test, preds, snr_test, min_samples=1)
                            
                            unique_labels = sorted(np.unique(np.concatenate([y_test, preds])))
                            class_names = [f"Class {lbl}" for lbl in unique_labels]
                            
                            primary_eval_metrics = {
                                'accuracy': full_metrics['accuracy'],
                                'per_snr_accuracy': snr_metrics['per_snr_accuracy'],
                                'confusion_matrix': conf_matrix.tolist(),
                                'normalized_confusion_matrix': conf_matrix_norm.tolist(),
                                'n_samples': len(X_test),
                                'f1_macro': full_metrics['f1_macro'],
                                'f1_weighted': full_metrics['f1_weighted'],
                                'precision_macro': full_metrics['precision_macro'],
                                'recall_macro': full_metrics['recall_macro'],
                                'training_time': res['training_time'],
                                'classification_report': full_metrics.get('classification_report', {}),
                                'class_metrics': full_metrics.get('per_class', {}),
                                'labels': class_names
                            }

                        if verbose:
                            print(f"  {model_code.upper()}: acc={res['test_accuracy']*100:.2f}% F1={res['f1_score']:.4f}")

            round_metrics['n_accepted'] = len(accepted)
            round_metrics['n_rejected'] = len(rejected)
            round_metrics['total_samples'] = len(merged_labels)

            # ── Dashboard State: Store aggregation results + round metrics ──
            try:
                from datetime import datetime as _dt
                _ts = _dt.now().isoformat()

                # Determine the primary model for dashboard display
                _primary = models_to_train[0] if models_to_train else 'knn'
                _cur_acc = round_metrics.get(f'{_primary}_accuracy', 0.0)

                # Build a result dict that matches what the dashboard expects
                _agg_result = {
                    'accuracy': _cur_acc,
                    'per_snr_accuracy': primary_eval_metrics.get('per_snr_accuracy', {}),
                    'confusion_matrix': primary_eval_metrics.get('confusion_matrix', []),
                    'normalized_confusion_matrix': primary_eval_metrics.get('normalized_confusion_matrix', []),
                    'num_clients': len(accepted),
                    'total_samples': len(merged_labels),
                    'training_time': primary_eval_metrics.get('training_time', 0.0),
                    'inference_time_ms_per_sample': 0.0,
                    'feature_dim': merged_features.shape[1] if len(merged_features) > 0 else 0,
                    'n_neighbors': 5,
                    'n_test_samples': primary_eval_metrics.get('n_samples', 0),
                    'macro_f1': primary_eval_metrics.get('f1_macro', 0),
                    'weighted_f1': primary_eval_metrics.get('f1_weighted', 0),
                    'precision': primary_eval_metrics.get('precision_macro', 0),
                    'recall': primary_eval_metrics.get('recall_macro', 0),
                    'support': primary_eval_metrics.get('n_samples', 0),
                    'classification_report': primary_eval_metrics.get('classification_report', {}),
                    'class_metrics': primary_eval_metrics.get('class_metrics', {}),
                    'labels': primary_eval_metrics.get('labels', []),
                }
                # Store as 'knn' type so the dashboard can read it
                store_aggregation_result('knn', _agg_result, _ts)

                # Store before/after for accuracy trends
                _prev_acc = 0.0
                if len(all_round_results) > 0:
                    _prev_acc = all_round_results[-1].get(f'{_primary}_accuracy', 0.0)
                _before = {
                    'knn_accuracy': _prev_acc,
                    'per_snr_accuracy': {},
                    'confusion_matrix': [],
                    'num_clients': len(accepted),
                    'timestamp': _ts,
                }
                _after = {
                    'knn_accuracy': _cur_acc,
                    'per_snr_accuracy': primary_eval_metrics.get('per_snr_accuracy', {}),
                    'confusion_matrix': primary_eval_metrics.get('confusion_matrix', []),
                    'timestamp': _ts,
                }
                store_aggregation_round(_before, _after)

                # Advance the auto-aggregation round counter
                reset_aggregation_state()
            except Exception:
                pass

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
                if model_code == 'mlp':
                    continue # Skip CV for iterative FedAvg
                factory = MODEL_FACTORIES.get(model_code)
                if factory:
                    cv = cross_validate(factory, merged_features, merged_labels, n_folds=5)
                    print(f"  {model_code.upper()}: {cv['accuracy_mean']:.4f} +/- {cv['accuracy_std']:.4f}")

        # 6. Final Global Evaluation & Plot Generation
        if verbose:
            print(f"\n--- Final Global Evaluation & Plot Generation ---")
            
        if len(global_test_labels) > 0:
            test_snrs = generate_synthetic_snr_values(len(global_test_labels))
            
            for model_code in models_to_train:
                final_model = None
                if model_code == 'mlp':
                    final_model = global_models["mlp"]
                else:
                    factory = MODEL_FACTORIES.get(model_code)
                    if factory and len(merged_features) > 0:
                        final_model = factory()
                        final_model.fit(merged_features, merged_labels)
                        
                if final_model is not None:
                    if verbose:
                        print(f"  Evaluating final {model_code.upper()} model and generating plots...")
                    evaluate_global_model(
                        model=final_model,
                        test_features=global_test_features,
                        test_labels=global_test_labels,
                        test_snrs=test_snrs
                    )

        # Save results
        os.makedirs("out/reports", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = f"out/reports/sim_{ts}.csv"
        df.to_csv(out_path, index=False)
        if verbose:
            print(f"\nResults saved to {out_path}")

        if own_tracker and tracker:
            best_acc = df[f'{models_to_train[0]}_accuracy'].max() if not df.empty and f'{models_to_train[0]}_accuracy' in df else 0
            best_f1 = df[f'{models_to_train[0]}_f1'].max() if not df.empty and f'{models_to_train[0]}_f1' in df else 0
            tracker.update_metrics({
                "best_accuracy": best_acc,
                "best_f1_macro": best_f1,
                "rounds_completed": len(df),
                "training_mode": training_mode
            })

        return df


def simulate_compare_strategies(
    dataset_key="rml2016.10a",
    filter_mode="analog",
    num_clients=4,
    num_byzantine=1,
    feature_mode="16d",
    models_to_train=("knn",),
    n_rounds=3,
    distribution="iid",
    dirichlet_alpha=0.5,
    dp_epsilon=None,
    strategies=("fedavg", "krum", "trimmed_mean"),
    verbose=True,
):
    """Run simulation across multiple strategies and generate comparative plots."""
    
    is_true_fedavg = "mlp" in models_to_train
    training_mode = "true_fedavg" if is_true_fedavg else "data_accumulation_simulation"
    
    tracker = ExperimentTracker(
        strategy="compare_all",
        model=models_to_train[0] if models_to_train else "unknown",
        feature_mode=feature_mode,
        dataset=dataset_key,
        num_clients=num_clients,
        num_byzantine=num_byzantine,
        n_rounds=n_rounds,
        distribution=distribution,
        training_mode=training_mode,
        random_seed=42,
    )

    with tracker:
        all_strategy_results = {}
        all_dfs = {}

        for strategy in strategies:
            display_name = {
                "fedavg": "FedAvg",
                "krum": "Krum",
                "trimmed_mean": "Trimmed Mean",
                "trust_weighted": "Trust Weighted",
                "full": "Full Pipeline",
            }.get(strategy, strategy)

            if verbose:
                print(f"\n{'#' * 60}")
                print(f"  Strategy: {display_name}")
                print(f"{'#' * 60}")

            if strategy == "fedavg":
                df = simulate(
                    dataset_key=dataset_key, filter_mode=filter_mode,
                    num_clients=num_clients, num_byzantine=0,  # FedAvg = no Byzantine
                    feature_mode=feature_mode, models_to_train=models_to_train,
                    n_rounds=n_rounds, distribution=distribution,
                    dirichlet_alpha=dirichlet_alpha, dp_epsilon=dp_epsilon,
                    byzantine_strategy="krum", dropout_rate=0.0,
                    verbose=verbose, tracker=tracker
                )
            else:
                df = simulate(
                    dataset_key=dataset_key, filter_mode=filter_mode,
                    num_clients=num_clients, num_byzantine=num_byzantine,
                    feature_mode=feature_mode, models_to_train=models_to_train,
                    n_rounds=n_rounds, distribution=distribution,
                    dirichlet_alpha=dirichlet_alpha, dp_epsilon=dp_epsilon,
                    byzantine_strategy=strategy, dropout_rate=0.0,
                    verbose=verbose, tracker=tracker
                )

            if df is not None:
                all_dfs[strategy] = df
                rounds_data = df.to_dict("records")
                all_strategy_results[strategy] = rounds_data

        if not all_strategy_results:
            print("No strategy results collected.")
            return {}

        # Generate comparative plots
        try:
            from central.evaluation_plots import (
                plot_byzantine_accuracy_comparison,
                plot_byzantine_f1_comparison,
                plot_byzantine_client_acceptance,
            )

            ts = time.strftime("%Y%m%d_%H%M%S")
            model_code = models_to_train[0] if models_to_train else "knn"

            acc_path = plot_byzantine_accuracy_comparison(
                all_strategy_results,
                metric_key=f"{model_code}_accuracy",
                title=f"{model_code.upper()} Accuracy vs. Federated Round ({training_mode})",
                timestamp=ts,
            )
            
            f1_path = plot_byzantine_f1_comparison(
                all_strategy_results,
                metric_key=f"{model_code}_f1",
                title=f"{model_code.upper()} F1 Macro vs. Federated Round ({training_mode})",
                timestamp=ts,
            )
            
            accept_path = plot_byzantine_client_acceptance(
                all_strategy_results,
                title="Client Acceptance/Rejection per Round",
                timestamp=ts,
            )
            
            tracker.update_metrics({
                "plots_generated": 3,
                "strategies_compared": list(strategies)
            })

        except Exception as e:
            print(f"  WARNING: Could not generate Byzantine comparison plots: {e}")
            import traceback
            traceback.print_exc()

        # Save combined results JSON
        os.makedirs("out/metrics", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        combined_path = f"out/metrics/byzantine_comparison_{ts}.json"
        try:
            serializable = {}
            for strategy, rounds_data in all_strategy_results.items():
                serializable[strategy] = [
                    {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in r.items()}
                    for r in rounds_data
                ]
            with open(combined_path, "w") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            print(f"  WARNING: Could not save combined results: {e}")

        return all_strategy_results
 
 
def simulate_multi_run(
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
    dropout_rate=0.0,
    num_runs=5,
    verbose=True,
    smote=False,
):
    """Run full experiment N times and aggregate results."""
    all_runs_results = {} # strategy -> List[RunData]
    
    # We always compare FedAvg vs Selected Strategy in multi-run for context
    strategies = ["fedavg", byzantine_strategy] if byzantine_strategy != "fedavg" else ["fedavg"]
    
    for strategy in strategies:
        print(f"\n>>> STATISTICAL EVALUATION: Strategy={strategy}, Runs={num_runs}")
        strategy_runs = []
        for run_idx in range(num_runs):
            seed = 42 + run_idx
            print(f"  Run {run_idx+1}/{num_runs} (seed={seed})...")
            
            # Run simulation
            df = simulate(
                dataset_key=dataset_key,
                filter_mode=filter_mode,
                num_clients=num_clients,
                num_byzantine=num_byzantine,
                feature_mode=feature_mode,
                models_to_train=models_to_train,
                n_rounds=n_rounds,
                distribution=distribution,
                dirichlet_alpha=dirichlet_alpha,
                dp_epsilon=dp_epsilon,
                byzantine_strategy=strategy,
                dropout_rate=dropout_rate,
                verbose=False, # less noise
                random_seed=seed,
                smote=smote
            )
            
            if df is not None:
                # Convert DF to list of dicts for plotting function
                strategy_runs.append(df.to_dict(orient='records'))
        
        all_runs_results[strategy] = strategy_runs
 
    # Aggregate and Plot
    from central.evaluation_plots import plot_statistical_comparison
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_code = models_to_train[0] if models_to_train else "knn"
    
    # Accuracy Plot
    plot_statistical_comparison(
        all_runs_results,
        metric_key=f"{model_code}_accuracy",
        title=f"Mean Accuracy vs. Round (n={num_runs}, {distribution})",
        timestamp=ts
    )
    
    # F1 Plot
    plot_statistical_comparison(
        all_runs_results,
        metric_key=f"{model_code}_f1",
        title=f"Mean F1 Macro vs. Round (n={num_runs}, {distribution})",
        ylabel="F1 Macro (%)",
        timestamp=ts
    )
    
    # Save statistics CSV
    stats = []
    for strategy, runs in all_runs_results.items():
        if not runs: continue
        final_round_idx = -1
        final_vals = [run[final_round_idx][f"{model_code}_accuracy"] for run in runs]
        f1_vals = [run[final_round_idx][f"{model_code}_f1"] for run in runs]
        
        stats.append({
            "strategy": strategy,
            "mean_accuracy": np.mean(final_vals),
            "std_accuracy": np.std(final_vals),
            "mean_f1": np.mean(f1_vals),
            "std_f1": np.std(f1_vals),
            "num_runs": num_runs
        })
    
    stats_df = pd.DataFrame(stats)
    os.makedirs("out/reports", exist_ok=True)
    stats_path = f"out/reports/statistical_summary_{ts}.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStatistical summary saved to {stats_path}")
    print(stats_df.to_string())
    
    return all_runs_results


def main():
    parser = argparse.ArgumentParser(description="RadioFed FL Simulation")
    parser.add_argument("--dataset", default="rml2016.10a")
    parser.add_argument("--filter", default="analog", choices=["all", "analog"])
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--byzantine", type=int, default=1)
    parser.add_argument("--features", default="16d", choices=["8d", "16d", "24d", "32d"])
    parser.add_argument("--models", default="knn,rf", help="Comma-separated model codes")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--noniid", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for non-IID")
    parser.add_argument("--dp-epsilon", type=float, default=None, help="DP epsilon (None=disabled)")
    parser.add_argument("--defense", default="krum", choices=["krum", "trimmed_mean", "trust_weighted", "full"])
    parser.add_argument("--dropout", type=float, default=0.0, help="Probability of a client disconnecting in each round")
    parser.add_argument("--compare-strategies", action="store_true", help="Run FedAvg/Krum/Trimmed Mean comparison")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for statistical evaluation (default=1)")
    parser.add_argument("--smote", action="store_true", help="Enable SMOTE oversampling for minority classes")
    args = parser.parse_args()

    if args.runs > 1:
        simulate_multi_run(
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
            dropout_rate=args.dropout,
            num_runs=args.runs,
            verbose=True,
            smote=args.smote
        )
    elif args.compare_strategies:
        simulate_compare_strategies(
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
            verbose=True,
        )
    else:
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
            dropout_rate=args.dropout,
            verbose=True,
            smote=args.smote
        )


if __name__ == "__main__":
    main()
