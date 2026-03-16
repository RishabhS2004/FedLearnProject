"""
Byzantine Fault Tolerance Module for Federated Learning

Implements Byzantine-resilient aggregation strategies to detect and mitigate
malicious or faulty client updates in the federated learning system.

Strategies implemented:
1. Krum: Selects the client update closest to others (most representative)
2. Trimmed Mean: Removes extreme values before averaging
3. Trust Scoring: Maintains per-client trust scores based on update quality
4. Statistical Anomaly Detection: Detects outlier feature distributions
5. Cosine Similarity Filtering: Filters updates too dissimilar from the median
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from threading import Lock
from scipy.spatial.distance import cdist, cosine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logger = logging.getLogger("federated_central")


# ─── Trust Score Management ──────────────────────────────────────────────────

_trust_scores: Dict[str, float] = {}
_trust_history: Dict[str, List[Dict]] = {}
_trust_lock = Lock()

INITIAL_TRUST = 0.5
MIN_TRUST = 0.0
MAX_TRUST = 1.0
TRUST_DECAY = 0.05
TRUST_REWARD = 0.1
TRUST_PENALTY = 0.2
TRUST_THRESHOLD = 0.3  # Below this, client is considered potentially malicious


def get_trust_score(client_id: str) -> float:
    """Get current trust score for a client."""
    with _trust_lock:
        return _trust_scores.get(client_id, INITIAL_TRUST)


def get_all_trust_scores() -> Dict[str, float]:
    """Get all client trust scores."""
    with _trust_lock:
        return dict(_trust_scores)


def get_trust_history(client_id: str) -> List[Dict]:
    """Get trust score history for a client."""
    with _trust_lock:
        return list(_trust_history.get(client_id, []))


def update_trust_score(client_id: str, delta: float, reason: str) -> float:
    """
    Update trust score for a client.

    Args:
        client_id: Client identifier
        delta: Change in trust score (positive = reward, negative = penalty)
        reason: Reason for the trust update

    Returns:
        New trust score
    """
    with _trust_lock:
        current = _trust_scores.get(client_id, INITIAL_TRUST)
        new_score = max(MIN_TRUST, min(MAX_TRUST, current + delta))
        _trust_scores[client_id] = new_score

        if client_id not in _trust_history:
            _trust_history[client_id] = []
        _trust_history[client_id].append({
            'timestamp': datetime.now().isoformat(),
            'old_score': round(current, 4),
            'new_score': round(new_score, 4),
            'delta': round(delta, 4),
            'reason': reason
        })
        # Keep last 50 entries
        if len(_trust_history[client_id]) > 50:
            _trust_history[client_id] = _trust_history[client_id][-50:]

        return new_score


def initialize_trust(client_id: str) -> None:
    """Initialize trust score for a new client."""
    with _trust_lock:
        if client_id not in _trust_scores:
            _trust_scores[client_id] = INITIAL_TRUST
            _trust_history[client_id] = [{
                'timestamp': datetime.now().isoformat(),
                'old_score': 0.0,
                'new_score': INITIAL_TRUST,
                'delta': INITIAL_TRUST,
                'reason': 'Client registered'
            }]


def reset_trust_scores() -> None:
    """Reset all trust scores (for testing)."""
    with _trust_lock:
        _trust_scores.clear()
        _trust_history.clear()


# ─── Anomaly Detection ───────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Statistical anomaly detection for client updates.

    Detects Byzantine behavior through:
    - Feature distribution analysis (mean/std comparison)
    - Label distribution analysis (class balance check)
    - Model quality validation (accuracy sanity check)
    """

    def __init__(self, z_threshold: float = 3.0, min_accuracy: float = 0.4):
        """
        Args:
            z_threshold: Z-score threshold for outlier detection
            min_accuracy: Minimum acceptable model accuracy
        """
        self.z_threshold = z_threshold
        self.min_accuracy = min_accuracy
        self.history: List[Dict] = []

    def check_features(
        self,
        client_features: np.ndarray,
        all_client_features: List[np.ndarray],
        client_id: str
    ) -> Tuple[bool, str, float]:
        """
        Check if client features are statistically anomalous.

        Args:
            client_features: Features from the client being checked
            all_client_features: Features from all clients
            client_id: Client identifier

        Returns:
            Tuple of (is_anomalous, reason, anomaly_score)
        """
        if len(all_client_features) < 2:
            return False, "Not enough clients for comparison", 0.0

        # Compute mean of each client's feature means
        client_means = [np.mean(f, axis=0) for f in all_client_features]
        client_stds = [np.std(f, axis=0) for f in all_client_features]

        global_mean = np.mean(client_means, axis=0)
        global_std = np.std(client_means, axis=0) + 1e-10

        this_mean = np.mean(client_features, axis=0)
        z_scores = np.abs((this_mean - global_mean) / global_std)
        max_z = float(np.max(z_scores))
        mean_z = float(np.mean(z_scores))

        if max_z > self.z_threshold:
            reason = f"Feature distribution anomaly: max z-score={max_z:.2f} (threshold={self.z_threshold})"
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'client_id': client_id,
                'type': 'feature_anomaly',
                'score': mean_z,
                'reason': reason
            })
            return True, reason, mean_z

        return False, "Features within normal range", mean_z

    def check_labels(
        self,
        client_labels: np.ndarray,
        all_client_labels: List[np.ndarray],
        client_id: str
    ) -> Tuple[bool, str, float]:
        """
        Check if client label distribution is anomalous.

        Args:
            client_labels: Labels from the client being checked
            all_client_labels: Labels from all clients
            client_id: Client identifier

        Returns:
            Tuple of (is_anomalous, reason, anomaly_score)
        """
        if len(all_client_labels) < 2:
            return False, "Not enough clients for comparison", 0.0

        unique_labels = np.unique(np.concatenate(all_client_labels))

        # Compute label distributions for each client
        distributions = []
        for labels in all_client_labels:
            dist = np.array([np.sum(labels == l) / len(labels) for l in unique_labels])
            distributions.append(dist)

        this_dist = np.array([np.sum(client_labels == l) / len(client_labels) for l in unique_labels])
        global_mean_dist = np.mean(distributions, axis=0)
        global_std_dist = np.std(distributions, axis=0) + 1e-10

        z_scores = np.abs((this_dist - global_mean_dist) / global_std_dist)
        max_z = float(np.max(z_scores))

        if max_z > self.z_threshold:
            reason = f"Label distribution anomaly: z-score={max_z:.2f}"
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'client_id': client_id,
                'type': 'label_anomaly',
                'score': max_z,
                'reason': reason
            })
            return True, reason, max_z

        return False, "Label distribution normal", max_z

    def check_model_quality(
        self,
        model,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        client_id: str
    ) -> Tuple[bool, str, float]:
        """
        Check if client model quality is suspiciously low.

        Args:
            model: Trained model to evaluate
            test_features: Test features
            test_labels: Test labels
            client_id: Client identifier

        Returns:
            Tuple of (is_anomalous, reason, accuracy)
        """
        try:
            predictions = model.predict(test_features)
            accuracy = accuracy_score(test_labels, predictions)

            if accuracy < self.min_accuracy:
                reason = f"Model accuracy too low: {accuracy:.2%} (min={self.min_accuracy:.2%})"
                self.history.append({
                    'timestamp': datetime.now().isoformat(),
                    'client_id': client_id,
                    'type': 'model_quality',
                    'score': 1.0 - accuracy,
                    'reason': reason
                })
                return True, reason, accuracy

            return False, f"Model quality acceptable: {accuracy:.2%}", accuracy
        except Exception as e:
            reason = f"Model evaluation failed: {str(e)}"
            return True, reason, 0.0


# ─── Byzantine-Resilient Aggregation Strategies ──────────────────────────────

def krum_selection(
    client_features_list: List[np.ndarray],
    client_labels_list: List[np.ndarray],
    client_ids: List[str],
    num_byzantine: int = 1,
    num_select: int = None
) -> List[int]:
    """
    Krum selection: choose clients whose data is most similar to others.

    The Krum algorithm selects updates that are closest to the majority,
    making it resilient to up to f Byzantine clients out of n total,
    where n >= 2f + 3.

    Args:
        client_features_list: List of feature arrays, one per client
        client_labels_list: List of label arrays, one per client
        client_ids: List of client identifiers
        num_byzantine: Expected maximum number of Byzantine clients
        num_select: Number of clients to select (default: n - num_byzantine)

    Returns:
        List of selected client indices
    """
    n = len(client_features_list)
    if n < 3:
        logger.warning("Krum requires at least 3 clients, using all")
        return list(range(n))

    if num_select is None:
        num_select = max(1, n - num_byzantine)

    # Compute pairwise distances based on feature means
    feature_means = np.array([np.mean(f, axis=0) for f in client_features_list])
    pairwise_dist = cdist(feature_means, feature_means, metric='euclidean')

    # For each client, sum distances to closest (n - num_byzantine - 1) clients
    scores = []
    num_closest = max(1, n - num_byzantine - 1)
    for i in range(n):
        distances = sorted(pairwise_dist[i])
        # Skip self (distance 0), take the num_closest nearest
        score = sum(distances[1:num_closest + 1])
        scores.append(score)

    # Select clients with lowest scores (most central)
    selected_indices = np.argsort(scores)[:num_select].tolist()

    logger.info(f"Krum selected {len(selected_indices)}/{n} clients: "
                f"{[client_ids[i] for i in selected_indices]}")

    # Update trust scores
    for i in range(n):
        if i in selected_indices:
            update_trust_score(client_ids[i], TRUST_REWARD * 0.5,
                             "Selected by Krum (central update)")
        else:
            update_trust_score(client_ids[i], -TRUST_PENALTY * 0.3,
                             "Rejected by Krum (outlier update)")

    return selected_indices


def trimmed_mean_filter(
    client_features_list: List[np.ndarray],
    client_labels_list: List[np.ndarray],
    client_ids: List[str],
    trim_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trimmed mean aggregation: remove extreme feature values before merging.

    For each feature dimension, removes the top and bottom trim_ratio
    fraction of values across clients, then takes the mean.

    Args:
        client_features_list: List of feature arrays
        client_labels_list: List of label arrays
        client_ids: List of client identifiers
        trim_ratio: Fraction of extreme values to trim (each side)

    Returns:
        Tuple of (merged_features, merged_labels) after filtering
    """
    n = len(client_features_list)
    if n < 3:
        # Not enough clients to trim, merge all
        return (np.vstack(client_features_list),
                np.concatenate(client_labels_list))

    # Compute per-client feature means
    client_means = np.array([np.mean(f, axis=0) for f in client_features_list])

    # For each feature dimension, identify outlier clients
    num_trim = max(1, int(n * trim_ratio))
    outlier_clients = set()

    for dim in range(client_means.shape[1]):
        sorted_indices = np.argsort(client_means[:, dim])
        # Mark top and bottom as potential outliers
        for idx in sorted_indices[:num_trim]:
            outlier_clients.add(idx)
        for idx in sorted_indices[-num_trim:]:
            outlier_clients.add(idx)

    # Count how many times each client is flagged
    # Only reject if flagged in majority of dimensions
    threshold = client_means.shape[1] * 0.5
    # Recount properly
    flag_counts = np.zeros(n)
    for dim in range(client_means.shape[1]):
        sorted_indices = np.argsort(client_means[:, dim])
        for idx in sorted_indices[:num_trim]:
            flag_counts[idx] += 1
        for idx in sorted_indices[-num_trim:]:
            flag_counts[idx] += 1

    # Keep clients not excessively flagged
    kept_indices = [i for i in range(n) if flag_counts[i] < threshold]
    if len(kept_indices) < 2:
        # Keep at least 2 clients
        kept_indices = np.argsort(flag_counts)[:max(2, n - num_trim)].tolist()

    logger.info(f"Trimmed mean kept {len(kept_indices)}/{n} clients: "
                f"{[client_ids[i] for i in kept_indices]}")

    # Update trust scores
    for i in range(n):
        if i in kept_indices:
            update_trust_score(client_ids[i], TRUST_REWARD * 0.3,
                             "Passed trimmed mean filter")
        else:
            update_trust_score(client_ids[i], -TRUST_PENALTY * 0.5,
                             "Flagged by trimmed mean filter (extreme values)")

    kept_features = [client_features_list[i] for i in kept_indices]
    kept_labels = [client_labels_list[i] for i in kept_indices]

    return np.vstack(kept_features), np.concatenate(kept_labels)


def cosine_similarity_filter(
    client_features_list: List[np.ndarray],
    client_labels_list: List[np.ndarray],
    client_ids: List[str],
    threshold: float = 0.5
) -> List[int]:
    """
    Filter clients whose feature distribution is too dissimilar from the median.

    Args:
        client_features_list: List of feature arrays
        client_labels_list: List of label arrays
        client_ids: List of client identifiers
        threshold: Minimum cosine similarity to keep (0=orthogonal, 1=identical)

    Returns:
        List of kept client indices
    """
    n = len(client_features_list)
    if n < 3:
        return list(range(n))

    # Compute feature mean vectors
    feature_means = np.array([np.mean(f, axis=0) for f in client_features_list])
    median_features = np.median(feature_means, axis=0)

    # Compute cosine similarity to median
    similarities = []
    for i in range(n):
        sim = 1.0 - cosine(feature_means[i], median_features)
        similarities.append(sim)

    kept = [i for i in range(n) if similarities[i] >= threshold]
    if len(kept) < 2:
        # Keep at least 2 most similar
        kept = np.argsort(similarities)[-max(2, n // 2):].tolist()

    logger.info(f"Cosine filter kept {len(kept)}/{n} clients "
                f"(threshold={threshold})")

    for i in range(n):
        if i in kept:
            update_trust_score(client_ids[i], TRUST_REWARD * 0.2,
                             f"Cosine similarity OK ({similarities[i]:.3f})")
        else:
            update_trust_score(client_ids[i], -TRUST_PENALTY * 0.4,
                             f"Low cosine similarity ({similarities[i]:.3f})")

    return kept


# ─── Main Byzantine-Resilient Aggregation ─────────────────────────────────────

class ByzantineResilientAggregator:
    """
    Complete Byzantine-resilient aggregation pipeline.

    Combines multiple defense mechanisms:
    1. Trust-based filtering (reject low-trust clients)
    2. Statistical anomaly detection (feature + label checks)
    3. Krum selection (geometric median filtering)
    4. Model quality validation
    """

    def __init__(
        self,
        strategy: str = "krum",
        max_byzantine_fraction: float = 0.3,
        trust_threshold: float = TRUST_THRESHOLD,
        anomaly_z_threshold: float = 3.0,
        min_model_accuracy: float = 0.4,
        cosine_threshold: float = 0.5
    ):
        """
        Args:
            strategy: Aggregation strategy ('krum', 'trimmed_mean', 'trust_weighted', 'full')
            max_byzantine_fraction: Maximum expected fraction of Byzantine clients
            trust_threshold: Minimum trust score to participate
            anomaly_z_threshold: Z-score threshold for anomaly detection
            min_model_accuracy: Minimum acceptable model accuracy
            cosine_threshold: Minimum cosine similarity threshold
        """
        self.strategy = strategy
        self.max_byzantine_fraction = max_byzantine_fraction
        self.trust_threshold = trust_threshold
        self.cosine_threshold = cosine_threshold
        self.anomaly_detector = AnomalyDetector(
            z_threshold=anomaly_z_threshold,
            min_accuracy=min_model_accuracy
        )
        self.aggregation_log: List[Dict] = []

    def filter_and_aggregate(
        self,
        client_features_list: List[np.ndarray],
        client_labels_list: List[np.ndarray],
        client_ids: List[str],
        client_models: List = None
    ) -> Dict:
        """
        Run the full Byzantine-resilient aggregation pipeline.

        Args:
            client_features_list: List of feature arrays per client
            client_labels_list: List of label arrays per client
            client_ids: List of client identifiers
            client_models: Optional list of trained models for quality checks

        Returns:
            Dict with:
                - features: Filtered and merged features
                - labels: Filtered and merged labels
                - accepted_clients: List of accepted client IDs
                - rejected_clients: List of (client_id, reason) tuples
                - defense_report: Detailed report of defense actions
        """
        n_original = len(client_ids)
        accepted = list(range(n_original))
        rejected = []
        defense_actions = []
        timestamp = datetime.now().isoformat()

        logger.info(f"Byzantine filtering {n_original} clients using strategy='{self.strategy}'")

        # ── Step 1: Trust-based filtering ──
        if self.strategy in ('trust_weighted', 'full'):
            trust_rejected = []
            for i in list(accepted):
                score = get_trust_score(client_ids[i])
                if score < self.trust_threshold:
                    trust_rejected.append(i)
                    rejected.append((client_ids[i],
                                   f"Low trust score: {score:.3f} < {self.trust_threshold}"))
                    defense_actions.append({
                        'step': 'trust_filter',
                        'client': client_ids[i],
                        'action': 'rejected',
                        'score': score
                    })

            for i in trust_rejected:
                accepted.remove(i)

            if trust_rejected:
                logger.info(f"Trust filter rejected {len(trust_rejected)} clients")

        # ── Step 2: Statistical anomaly detection ──
        if self.strategy in ('full',) and len(accepted) >= 3:
            current_features = [client_features_list[i] for i in accepted]
            current_labels = [client_labels_list[i] for i in accepted]

            anomaly_rejected = []
            for idx, i in enumerate(list(accepted)):
                is_anomalous, reason, score = self.anomaly_detector.check_features(
                    current_features[idx], current_features, client_ids[i]
                )
                if is_anomalous:
                    anomaly_rejected.append(i)
                    rejected.append((client_ids[i], reason))
                    defense_actions.append({
                        'step': 'anomaly_detection',
                        'client': client_ids[i],
                        'action': 'rejected',
                        'reason': reason,
                        'score': score
                    })
                    update_trust_score(client_ids[i], -TRUST_PENALTY,
                                     f"Feature anomaly detected (z={score:.2f})")

            for i in anomaly_rejected:
                if i in accepted:
                    accepted.remove(i)

        # ── Step 3: Strategy-specific filtering ──
        if len(accepted) >= 3:
            current_features = [client_features_list[i] for i in accepted]
            current_labels = [client_labels_list[i] for i in accepted]
            current_ids = [client_ids[i] for i in accepted]

            if self.strategy in ('krum', 'full'):
                num_byzantine = max(1, int(len(accepted) * self.max_byzantine_fraction))
                selected = krum_selection(
                    current_features, current_labels, current_ids,
                    num_byzantine=num_byzantine
                )
                new_accepted = [accepted[i] for i in selected]
                for i in accepted:
                    if i not in new_accepted:
                        rejected.append((client_ids[i], "Rejected by Krum selection"))
                        defense_actions.append({
                            'step': 'krum_selection',
                            'client': client_ids[i],
                            'action': 'rejected'
                        })
                accepted = new_accepted

            elif self.strategy == 'trimmed_mean':
                # Trimmed mean returns merged data directly
                merged_features, merged_labels = trimmed_mean_filter(
                    current_features, current_labels, current_ids,
                    trim_ratio=self.max_byzantine_fraction
                )
                defense_actions.append({
                    'step': 'trimmed_mean',
                    'action': 'applied',
                    'original_clients': len(current_ids)
                })
                # Return early with merged data
                report = self._build_report(
                    n_original, [current_ids[i] for i in range(len(current_ids))],
                    rejected, defense_actions, timestamp
                )
                return {
                    'features': merged_features,
                    'labels': merged_labels,
                    'accepted_clients': current_ids,
                    'rejected_clients': rejected,
                    'defense_report': report
                }

        # ── Step 4: Cosine similarity check ──
        if self.strategy == 'full' and len(accepted) >= 3:
            current_features = [client_features_list[i] for i in accepted]
            current_labels = [client_labels_list[i] for i in accepted]
            current_ids = [client_ids[i] for i in accepted]

            kept = cosine_similarity_filter(
                current_features, current_labels, current_ids,
                threshold=self.cosine_threshold
            )
            new_accepted = [accepted[i] for i in kept]
            for i in accepted:
                if i not in new_accepted:
                    rejected.append((client_ids[i], "Low cosine similarity"))
            accepted = new_accepted

        # Ensure at least 1 client remains
        if not accepted:
            logger.warning("All clients rejected! Falling back to most trusted client")
            scores = [(i, get_trust_score(client_ids[i])) for i in range(n_original)]
            best = max(scores, key=lambda x: x[1])
            accepted = [best[0]]
            defense_actions.append({
                'step': 'fallback',
                'action': 'selected_most_trusted',
                'client': client_ids[best[0]]
            })

        # ── Merge accepted data ──
        accepted_features = [client_features_list[i] for i in accepted]
        accepted_labels = [client_labels_list[i] for i in accepted]
        accepted_ids = [client_ids[i] for i in accepted]

        merged_features = np.vstack(accepted_features)
        merged_labels = np.concatenate(accepted_labels)

        # Reward accepted clients
        for cid in accepted_ids:
            update_trust_score(cid, TRUST_REWARD, "Passed all Byzantine checks")

        report = self._build_report(
            n_original, accepted_ids, rejected, defense_actions, timestamp
        )

        logger.info(f"Byzantine filtering complete: {len(accepted_ids)}/{n_original} "
                    f"clients accepted, {len(rejected)} rejected")

        return {
            'features': merged_features,
            'labels': merged_labels,
            'accepted_clients': accepted_ids,
            'rejected_clients': rejected,
            'defense_report': report
        }

    def _build_report(
        self,
        n_original: int,
        accepted_ids: List[str],
        rejected: List[Tuple[str, str]],
        defense_actions: List[Dict],
        timestamp: str
    ) -> Dict:
        """Build a comprehensive defense report."""
        report = {
            'timestamp': timestamp,
            'strategy': self.strategy,
            'total_clients': n_original,
            'accepted_count': len(accepted_ids),
            'rejected_count': len(rejected),
            'accepted_clients': accepted_ids,
            'rejected_clients': [{'client_id': r[0], 'reason': r[1]} for r in rejected],
            'defense_actions': defense_actions,
            'trust_scores': {cid: get_trust_score(cid) for cid in
                           accepted_ids + [r[0] for r in rejected]},
            'anomaly_history': self.anomaly_detector.history[-10:]
        }

        self.aggregation_log.append(report)
        if len(self.aggregation_log) > 50:
            self.aggregation_log = self.aggregation_log[-50:]

        return report

    def get_aggregation_log(self) -> List[Dict]:
        """Get full aggregation log."""
        return list(self.aggregation_log)


# ─── Global Instance ──────────────────────────────────────────────────────────

_aggregator = ByzantineResilientAggregator(strategy='full')


def get_byzantine_aggregator() -> ByzantineResilientAggregator:
    """Get global Byzantine aggregator instance."""
    return _aggregator


def set_byzantine_strategy(strategy: str) -> None:
    """Set the Byzantine defense strategy."""
    global _aggregator
    _aggregator = ByzantineResilientAggregator(strategy=strategy)
    logger.info(f"Byzantine strategy set to: {strategy}")


# ─── Adaptive Threshold ──────────────────────────────────────────────────────

_adaptive_state = {
    'round': 0,
    'initial_threshold': 0.2,
    'current_threshold': 0.2,
    'max_threshold': 0.5,
    'tighten_rate': 0.03,
    'false_positives': 0,
    'false_negatives': 0,
    'history': [],
}
_adaptive_lock = Lock()


def get_adaptive_threshold() -> float:
    """Get the current adaptive trust threshold."""
    with _adaptive_lock:
        return _adaptive_state['current_threshold']


def get_adaptive_state() -> Dict:
    """Get full adaptive threshold state."""
    with _adaptive_lock:
        return dict(_adaptive_state)


def advance_adaptive_threshold(accepted: int, rejected: int, round_accuracy: float = 0.0):
    """
    Tighten threshold after each round. Starts permissive, becomes stricter.
    Called after each aggregation round.
    """
    with _adaptive_lock:
        _adaptive_state['round'] += 1
        r = _adaptive_state['round']
        old = _adaptive_state['current_threshold']

        # Tighten: move toward max_threshold over rounds
        new = min(
            _adaptive_state['max_threshold'],
            _adaptive_state['initial_threshold'] + _adaptive_state['tighten_rate'] * r
        )
        _adaptive_state['current_threshold'] = new

        # Track detection quality
        _adaptive_state['history'].append({
            'round': r, 'threshold': new,
            'accepted': accepted, 'rejected': rejected,
            'accuracy': round_accuracy,
        })
        if len(_adaptive_state['history']) > 50:
            _adaptive_state['history'] = _adaptive_state['history'][-50:]

        logger.info(f"Adaptive threshold: {old:.3f} → {new:.3f} (round {r})")
        return new


def record_byzantine_detection(true_positive: bool):
    """Record whether a detection was correct (for FP/FN tracking)."""
    with _adaptive_lock:
        if not true_positive:
            _adaptive_state['false_positives'] += 1
        # true_positive doesn't increment FN; FN would require knowing about missed attacks
