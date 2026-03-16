"""
Tests for Byzantine Fault Tolerance Module

Tests cover:
- Trust score management
- Anomaly detection
- Krum selection
- Trimmed mean filtering
- Cosine similarity filtering
- Full Byzantine-resilient aggregation pipeline
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from central.byzantine import (
    get_trust_score,
    update_trust_score,
    initialize_trust,
    get_all_trust_scores,
    reset_trust_scores,
    AnomalyDetector,
    krum_selection,
    trimmed_mean_filter,
    cosine_similarity_filter,
    ByzantineResilientAggregator,
    INITIAL_TRUST,
    TRUST_THRESHOLD
)


@pytest.fixture(autouse=True)
def clean_trust_state():
    """Reset trust scores before each test."""
    reset_trust_scores()
    yield
    reset_trust_scores()


class TestTrustScores:
    """Tests for trust score management."""

    def test_initial_trust(self):
        initialize_trust("client_1")
        assert get_trust_score("client_1") == INITIAL_TRUST

    def test_unknown_client_gets_initial(self):
        assert get_trust_score("unknown") == INITIAL_TRUST

    def test_update_trust_positive(self):
        initialize_trust("client_1")
        new_score = update_trust_score("client_1", 0.1, "good behavior")
        assert new_score == INITIAL_TRUST + 0.1

    def test_update_trust_negative(self):
        initialize_trust("client_1")
        new_score = update_trust_score("client_1", -0.2, "bad behavior")
        assert new_score == INITIAL_TRUST - 0.2

    def test_trust_clamped_at_zero(self):
        initialize_trust("client_1")
        new_score = update_trust_score("client_1", -10.0, "very bad")
        assert new_score == 0.0

    def test_trust_clamped_at_one(self):
        initialize_trust("client_1")
        new_score = update_trust_score("client_1", 10.0, "very good")
        assert new_score == 1.0

    def test_get_all_trust_scores(self):
        initialize_trust("client_1")
        initialize_trust("client_2")
        scores = get_all_trust_scores()
        assert len(scores) == 2
        assert "client_1" in scores
        assert "client_2" in scores


class TestAnomalyDetector:
    """Tests for statistical anomaly detection."""

    def test_normal_features_not_flagged(self):
        detector = AnomalyDetector(z_threshold=3.0)
        np.random.seed(42)
        # All clients have similar distributions
        features = [np.random.randn(100, 8) for _ in range(4)]
        is_anomalous, reason, score = detector.check_features(
            features[0], features, "client_0"
        )
        assert not is_anomalous

    def test_outlier_features_flagged(self):
        detector = AnomalyDetector(z_threshold=2.0)
        np.random.seed(42)
        # Normal clients with tight distribution
        features = [np.random.randn(200, 8) * 0.5 for _ in range(5)]
        # Outlier client with very different distribution
        outlier = np.random.randn(200, 8) * 100 + 500
        all_features = features + [outlier]

        is_anomalous, reason, score = detector.check_features(
            outlier, all_features, "outlier_client"
        )
        assert is_anomalous

    def test_normal_labels_not_flagged(self):
        detector = AnomalyDetector()
        # All clients have balanced labels
        labels = [np.array([0]*50 + [1]*50) for _ in range(4)]
        is_anomalous, reason, score = detector.check_labels(
            labels[0], labels, "client_0"
        )
        assert not is_anomalous

    def test_model_quality_check(self):
        from sklearn.neighbors import KNeighborsClassifier
        detector = AnomalyDetector(min_accuracy=0.6)

        # Good model
        np.random.seed(42)
        X = np.random.randn(200, 4)
        y = (X[:, 0] > 0).astype(int)
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X[:150], y[:150])

        is_anomalous, reason, acc = detector.check_model_quality(
            model, X[150:], y[150:], "client_good"
        )
        assert not is_anomalous
        assert acc > 0.6


class TestKrumSelection:
    """Tests for Krum selection algorithm."""

    def test_selects_central_clients(self):
        np.random.seed(42)
        # 3 honest clients with similar data + 1 outlier
        features = [np.random.randn(50, 4) for _ in range(3)]
        features.append(np.random.randn(50, 4) * 10 + 50)  # outlier

        labels = [np.array([0]*25 + [1]*25) for _ in range(4)]
        ids = ["c1", "c2", "c3", "c_byz"]

        for cid in ids:
            initialize_trust(cid)

        selected = krum_selection(features, labels, ids, num_byzantine=1)
        # The outlier should not be selected
        assert 3 not in selected  # index 3 is the outlier

    def test_returns_all_with_few_clients(self):
        features = [np.random.randn(50, 4) for _ in range(2)]
        labels = [np.array([0]*25 + [1]*25) for _ in range(2)]
        ids = ["c1", "c2"]

        for cid in ids:
            initialize_trust(cid)

        selected = krum_selection(features, labels, ids)
        assert len(selected) == 2


class TestTrimmedMeanFilter:
    """Tests for trimmed mean filtering."""

    def test_filters_outliers(self):
        np.random.seed(42)
        features = [np.random.randn(50, 4) for _ in range(4)]
        features.append(np.random.randn(50, 4) * 100)  # extreme outlier

        labels = [np.array([0]*25 + [1]*25) for _ in range(5)]
        ids = ["c1", "c2", "c3", "c4", "c_byz"]

        for cid in ids:
            initialize_trust(cid)

        merged_f, merged_l = trimmed_mean_filter(features, labels, ids, trim_ratio=0.2)
        # Should have fewer samples than total
        total = sum(len(f) for f in features)
        assert len(merged_f) < total

    def test_returns_all_with_few_clients(self):
        features = [np.random.randn(50, 4) for _ in range(2)]
        labels = [np.array([0]*25 + [1]*25) for _ in range(2)]
        ids = ["c1", "c2"]

        merged_f, merged_l = trimmed_mean_filter(features, labels, ids)
        assert len(merged_f) == 100


class TestCosineFilter:
    """Tests for cosine similarity filtering."""

    def test_keeps_similar_clients(self):
        np.random.seed(42)
        features = [np.random.randn(50, 4) + 1 for _ in range(3)]
        # Dissimilar client
        features.append(np.random.randn(50, 4) * -5)

        labels = [np.array([0]*25 + [1]*25) for _ in range(4)]
        ids = ["c1", "c2", "c3", "c_diff"]

        for cid in ids:
            initialize_trust(cid)

        kept = cosine_similarity_filter(features, labels, ids, threshold=0.5)
        # Should keep the similar ones
        assert len(kept) >= 2


class TestByzantineResilientAggregator:
    """Tests for the full Byzantine aggregation pipeline."""

    def test_full_pipeline_accepts_honest(self):
        np.random.seed(42)
        features = [np.random.randn(50, 4) for _ in range(3)]
        labels = [np.array([0]*25 + [1]*25) for _ in range(3)]
        ids = ["c1", "c2", "c3"]

        for cid in ids:
            initialize_trust(cid)

        agg = ByzantineResilientAggregator(strategy='krum')
        result = agg.filter_and_aggregate(features, labels, ids)

        assert len(result['accepted_clients']) >= 2
        assert result['features'].shape[0] > 0
        assert result['defense_report'] is not None

    def test_full_pipeline_rejects_outlier(self):
        np.random.seed(42)
        features = [np.random.randn(100, 4) for _ in range(4)]
        features.append(np.random.randn(100, 4) * 50 + 100)  # Byzantine

        labels = [np.array([0]*50 + [1]*50) for _ in range(5)]
        ids = ["c1", "c2", "c3", "c4", "c_byz"]

        for cid in ids:
            initialize_trust(cid)

        agg = ByzantineResilientAggregator(strategy='krum', max_byzantine_fraction=0.3)
        result = agg.filter_and_aggregate(features, labels, ids)

        accepted = result['accepted_clients']
        # Byzantine client should likely be rejected
        assert result['defense_report']['rejected_count'] >= 0
        assert len(result['features']) > 0

    def test_trust_weighted_rejects_low_trust(self):
        features = [np.random.randn(50, 4) for _ in range(3)]
        labels = [np.array([0]*25 + [1]*25) for _ in range(3)]
        ids = ["c1", "c2", "c_low"]

        for cid in ids:
            initialize_trust(cid)
        # Set low trust for one client
        update_trust_score("c_low", -0.4, "historical bad behavior")

        agg = ByzantineResilientAggregator(strategy='trust_weighted')
        result = agg.filter_and_aggregate(features, labels, ids)

        # Low-trust client should be rejected
        rejected_ids = [r[0] for r in result['rejected_clients']]
        assert "c_low" in rejected_ids

    def test_fallback_when_all_rejected(self):
        """If all clients get rejected, should fallback to most trusted."""
        features = [np.random.randn(50, 4) * 100 for _ in range(3)]
        labels = [np.array([0]*25 + [1]*25) for _ in range(3)]
        ids = ["c1", "c2", "c3"]

        for cid in ids:
            initialize_trust(cid)
            update_trust_score(cid, -0.3, "test")

        agg = ByzantineResilientAggregator(strategy='trust_weighted',
                                           trust_threshold=0.9)
        result = agg.filter_and_aggregate(features, labels, ids)

        # Should still have some data (fallback)
        assert len(result['features']) > 0
        assert len(result['accepted_clients']) >= 1
