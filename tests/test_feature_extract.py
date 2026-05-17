"""
Unit tests for feature extraction from I/Q samples.

Tests feature extraction functionality including statistical,
frequency domain, and time domain features.
"""

import pytest
import numpy as np
from client.feature_extract import (
    extract_features_from_iq,
    process_dataset,
    normalize_features
)


@pytest.fixture
def sample_iq():
    """Create a simple synthetic I/Q sample."""
    np.random.seed(42)
    return np.random.randn(2, 128).astype(np.float32)


@pytest.fixture
def batch_samples():
    """Create a batch of samples for dataset processing."""
    np.random.seed(42)
    samples = np.random.randn(50, 2, 128).astype(np.float32)
    labels = np.random.randint(0, 11, size=50)
    return samples, labels


def test_extract_features_from_iq_shape(sample_iq):
    """Test that feature extraction returns correct shape."""
    features = extract_features_from_iq(sample_iq)
    
    assert features.shape == (16,)
    assert features.dtype == np.float32


def test_extract_features_from_iq_invalid_shape():
    """Test that invalid input shape raises ValueError."""
    invalid_sample = np.random.randn(3, 128) 
    
    with pytest.raises(ValueError):
        extract_features_from_iq(invalid_sample)
    
    invalid_sample2 = np.random.randn(2, 64)  
    
    with pytest.raises(ValueError):
        extract_features_from_iq(invalid_sample2)


def test_extract_features_from_iq_no_nan(sample_iq):
    """Test that extracted features contain no NaN values."""
    features = extract_features_from_iq(sample_iq)
    
    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))


def test_extract_features_from_iq_deterministic(sample_iq):
    """Test that same input produces same features."""
    features1 = extract_features_from_iq(sample_iq)
    features2 = extract_features_from_iq(sample_iq)
    
    np.testing.assert_array_equal(features1, features2)


def test_extract_features_statistical():
    """Test that statistical features are computed correctly."""
    
    i_channel = np.array([1.0, 2.0, 3.0, 4.0, 5.0] + [0.0] * 123)
    q_channel = np.array([2.0, 4.0, 6.0, 8.0, 10.0] + [0.0] * 123)
    sample = np.vstack([i_channel, q_channel]).astype(np.float32)
    
    features = extract_features_from_iq(sample)
    
    
    assert isinstance(features[0], (float, np.floating))
    
    
    for i in range(10):  
        assert np.isfinite(features[i])


def test_extract_features_frequency_domain(sample_iq):
    """Test that frequency domain features are computed."""
    features = extract_features_from_iq(sample_iq)
    
    
    assert features[10] > 0
    
    
    assert features[11] >= 0
    assert features[11] <= 0.5 
    
    
    assert np.isfinite(features[12])
    assert np.isfinite(features[13])


def test_extract_features_time_domain(sample_iq):
    """Test that time domain features are computed."""
    features = extract_features_from_iq(sample_iq)
    
    
    assert features[14] >= 0
    assert features[14] <= 1
    
    
    assert features[15] > 0


def test_extract_features_zero_signal():
    """Test feature extraction with zero signal."""
    zero_sample = np.zeros((2, 128), dtype=np.float32)
    features = extract_features_from_iq(zero_sample)
    
    
    assert features.shape == (16,)
    assert not np.any(np.isnan(features))


def test_process_dataset_shape(batch_samples):
    """Test that dataset processing returns correct shapes."""
    samples, labels = batch_samples
    features, output_labels = process_dataset(
        samples, 
        labels, 
        verbose=False
    )
    
    assert features.shape == (50, 16)
    assert output_labels.shape == (50,)
    np.testing.assert_array_equal(output_labels, labels)


def test_process_dataset_verbose(batch_samples):
    """Test that verbose mode doesn't cause errors."""
    samples, labels = batch_samples
    
    features, output_labels = process_dataset(
        samples[:10], 
        labels[:10], 
        verbose=True
    )
    
    assert features.shape == (10, 16)


def test_process_dataset_handles_errors(batch_samples):
    """Test that dataset processing handles individual sample errors."""
    samples, labels = batch_samples
    
    invalid_batch = samples.copy()
    invalid_batch[5] = np.nan  
    
    features, output_labels = process_dataset(
        invalid_batch, 
        labels, 
        verbose=False
    )
    
    
    assert features.shape == (50, 16)


def test_normalize_features():
    """Test feature normalization."""
    features = np.random.randn(100, 16).astype(np.float32)
    
    normalized, center, scale = normalize_features(features)
    
    
    assert normalized.shape == features.shape
    assert center.shape == (16,)
    assert scale.shape == (16,)


def test_normalize_features_zero_std():
    """Test normalization with zero standard deviation."""
    
    features = np.random.randn(100, 16).astype(np.float32)
    features[:, 5] = 1.0  
    
    normalized, center, scale = normalize_features(features)
    
    
    assert not np.any(np.isnan(normalized))
    assert not np.any(np.isinf(normalized))
    
    
    assert np.allclose(normalized[:, 5], 0.0)


def test_feature_extraction_consistency(batch_samples):
    """Test that features are consistent across multiple extractions."""
    samples, labels = batch_samples
    features1, _ = process_dataset(
        samples, 
        labels, 
        verbose=False
    )
    features2, _ = process_dataset(
        samples, 
        labels, 
        verbose=False
    )
    
    np.testing.assert_array_equal(features1, features2)
