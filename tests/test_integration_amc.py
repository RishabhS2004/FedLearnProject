"""
Integration tests for AMC Dashboard Enhancement

Tests the complete workflow including:
- Dataset partitioning
- Feature extraction
- Model training and timing
- Aggregation for KNN and Decision Tree

"""

import pytest
import tempfile
import os
import shutil
import numpy as np
import pickle
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


from data.partition_dataset import (
    load_radioml_pkl_dataset,
    filter_analog_modulations,
    partition_dataset,
    save_partition,
    validate_partitions
)
from client.feature_extract import (
    extract_analog_features,
    extract_features_from_iq,
    process_dataset,
    compute_instantaneous_amplitude,
    compute_instantaneous_frequency,
    compute_statistical_features
)
from central.aggregator import (
    aggregate_knn_models,
    evaluate_global_model
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def synthetic_dataset(temp_dir):
    """Create synthetic RadioML dataset with analog modulations."""
    dataset = {}
    
    
    modulations = ['AM-DSB', 'AM-SSB', 'WBFM']
    snrs = [-10, 0, 10, 18]
    samples_per_key = 100
    
    for mod in modulations:
        for snr in snrs:
           
            samples = np.random.randn(samples_per_key, 2, 128).astype(np.float32)
            dataset[(mod, snr)] = samples
    
    
    dataset_path = os.path.join(temp_dir, 'test_dataset.pkl')
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    return dataset, dataset_path




@pytest.mark.parametrize("num_clients", [2, 3, 5])
def test_partition_creation_with_different_client_counts(synthetic_dataset, num_clients):
    """Test creating partitions with 2, 3, and 5 clients."""
    dataset, dataset_path = synthetic_dataset
    
    
    loaded_dataset = load_radioml_pkl_dataset(dataset_path)
    filtered_dataset = filter_analog_modulations(loaded_dataset)
    
    partitions = partition_dataset(filtered_dataset, num_clients)
    
    
    assert len(partitions) == num_clients
    
   
    for i, partition in enumerate(partitions):
        assert len(partition) > 0, f"Partition {i} is empty"
        
        
        total_samples = sum(samples.shape[0] for samples in partition.values())
        assert total_samples > 0, f"Partition {i} has no samples"


def test_partition_balance(synthetic_dataset):
    """Test that partitions are balanced across clients."""
    dataset, dataset_path = synthetic_dataset
    
    loaded_dataset = load_radioml_pkl_dataset(dataset_path)
    filtered_dataset = filter_analog_modulations(loaded_dataset)
    
    num_clients = 3
    partitions = partition_dataset(filtered_dataset, num_clients, balance_classes=True)
    
    
    sample_counts = []
    for partition in partitions:
        total_samples = sum(samples.shape[0] for samples in partition.values())
        sample_counts.append(total_samples)
    
    
    max_count = max(sample_counts)
    min_count = min(sample_counts)
    max_diff = max_count - min_count
    
    
    assert max_diff <= 15, "Partitions are not balanced"


def test_partition_non_overlap(synthetic_dataset):
    """Test that partitions are non-overlapping."""
    dataset, dataset_path = synthetic_dataset
    
    loaded_dataset = load_radioml_pkl_dataset(dataset_path)
    filtered_dataset = filter_analog_modulations(loaded_dataset)
    
    num_clients = 3
    partitions = partition_dataset(filtered_dataset, num_clients, random_seed=42)
    
   
    for key in filtered_dataset.keys():
        
        partition_samples = []
        for partition in partitions:
            if key in partition:
                partition_samples.append(partition[key])
        
        
        total_partition_samples = sum(s.shape[0] for s in partition_samples)
        original_samples = filtered_dataset[key].shape[0]
        
        assert total_partition_samples == original_samples, f"Sample count mismatch for key {key}"


def test_partition_loading(synthetic_dataset, temp_dir):
    """Test loading partitions from saved files."""
    dataset, dataset_path = synthetic_dataset
    
    loaded_dataset = load_radioml_pkl_dataset(dataset_path)
    filtered_dataset = filter_analog_modulations(loaded_dataset)
    
    num_clients = 3
    partitions = partition_dataset(filtered_dataset, num_clients)
    
    
    partition_dir = os.path.join(temp_dir, 'partitions')
    for i, partition in enumerate(partitions):
        partition_path = os.path.join(partition_dir, f'client_{i}.pkl')
        save_partition(partition, partition_path)
    
    
    for i in range(num_clients):
        partition_path = os.path.join(partition_dir, f'client_{i}.pkl')
        assert os.path.exists(partition_path), f"Partition file {i} not found"
        
        
        with open(partition_path, 'rb') as f:
            loaded_partition = pickle.load(f)
        
        
        assert isinstance(loaded_partition, dict)
        assert len(loaded_partition) > 0
        
        
        for key, samples in loaded_partition.items():
            assert samples.shape[1:] == (2, 128)


def test_partition_validation(synthetic_dataset):
    """Test partition validation function."""
    dataset, dataset_path = synthetic_dataset
    
    loaded_dataset = load_radioml_pkl_dataset(dataset_path)
    filtered_dataset = filter_analog_modulations(loaded_dataset)
    
    num_clients = 3
    partitions = partition_dataset(filtered_dataset, num_clients)
    
    
    validate_partitions(partitions)
    
    
    invalid_partitions = [{}]
    with pytest.raises(ValueError):
        validate_partitions(invalid_partitions)




@pytest.fixture
def signal_types():
    """Create different signal types for testing."""
    np.random.seed(42)
    signals = {}
    
    
    signals['random'] = np.random.randn(128) + 1j * np.random.randn(128)
    signals['constant'] = np.ones(128, dtype=complex)
    signals['zero'] = np.zeros(128, dtype=complex)
    t = np.linspace(0, 1, 128)
    signals['high_snr'] = np.exp(1j * 2 * np.pi * 5 * t)
    signals['low_snr'] = np.exp(1j * 2 * np.pi * 5 * t) + 2 * (np.random.randn(128) + 1j * np.random.randn(128))
    
    return signals


def test_8d_feature_vector_generation(signal_types):
    """Test that 8D feature vectors are generated correctly."""
    for signal_type, signal in signal_types.items():
        features = extract_analog_features(signal, fs=128)
        

        assert features.shape == (8,), f"Wrong shape for {signal_type}"
        
    
        assert features.dtype == np.float32, f"Wrong dtype for {signal_type}"
        

        assert not np.any(np.isnan(features)), f"NaN in features for {signal_type}"
        assert not np.any(np.isinf(features)), f"Inf in features for {signal_type}"


@pytest.mark.parametrize("snr_db", [-10, -5, 0, 5, 10, 15, 20])
def test_feature_extraction_with_various_snrs(snr_db):
    """Test feature extraction with signals at different SNRs."""
    t = np.linspace(0, 1, 128)
    clean_signal = np.exp(1j * 2 * np.pi * 5 * t)
    
  
    snr_linear = 10 ** (snr_db / 10)
    noise_power = 1 / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(128) + 1j * np.random.randn(128))
    noisy_signal = clean_signal + noise
    
    
    features = extract_analog_features(noisy_signal, fs=128)
    
    
    assert features.shape == (8,)
    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    
    constant_signal = np.ones(128, dtype=complex)
    features_constant = extract_analog_features(constant_signal, fs=128)
    
    assert not np.any(np.isnan(features_constant)), "NaN in constant signal features"
    assert not np.any(np.isinf(features_constant)), "Inf in constant signal features"
    
    
    zero_signal = np.zeros(128, dtype=complex)
    features_zero = extract_analog_features(zero_signal, fs=128)
    
    assert not np.any(np.isnan(features_zero)), "NaN in zero signal features"
    assert not np.any(np.isinf(features_zero)), "Inf in zero signal features"
    
    
    tiny_signal = 1e-10 * np.random.randn(128) + 1j * 1e-10 * np.random.randn(128)
    features_tiny = extract_analog_features(tiny_signal, fs=128)
    
    assert not np.any(np.isnan(features_tiny)), "NaN in tiny signal features"
    assert not np.any(np.isinf(features_tiny)), "Inf in tiny signal features"


def test_process_dataset_batch():
    """Test processing a batch of samples."""
  
    n_samples = 50
    iq_samples = np.random.randn(n_samples, 2, 128).astype(np.float32)
    labels = np.random.randint(0, 2, size=n_samples)
    
    
    features, output_labels = process_dataset(iq_samples, labels, verbose=False, use_analog_features=True)
    
  
    assert features.shape == (n_samples, 8)
    assert output_labels.shape == (n_samples,)
    
 
    np.testing.assert_array_equal(output_labels, labels)
    
   
    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))




@pytest.fixture
def training_data():
    """Create synthetic feature data for training."""
    np.random.seed(42)
    
    n_samples = 200
    n_features = 8
    n_classes = 2  # AM, FM
    
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = np.random.randint(0, n_classes, size=n_samples)
    
    X_test = np.random.randn(50, n_features).astype(np.float32)
    y_test = np.random.randint(0, n_classes, size=50)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


def test_knn_training(training_data):
    """Test KNN model training."""
    knn = KNeighborsClassifier(n_neighbors=5)
    
  
    start_time = time.time()
    knn.fit(training_data['X_train'], training_data['y_train'])
    training_time = time.time() - start_time
    
    
    assert hasattr(knn, 'classes_')
    
    
    assert training_time >= 0
    assert training_time < 10  #


def test_dt_training(training_data):
    """Test Decision Tree model training."""
    dt = DecisionTreeClassifier(random_state=42)
    
    
    start_time = time.time()
    dt.fit(training_data['X_train'], training_data['y_train'])
    training_time = time.time() - start_time
    
    
    assert hasattr(dt, 'tree_')
    
    
    assert training_time >= 0
    assert training_time < 10


def test_knn_inference_timing(training_data):
    """Test KNN inference timing measurement."""
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(training_data['X_train'], training_data['y_train'])
    
    
    start_time = time.time()
    predictions = knn.predict(training_data['X_test'])
    total_inference_time = time.time() - start_time
    
    
    per_sample_time_ms = (total_inference_time / len(training_data['X_test'])) * 1000
    
    
    assert len(predictions) == len(training_data['X_test'])
    
    
    assert per_sample_time_ms >= 0
    assert per_sample_time_ms < 100


def test_model_serialization(training_data, temp_dir):
    """Test model serialization and loading."""
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(training_data['X_train'], training_data['y_train'])
    
    
    model_path = os.path.join(temp_dir, 'knn_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(knn, f)
    
    
    assert os.path.exists(model_path)
    
    with open(model_path, 'rb') as f:
        loaded_knn = pickle.load(f)
    
    
    original_predictions = knn.predict(training_data['X_test'])
    loaded_predictions = loaded_knn.predict(training_data['X_test'])
    
    np.testing.assert_array_equal(original_predictions, loaded_predictions)




def test_knn_aggregation(temp_dir):
    """Test KNN model aggregation from multiple clients."""
    np.random.seed(42)
    num_clients = 3
    client_models_info = []
    
    for i in range(num_clients):
        n_samples = 100 + i * 50
        X = np.random.randn(n_samples, 8).astype(np.float32)
        y = np.random.randint(0, 2, size=n_samples)
        
       
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)
        
        
        model_path = os.path.join(temp_dir, f'client_{i}_knn.pkl')
        features_path = os.path.join(temp_dir, f'client_{i}_features.pkl')
        labels_path = os.path.join(temp_dir, f'client_{i}_labels.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(knn, f)
        with open(features_path, 'wb') as f:
            pickle.dump(X, f)
        with open(labels_path, 'wb') as f:
            pickle.dump(y, f)
        
        client_models_info.append({
            'client_id': f'client_{i}',
            'model_path': model_path,
            'features_path': features_path,
            'labels_path': labels_path,
            'n_samples': len(X)
        })
    
    
    result = aggregate_knn_models(client_models_info, n_neighbors=5)
    
    
    assert 'global_model' in result
    assert 'total_samples' in result
    assert 'num_clients' in result
    assert result['num_clients'] == num_clients
    
   
    global_knn = result['global_model']
    X_test = np.random.randn(10, 8).astype(np.float32)
    predictions = global_knn.predict(X_test)
    assert len(predictions) == len(X_test)


def test_global_model_accuracy_evaluation(temp_dir):
    """Test global model accuracy evaluation."""
    np.random.seed(42)
    
    
    X_test = np.random.randn(100, 8).astype(np.float32)
    y_test = np.random.randint(0, 2, size=100)
    
    
    X_train = np.random.randn(200, 8).astype(np.float32)
    y_train = np.random.randint(0, 2, size=200)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
   
    eval_result = evaluate_global_model(knn, X_test, y_test)
    

    assert 'accuracy' in eval_result
    assert 'confusion_matrix' in eval_result
    assert 'n_samples' in eval_result
    
    assert 0.0 <= eval_result['accuracy'] <= 1.0
    assert eval_result['n_samples'] == len(X_test)
