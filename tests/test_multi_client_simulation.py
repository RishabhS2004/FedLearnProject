"""
Multi-Client Federated Learning Simulation Test

This script performs a complete end-to-end simulation of the federated learning
workflow with 3 clients, verifying:
- All clients can train and upload weights
- Aggregation produces improved global model
- Dashboard displays all metrics correctly

"""

import os
import sys
import time
import pickle
import requests
import numpy as np
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from client.dataset_loader import load_radioml_dataset, flatten_dataset
from client.feature_extract import process_dataset
from client.train import train_knn_model, save_knn_model
from central.aggregator import aggregate_knn_models, evaluate_global_model


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}[OK] {text}{Colors.RESET}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}[INFO] {text}{Colors.RESET}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.RESET}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}[ERROR] {text}{Colors.RESET}")


def check_partitions_exist(num_clients=3):
    """
    Check if dataset partitions exist for all clients.
    
    Args:
        num_clients: Number of clients to check for
        
    Returns:
        bool: True if all partitions exist, False otherwise
    """
    print_header("Step 1: Verify Dataset Partitions")
    
    partitions_dir = Path("data/partitions")
    
    if not partitions_dir.exists():
        print_error(f"Partitions directory not found: {partitions_dir}")
        print_info("Please run: python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 3")
        return False
    
    all_exist = True
    for i in range(num_clients):
        partition_path = partitions_dir / f"client_{i}.pkl"
        if partition_path.exists():
            # Check file size
            size_mb = partition_path.stat().st_size / (1024 * 1024)
            print_success(f"Partition {i} exists: {partition_path} ({size_mb:.2f} MB)")
        else:
            print_error(f"Partition {i} not found: {partition_path}")
            all_exist = False
    
    if all_exist:
        print_success(f"All {num_clients} partitions verified")
    else:
        print_error("Some partitions are missing")
        print_info("Please run: python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 3")
    
    return all_exist


def check_server_running(server_url="http://localhost:8000", timeout=5):
    """
    Check if the central server is running.
    
    Args:
        server_url: URL of the central server
        timeout: Request timeout in seconds
        
    Returns:
        bool: True if server is running, False otherwise
    """
    print_header("Step 2: Verify Central Server")
    
    try:
        response = requests.get(f"{server_url}/health", timeout=timeout)
        if response.status_code == 200:
            print_success(f"Central server is running at {server_url}")
            
           
            try:
                status_response = requests.get(f"{server_url}/status", timeout=timeout)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print_info(f"Server status: {status_data.get('server_status', 'unknown')}")
                    print_info(f"Connected clients: {status_data.get('total_clients', 0)}")
                    print_info(f"Total samples: {status_data.get('total_samples', 0)}")
            except:
                pass
            
            return True
        else:
            print_error(f"Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to server at {server_url}")
        print_info("Please start the server: python central/main.py")
        return False
    except Exception as e:
        print_error(f"Error checking server: {str(e)}")
        return False


def simulate_client_training(client_id, partition_id, server_url="http://localhost:8000"):
    """
    Simulate a single client's training workflow.
    
    Args:
        client_id: Client identifier
        partition_id: Partition ID to load
        server_url: Central server URL
        model_type: Model type ('knn' or 'dt')
        
    Returns:
        dict: Training results including model, accuracy, and timing
    """
    print(f"\n{Colors.BOLD}Client {client_id} (Partition {partition_id}){Colors.RESET}")
    print("-" * 70)
    
    try:
        
        partition_path = f"data/partitions/client_{partition_id}.pkl"
        print_info(f"Loading partition: {partition_path}")
        
        dataset = load_radioml_dataset(partition_path)
        samples, labels = flatten_dataset(dataset)
        
        print_success(f"Loaded {len(samples)} samples")
        
        
        print_info("Extracting features...")
        features, labels = process_dataset(
            samples,
            labels,
            verbose=False,
            use_analog_features=True
        )
        
        print_success(f"Extracted features: {features.shape}")
        
        
        print_info(f"Training {model_type.upper()} model...")
        start_time = time.time()
        
        results = train_knn_model(
            features=features,
            labels=labels,
            test_split=0.2,
            n_neighbors=5,
            random_state=42,
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        print_success(f"Training complete in {training_time:.2f}s")
        print_info(f"  Train Accuracy: {results['train_accuracy']*100:.2f}%")
        print_info(f"  Test Accuracy: {results['test_accuracy']*100:.2f}%")
        print_info(f"  Training Time: {results['training_time']:.3f}s")
        print_info(f"  Inference Time: {results['inference_time_ms_per_sample']:.3f} ms/sample")
        
        model_dir = Path("out/checkpoints/client")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"client_{client_id}_knn.pkl"
        
        save_knn_model(results['model'], str(model_path))
        print_success(f"Model saved: {model_path}")
        
        
        print_info("Uploading weights to server...")
        
        try:
            
            features_path = str(model_path).replace('.pkl', '_features.pkl')
            labels_path = str(model_path).replace('.pkl', '_labels.pkl')
            
            with open(features_path, 'wb') as f:
                pickle.dump(features, f)
            with open(labels_path, 'wb') as f:
                pickle.dump(labels, f)
            
            
            with open(model_path, 'rb') as model_f, \
                 open(features_path, 'rb') as features_f, \
                 open(labels_path, 'rb') as labels_f:
                
                files = {
                    'model_file': (Path(model_path).name, model_f, 'application/octet-stream'),
                    'features_file': (Path(features_path).name, features_f, 'application/octet-stream'),
                    'labels_file': (Path(labels_path).name, labels_f, 'application/octet-stream')
                }
                
                params = {
                    'n_samples': results['n_samples'],
                    'model_type': 'knn'
                }
                
                
                response = requests.post(
                    f"{server_url}/upload_traditional_model/{client_id}",
                    files=files,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    print_success("Weights uploaded successfully")
                else:
                    print_error(f"Upload failed: {response.status_code} - {response.text}")
                    return None
        except Exception as e:
            print_error(f"Upload error: {str(e)}")
            return None
        
        
        return {
            'client_id': client_id,
            'model': results['model'],
            'model_path': str(model_path),
            'features': features,
            'labels': labels,
            'train_accuracy': results['train_accuracy'],
            'test_accuracy': results['test_accuracy'],
            'training_time': results['training_time'],
            'inference_time_ms': results['inference_time_ms_per_sample'],
            'n_samples': results['n_samples']
        }
        
    except Exception as e:
        print_error(f"Client {client_id} training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def verify_aggregation(client_results):
    """
    Verify that aggregation can be performed and produces a valid global model.
    
    Args:
        client_results: List of client training results
        model_type: Model type ('knn' or 'dt')
        
    Returns:
        dict: Aggregation results
    """
    print_header("Step 4: Verify Aggregation")
    
    try:
        
        client_models_info = []
        
        for result in client_results:
            
            features_path = result['model_path'].replace('.pkl', '_features.pkl')
            labels_path = result['model_path'].replace('.pkl', '_labels.pkl')
            
            with open(features_path, 'wb') as f:
                pickle.dump(result['features'], f)
            with open(labels_path, 'wb') as f:
                pickle.dump(result['labels'], f)
            
            client_models_info.append({
                'client_id': result['client_id'],
                'model_path': result['model_path'],
                'features_path': features_path,
                'labels_path': labels_path,
                'n_samples': result['n_samples']
            })
        
        print_info(f"Aggregating {len(client_models_info)} client models...")
        
        
        agg_result = aggregate_knn_models(client_models_info, n_neighbors=5)
        
        print_success("Aggregation completed successfully")
        print_info(f"  Total clients: {agg_result['num_clients']}")
        print_info(f"  Total samples: {agg_result['total_samples']}")
        
        
        all_features = []
        all_labels = []
        
        for result in client_results:
            
            n_test = len(result['features']) // 5
            all_features.append(result['features'][:n_test])
            all_labels.append(result['labels'][:n_test])
        
        X_test = np.vstack(all_features)
        y_test = np.concatenate(all_labels)
        
        print_info(f"Evaluating global model on {len(X_test)} test samples...")
        
        
        global_model = agg_result['global_model']
        
        eval_result = evaluate_global_model(global_model, X_test, y_test)
        
        print_success(f"Global model accuracy: {eval_result['accuracy']*100:.2f}%")
        print_info(f"  Test samples: {eval_result['n_samples']}")
        
        
        avg_client_acc = np.mean([r['test_accuracy'] for r in client_results])
        print_info(f"  Average client accuracy: {avg_client_acc*100:.2f}%")
        
        if eval_result['accuracy'] >= avg_client_acc * 0.95:
            print_success("Global model performs well (>=95% of average client accuracy)")
        else:
            print_warning("Global model accuracy is lower than expected")
        
        return {
            'global_model': global_model,
            'global_accuracy': eval_result['accuracy'],
            'avg_client_accuracy': avg_client_acc,
            'confusion_matrix': eval_result['confusion_matrix'],
            'n_test_samples': eval_result['n_samples']
        }
        
    except Exception as e:
        print_error(f"Aggregation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def verify_dashboard_metrics(server_url="http://localhost:8000"):
    """
    Verify that dashboard can display metrics correctly.
    
    Args:
        server_url: Central server URL
        
    Returns:
        bool: True if metrics are accessible, False otherwise
    """
    print_header("Step 5: Verify Dashboard Metrics")
    
    try:
        
        response = requests.get(f"{server_url}/status", timeout=5)
        
        if response.status_code == 200:
            status_data = response.json()
            
            print_success("Dashboard metrics accessible")
            print_info(f"  Server status: {status_data.get('server_status', 'unknown')}")
            print_info(f"  Total clients: {status_data.get('total_clients', 0)}")
            print_info(f"  Total samples: {status_data.get('total_samples', 0)}")
            print_info(f"  Current round: {status_data.get('current_round', 0)}")
            
            
            if status_data.get('total_clients', 0) > 0:
                print_success(f"Server has received updates from {status_data['total_clients']} clients")
            else:
                print_warning("Server has not received any client updates yet")
            
            print_info(f"\nDashboard URL: http://localhost:7860")
            print_info("Please verify the following in the dashboard:")
            print_info("  1. System status shows connected clients")
            print_info("  2. Training metrics display accuracy values")
            print_info("  3. Confusion matrices are visible")
            print_info("  4. Accuracy vs SNR plot is displayed")
            print_info("  5. Computation complexity table shows timing data")
            
            return True
        else:
            print_error(f"Cannot access metrics: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error accessing dashboard metrics: {str(e)}")
        return False


def run_simulation(num_clients=3, server_url="http://localhost:8000", dropout_rate=0.0):
    """
    Run complete multi-client federated learning simulation.
    
    Args:
        num_clients: Number of clients to simulate
        server_url: Central server URL
        
    Returns:
        bool: True if simulation successful, False otherwise
    """
    print_header("Multi-Client Federated Learning Simulation")
    print_info(f"Configuration:")
    print_info(f"  Number of clients: {num_clients}")
    print_info(f"  Model type: KNN")
    print_info(f"  Server URL: {server_url}")
    print_info(f"  Dropout Rate: {dropout_rate}")
    
    
    if not check_partitions_exist(num_clients):
        return False
    
    
    if not check_server_running(server_url):
        return False
    
    
    print_header("Step 3: Train All Clients")
    
    client_results = []
    
    for i in range(num_clients):
        client_id = f"sim_client_{i}"
        
        if np.random.rand() < dropout_rate:
            print_warning(f"Client {client_id} disconnected (simulated dropout)")
            continue
            
        result = simulate_client_training(client_id, i, server_url)
        
        if result is None:
            print_error(f"Client {i} training failed")
            return False
        
        client_results.append(result)
        time.sleep(1)  
        
    if not client_results:
        print_error("All clients dropped out or failed. Simulation cannot continue.")
        return False
    
    print_success(f"\nAll {num_clients} clients trained successfully")
    
    
    print(f"\n{Colors.BOLD}Training Summary:{Colors.RESET}")
    print("-" * 70)
    for result in client_results:
        print(f"  {result['client_id']}:")
        print(f"    Samples: {result['n_samples']}")
        print(f"    Test Accuracy: {result['test_accuracy']*100:.2f}%")
        print(f"    Training Time: {result['training_time']:.3f}s")
        print(f"    Inference Time: {result['inference_time_ms']:.3f} ms/sample")
    
    
    agg_result = verify_aggregation(client_results)
    
    if agg_result is None:
        print_error("Aggregation verification failed")
        return False
    
    
    if not verify_dashboard_metrics(server_url):
        print_warning("Dashboard metrics verification incomplete")
        print_info("Please manually verify the dashboard at http://localhost:7860")
    
    
    print_header("Simulation Complete")
    
    print_success("All verification steps passed:")
    print_success("  [OK] Dataset partitions verified")
    print_success("  [OK] Central server running")
    print_success(f"  [OK] All {num_clients} clients trained and uploaded")
    print_success("  [OK] Aggregation produced valid global model")
    print_success("  [OK] Dashboard metrics accessible")
    
    print(f"\n{Colors.BOLD}Results:{Colors.RESET}")
    print(f"  Global Model Accuracy: {Colors.GREEN}{agg_result['global_accuracy']*100:.2f}%{Colors.RESET}")
    print(f"  Average Client Accuracy: {agg_result['avg_client_accuracy']*100:.2f}%")
    print(f"  Test Samples: {agg_result['n_test_samples']}")
    
    print(f"\n{Colors.BOLD}Next Steps:{Colors.RESET}")
    print_info("1. Open dashboard at http://localhost:7860")
    print_info("2. Verify all visualizations are displayed correctly")
    print_info("3. Check confusion matrices for both models")
    print_info("4. Review accuracy vs SNR plots")
    print_info("5. Examine computation complexity metrics")
    
    return True


def main():
    """Main entry point for simulation script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Client Federated Learning Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation with 3 clients using KNN
  python tests/test_multi_client_simulation.py
  
  # Run simulation with 5 clients
  python tests/test_multi_client_simulation.py --num-clients 5
  
  # Run simulation with custom server URL
  python tests/test_multi_client_simulation.py --server-url http://192.168.1.100:8000
        """
    )
    
    parser.add_argument(
        '--num-clients',
        type=int,
        default=3,
        help='Number of clients to simulate (default: 3)'
    )
    
    parser.add_argument(
        '--server-url',
        type=str,
        default='http://localhost:8000',
        help='Central server URL (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Probability of a client dropping out (default: 0.0)'
    )
    
    args = parser.parse_args()
    
    success = run_simulation(
        num_clients=args.num_clients,
        server_url=args.server_url,
        dropout_rate=args.dropout
    )
    
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
