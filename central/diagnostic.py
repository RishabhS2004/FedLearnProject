"""
Server Diagnostic Script

This script checks the server status and helps diagnose why aggregation isn't working.
"""

import requests
import json
import sys

SERVER_URL = "http://localhost:8000"

def print_header(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def check_server_health():
    """Check if server is running."""
    print_header("Step 1: Check Server Health")
    
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is running")
            data = response.json()
            print(f"  Status: {data.get('status')}")
            print(f"  Timestamp: {data.get('timestamp')}")
            return True
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {SERVER_URL}")
        print("  Please start the server: python central/main.py")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def check_server_status():
    """Check server status and client uploads."""
    print_header("Step 2: Check Server Status")
    
    try:
        response = requests.get(f"{SERVER_URL}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            print(f"Server Status: {data.get('server_status')}")
            print(f"Total Clients: {data.get('total_clients')}")
            print(f"Total Samples: {data.get('total_samples')}")
            print(f"Last Aggregation: {data.get('last_aggregation')}")
            
            if data.get('last_aggregation') is None:
                print("\n WARNING: No aggregation has been run yet!")
                print("  This is why the dashboard shows no data.")
            
            
            clients = data.get('clients', [])
            if clients:
                print(f"\nClients ({len(clients)}):")
                
                
                knn_clients = []
                unknown_clients = []
                
                for client in clients:
                    client_id = client.get('client_id')
                    model_type = client.get('model_type', 'unknown')
                    
                    weights_path = client.get('weights_path', '')
                    if 'knn' in weights_path.lower() or model_type == 'knn':
                        model_type = 'knn'
                    
                    if model_type == 'knn':
                        knn_clients.append(client_id)
                    else:
                        unknown_clients.append(client_id)
                
                if knn_clients:
                    print(f"\n  KNN Models ({len(knn_clients)}):")
                    for cid in knn_clients:
                        print(f"    - {cid}")
                
                if unknown_clients:
                    print(f"\n  Unknown Model Type ({len(unknown_clients)}):")
                    for cid in unknown_clients:
                        print(f"    - {cid}")
                
                return {
                    'knn': len(knn_clients),
                    'has_aggregation': data.get('last_aggregation') is not None
                }
            else:
                print("\n No clients have uploaded yet")
                print("  Please train and upload from clients first")
                return None
        else:
            print(f" Server returned status code: {response.status_code}")
            return None
    except Exception as e:
        print(f" Error: {str(e)}")
        return None


def suggest_aggregation(status_info):
    """Suggest which aggregation to run."""
    print_header("Step 3: Aggregation Recommendations")
    
    if status_info is None:
        print("Cannot make recommendations - no client data available")
        return
    
    if status_info['has_aggregation']:
        print(" Aggregation has already been run")
        print("  Dashboard should be showing data")
        print("\nTo run aggregation again:")
    else:
        print(" No aggregation has been run yet")
        print("\nTo fix the dashboard, run aggregation:")
    
    print()
    
    if status_info['knn'] > 0:
        print(f" KNN Aggregation ({status_info['knn']} clients available):")
        print(f'  curl -X POST "{SERVER_URL}/aggregate"')
        print()
    else:
        print(" No KNN models available for aggregation")
        print("  Please train and upload from clients first")


def check_dashboard():
    """Check if dashboard is accessible."""
    print_header("Step 4: Check Dashboard")
    
    dashboard_url = "http://localhost:7860"
    
    try:
        response = requests.get(dashboard_url, timeout=5)
        if response.status_code == 200:
            print(f" Dashboard is accessible at {dashboard_url}")
            print("\nOpen in browser to view visualizations")
        else:
            print(f" Dashboard returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f" Cannot connect to dashboard at {dashboard_url}")
        print("  Dashboard may not be running")
    except Exception as e:
        print(f" Error checking dashboard: {str(e)}")


def main():
    """Run all diagnostic checks."""
    print_header("Server Diagnostic Tool")
    print("This tool checks why aggregation isn't working")

    if not check_server_health():
        print("\n" + "="*70)
        print("DIAGNOSIS: Server is not running")
        print("="*70)
        print("\nSOLUTION: Start the server with: python central/main.py")
        return 1
    
    status_info = check_server_status()
    suggest_aggregation(status_info)
    check_dashboard()
    print_header("Summary")
    
    if status_info and not status_info['has_aggregation']:
        print("DIAGNOSIS: Aggregation has not been run")
        print("\nThis is why the dashboard shows no data.")
        print("\nSOLUTION:")
        print("1. Run the aggregation command shown above")
        print("2. Refresh the dashboard (http://localhost:7860)")
        print("3. Verify metrics are displayed")
    elif status_info and status_info['has_aggregation']:
        print(" Server is running")
        print(" Clients have uploaded")
        print(" Aggregation has been run")
        print("\nDashboard should be showing data!")
        print("If not, try:")
        print("1. Hard refresh browser (Ctrl+F5 or Cmd+Shift+R)")
        print("2. Check browser console for errors (F12)")
        print("3. Run aggregation again")
    else:
        print("DIAGNOSIS: No clients have uploaded")
        print("\nSOLUTION:")
        print("1. Start clients: python client/main.py --port 7861 --auto-id")
        print("2. Train models on each client")
        print("3. Upload weights (auto-upload should be enabled)")
        print("4. Run aggregation")
    
    print("\n" + "="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
