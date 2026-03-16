"""
RadioFed Central Server — Launcher

Starts both:
1. FastAPI REST server (port 8000) — client-server communication
2. FastHTML Dashboard (port 7860)  — monitoring UI
"""

import sys, os, time, threading, logging, socket

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from central.state import load_config, save_config
from central.server import app as fastapi_app
from central.dashboard_app import create_dashboard_app
from central.utils import setup_logging, ensure_directories

server_thread = None
dashboard_thread = None
logger = None
config = None


def initialize():
    global logger, config
    ensure_directories()
    logger = setup_logging("INFO")
    logger.info("Initializing RadioFed Central Server")
    try:
        config = load_config()
    except Exception:
        config = {
            "model_save_path": "./central/model_store/global_knn_model.pkl",
            "host": "127.0.0.1", "port": 8000, "log_level": "INFO",
            "auto_aggregation_enabled": True, "auto_aggregation_threshold": 2,
        }


def is_port_available(host, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def wait_for_ready(host, port, timeout=10):
    import requests
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200: return True
        except Exception:
            pass
        time.sleep(.5)
    return False


def start_fastapi(host, port):
    global server_thread
    actual = "127.0.0.1" if host in ("0.0.0.0", "127.0.0.1") else host
    if not is_port_available(actual, port):
        logger.error(f"Port {port} is in use")
        return False

    def run():
        uvicorn.run(fastapi_app, host=actual, port=port, log_level="warning")

    server_thread = threading.Thread(target=run, daemon=True)
    server_thread.start()

    if wait_for_ready(actual, port):
        logger.info(f"FastAPI ready at http://localhost:{port}")
        return True
    logger.error("FastAPI failed to start")
    return False


def main():
    try:
        initialize()

        host = config.get("host", "127.0.0.1")
        api_port = config.get("port", 8000)
        dash_port = 7860

        logger.info("=" * 56)
        logger.info("  RadioFed - Byzantine-Resilient Federated Learning")
        logger.info("=" * 56)

        # Start FastAPI
        if not start_fastapi(host, api_port):
            logger.error("Cannot start FastAPI. Exiting.")
            sys.exit(1)

        logger.info(f"  API:       http://localhost:{api_port}")
        logger.info(f"  API Docs:  http://localhost:{api_port}/docs")
        logger.info(f"  Dashboard: http://localhost:{dash_port}")
        logger.info("=" * 56)

        # Start FastHTML Dashboard on main thread via uvicorn directly
        dashboard_app = create_dashboard_app(port=dash_port)
        uvicorn.run(dashboard_app, host="127.0.0.1", port=dash_port, log_level="info")

    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    except Exception as e:
        logger.error(f"Fatal: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
