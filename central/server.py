"""
FastAPI Server for Federated Learning Central Server

REST API endpoints for client communication including model upload,
Byzantine-resilient aggregation, global model download, trust scores,
and status queries. Supports KNN and Decision Tree models.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import logging
import threading
from typing import Optional

from central.state import (
    load_config,
    register_client_connection,
    update_client_training_status,
    register_client_upload,
    get_client_status,
    get_all_client_weights,
    get_registry_stats,
    store_aggregation_result,
    get_latest_aggregation_result,
    track_client_upload,
    get_pending_uploads_count,
    get_auto_aggregation_threshold,
    should_trigger_aggregation,
    initialize_auto_aggregation_state,
    initialize_metrics_history
)
from central.aggregator import (
    aggregate_knn_models,
    aggregate_dt_models,
    save_knn_model,
    save_dt_model,
    evaluate_global_model
)
from central.byzantine import (
    get_all_trust_scores,
    get_trust_score,
    get_trust_history,
    get_byzantine_aggregator,
    set_byzantine_strategy,
    initialize_trust
)
from central.utils import setup_logging, ensure_directories


# Initialize FastAPI app
app = FastAPI(
    title="RadioFed Central Server",
    description="Byzantine-Resilient Federated Learning for Automatic Modulation Classification",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
logger = None
config = None
last_aggregation_time = None

# Aggregation status tracking
_aggregation_in_progress = False
_aggregation_lock = threading.Lock()


def _initialize_server():
    """Initialize server configuration and logging."""
    global logger, config

    if logger is not None and config is not None:
        return

    ensure_directories()

    try:
        config = load_config()
        log_level = config.get("log_level", "INFO")
    except Exception as e:
        print(f"Warning: Could not load config, using defaults: {e}")
        config = {
            "model_save_path": "./central/model_store/global_knn_model.pkl",
            "host": "0.0.0.0",
            "port": 8000,
            "log_level": "INFO",
            "auto_aggregation_enabled": True,
            "auto_aggregation_threshold": 2
        }
        log_level = "INFO"

    logger = setup_logging(log_level)
    logger.info("RadioFed Central Server starting up")

    try:
        initialize_auto_aggregation_state()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    except Exception as e:
        logger.warning(f"Could not initialize auto-aggregation state: {e}")

    try:
        initialize_metrics_history()
    except Exception as e:
        logger.warning(f"Could not initialize metrics history: {e}")


@app.on_event("startup")
async def startup_event():
    _initialize_server()


def perform_auto_aggregation():
    """Execute complete auto-aggregation workflow with Byzantine filtering."""
    global _aggregation_in_progress, last_aggregation_time

    try:
        logger.info("Starting auto-aggregation workflow...")

        # Capture before-aggregation metrics
        try:
            from central.state import capture_current_metrics
            before_metrics = capture_current_metrics()
        except Exception as e:
            logger.warning(f"Could not capture before-metrics: {e}")
            before_metrics = {
                'knn_accuracy': 0.0, 'per_snr_accuracy': {},
                'confusion_matrix': [], 'num_clients': 0,
                'timestamp': datetime.now().isoformat()
            }

        # Get all client data
        client_weights_info = get_all_client_weights()
        if not client_weights_info:
            logger.warning("No client models available")
            return

        knn_clients = [c for c in client_weights_info if c.get('model_type', 'knn') == 'knn']
        timestamp = datetime.now().isoformat()

        # KNN aggregation with Byzantine filtering
        if knn_clients:
            try:
                result = aggregate_knn_models(
                    knn_clients, n_neighbors=5, evaluate=True,
                    byzantine_filtering=True
                )
                save_knn_model(result['global_model'],
                             "./central/model_store/global_knn_model.pkl")
                store_aggregation_result('knn', result, timestamp)
                logger.info(f"KNN aggregation: {result['num_clients']} clients, "
                          f"accuracy={result.get('accuracy', 0):.4f}")

                if result.get('defense_report'):
                    report = result['defense_report']
                    logger.info(f"Byzantine: {report['accepted_count']}/{report['total_clients']} accepted")
            except Exception as e:
                logger.error(f"KNN aggregation error: {e}")

        # DT aggregation with Byzantine filtering
        dt_clients = [c for c in client_weights_info if c.get('model_type') == 'dt']
        if dt_clients:
            try:
                result = aggregate_dt_models(
                    dt_clients, evaluate=True, byzantine_filtering=True
                )
                save_dt_model(result['global_model'],
                            "./central/model_store/global_dt_model.pkl")
                store_aggregation_result('dt', result, timestamp)
                logger.info(f"DT aggregation: accuracy={result.get('accuracy', 0):.4f}")
            except Exception as e:
                logger.error(f"DT aggregation error: {e}")

        # Evaluate and store round metrics
        try:
            from central.state import evaluate_global_model as eval_global
            after_metrics = eval_global()
        except Exception:
            after_metrics = {
                'knn_accuracy': 0.0, 'per_snr_accuracy': {},
                'confusion_matrix': [], 'timestamp': datetime.now().isoformat()
            }

        try:
            from central.state import store_aggregation_round
            store_aggregation_round(before_metrics, after_metrics)
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")

        try:
            from central.state import reset_aggregation_state, get_auto_aggregation_state
            reset_aggregation_state()
        except Exception as e:
            logger.error(f"Failed to reset state: {e}")

        last_aggregation_time = datetime.now().isoformat()
        logger.info("Auto-aggregation workflow completed.")

    except Exception as e:
        logger.error(f"Auto-aggregation failed: {e}", exc_info=True)
    finally:
        with _aggregation_lock:
            global _aggregation_in_progress
            _aggregation_in_progress = False


def trigger_aggregation_async():
    """Trigger aggregation in background thread."""
    global _aggregation_in_progress

    with _aggregation_lock:
        if _aggregation_in_progress:
            logger.info("Aggregation already in progress")
            return
        _aggregation_in_progress = True

    thread = threading.Thread(target=perform_auto_aggregation, daemon=True)
    thread.start()


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "RadioFed Central Server - Byzantine-Resilient FL for AMC",
        "version": "2.0.0",
        "features": ["KNN", "Decision Tree", "Byzantine Fault Tolerance",
                     "Trust Scoring", "Krum Selection"],
        "endpoints": {
            "register": "POST /register/{client_id}",
            "training_status": "POST /training_status/{client_id}?training={bool}",
            "upload_model": "POST /upload_model/{client_id}",
            "aggregate": "POST /aggregate",
            "global_model": "GET /global_model",
            "aggregation_results": "GET /aggregation_results",
            "trust_scores": "GET /trust_scores",
            "byzantine_report": "GET /byzantine_report",
            "status": "GET /status",
            "health": "GET /health"
        }
    }


@app.post("/register/{client_id}")
async def register_client(client_id: str):
    _initialize_server()
    try:
        if not client_id or not client_id.strip():
            raise HTTPException(status_code=400, detail="Invalid client_id")

        register_client_connection(client_id)
        initialize_trust(client_id)
        logger.info(f"Client registered: {client_id}")

        return JSONResponse(status_code=200, content={
            "status": "success", "client_id": client_id,
            "trust_score": get_trust_score(client_id),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error registering client {client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training_status/{client_id}")
async def update_training_status(
    client_id: str,
    training: bool = Query(..., description="Whether client is training")
):
    _initialize_server()
    try:
        if not client_id or not client_id.strip():
            raise HTTPException(status_code=400, detail="Invalid client_id")
        update_client_training_status(client_id, training)
        return JSONResponse(status_code=200, content={
            "status": "success", "client_id": client_id,
            "training": training, "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_model/{client_id}")
async def upload_model(
    client_id: str,
    n_samples: int = Query(..., description="Number of training samples"),
    model_file: UploadFile = File(...),
    features_file: UploadFile = File(...),
    labels_file: UploadFile = File(...)
):
    _initialize_server()
    try:
        if not client_id or not client_id.strip():
            raise HTTPException(status_code=400, detail="Invalid client_id")
        if n_samples <= 0:
            raise HTTPException(status_code=400, detail="n_samples must be positive")

        models_dir = "./central/model_store"
        os.makedirs(models_dir, exist_ok=True)

        # Save uploaded files
        model_path = os.path.join(models_dir, f"{client_id}_knn_model.pkl")
        with open(model_path, "wb") as f:
            f.write(await model_file.read())

        features_path = os.path.join(models_dir, f"{client_id}_features.pkl")
        with open(features_path, "wb") as f:
            f.write(await features_file.read())

        labels_path = os.path.join(models_dir, f"{client_id}_labels.pkl")
        with open(labels_path, "wb") as f:
            f.write(await labels_file.read())

        # Register upload
        register_client_upload(
            client_id=client_id, n_samples=n_samples,
            weights_path=model_path, model_type='knn',
            model_path=model_path, features_path=features_path,
            labels_path=labels_path
        )

        initialize_trust(client_id)
        track_client_upload(client_id)

        pending = get_pending_uploads_count()
        threshold = get_auto_aggregation_threshold()

        logger.info(f"Upload from {client_id}: {pending}/{threshold}")

        if should_trigger_aggregation():
            trigger_aggregation_async()

        return JSONResponse(status_code=200, content={
            "status": "success", "client_id": client_id,
            "model_type": "knn", "n_samples": n_samples,
            "trust_score": get_trust_score(client_id),
            "timestamp": datetime.now().isoformat(),
            "upload_status": {
                "pending_uploads": pending,
                "threshold": threshold,
                "ready_for_aggregation": pending >= threshold
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error from {client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/aggregate")
async def aggregate():
    _initialize_server()
    global last_aggregation_time

    try:
        client_weights_info = get_all_client_weights()
        if not client_weights_info:
            raise HTTPException(status_code=400, detail="No client models available")

        knn_clients = [c for c in client_weights_info if c.get('model_type', 'knn') == 'knn']
        if not knn_clients:
            raise HTTPException(status_code=400, detail="No KNN models available")

        timestamp = datetime.now().isoformat()
        result = aggregate_knn_models(knn_clients, n_neighbors=5, evaluate=True,
                                     byzantine_filtering=True)
        save_knn_model(result['global_model'], "./central/model_store/global_knn_model.pkl")
        store_aggregation_result('knn', result, timestamp)

        last_aggregation_time = timestamp

        response = {
            "status": "success", "model_type": "knn",
            "num_clients": result['num_clients'],
            "total_samples": result['total_samples'],
            "accuracy": result.get('accuracy', 0.0),
            "trust_scores": result.get('trust_scores', {}),
            "timestamp": timestamp
        }

        if result.get('defense_report'):
            response['byzantine_report'] = {
                'accepted': result['defense_report']['accepted_count'],
                'rejected': result['defense_report']['rejected_count'],
                'strategy': result['defense_report']['strategy']
            }

        return JSONResponse(status_code=200, content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Aggregation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/global_model")
async def get_global_model():
    _initialize_server()
    try:
        path = "./central/model_store/global_knn_model.pkl"
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Global model not found")
        return FileResponse(path, media_type="application/octet-stream",
                          filename="global_knn_model.pkl")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/aggregation_results")
async def get_aggregation_results():
    _initialize_server()
    try:
        result = get_latest_aggregation_result('knn')
        if result is None:
            raise HTTPException(status_code=404, detail="No aggregation results found")

        response_result = {k: v for k, v in result['result'].items()
                         if k not in ('global_model', 'predictions')}

        return JSONResponse(status_code=200, content={
            "status": "success", "model_type": "knn",
            "result": response_result, "timestamp": result['timestamp']
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trust_scores")
async def get_trust_scores_endpoint():
    """Get all client trust scores."""
    _initialize_server()
    scores = get_all_trust_scores()
    return JSONResponse(status_code=200, content={
        "trust_scores": scores,
        "threshold": 0.3,
        "timestamp": datetime.now().isoformat()
    })


@app.get("/trust_history/{client_id}")
async def get_trust_history_endpoint(client_id: str):
    """Get trust score history for a specific client."""
    _initialize_server()
    history = get_trust_history(client_id)
    return JSONResponse(status_code=200, content={
        "client_id": client_id,
        "current_score": get_trust_score(client_id),
        "history": history
    })


@app.get("/byzantine_report")
async def get_byzantine_report():
    """Get latest Byzantine defense report."""
    _initialize_server()
    aggregator = get_byzantine_aggregator()
    log = aggregator.get_aggregation_log()

    return JSONResponse(status_code=200, content={
        "strategy": aggregator.strategy,
        "total_rounds": len(log),
        "latest_report": log[-1] if log else None,
        "timestamp": datetime.now().isoformat()
    })


@app.post("/byzantine_strategy")
async def set_strategy(strategy: str = Query(...)):
    """Set Byzantine defense strategy."""
    _initialize_server()
    valid = ['krum', 'trimmed_mean', 'trust_weighted', 'full']
    if strategy not in valid:
        raise HTTPException(status_code=400,
                          detail=f"Invalid strategy. Must be one of: {valid}")
    set_byzantine_strategy(strategy)
    return {"status": "success", "strategy": strategy}


@app.get("/status")
async def get_status():
    _initialize_server()
    try:
        clients = get_client_status()
        stats = get_registry_stats()
        trust_scores = get_all_trust_scores()

        return JSONResponse(status_code=200, content={
            "server_status": "running",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "last_aggregation": last_aggregation_time,
            "global_model_exists": os.path.exists("./central/model_store/global_knn_model.pkl"),
            "total_clients": stats['total_clients'],
            "total_samples": stats['total_samples'],
            "trust_scores": trust_scores,
            "byzantine_strategy": get_byzantine_aggregator().strategy,
            "clients": clients
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "byzantine_enabled": True,
        "timestamp": datetime.now().isoformat()
    }
