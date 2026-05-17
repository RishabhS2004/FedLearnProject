import os
import sys
import time
import json
import shutil
import platform
import subprocess
import traceback
import random
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

class ExperimentTracker:
    """
    Tracks a federated learning simulation experiment.
    
    Features:
    - Automatically captures system metadata (Python version, platform, git hash).
    - Snapshots filesystem before execution to isolate new artifacts.
    - Copies new artifacts into an isolated experiment directory after execution.
    - Captures exact CLI arguments and configuration.
    - Generates a summary.json and manifest.json.
    - Ensures deterministic random seeding.
    """
    
    # Directories to monitor for new artifacts
    MONITORED_DIRS = [
        "out/metrics",
        "out/plots",
        "out/reports",
        "out/predictions",
        "out/logs"
    ]
    
    def __init__(
        self,
        strategy: str,
        model: str,
        feature_mode: str,
        dataset: str,
        num_clients: int,
        num_byzantine: int,
        n_rounds: int,
        distribution: str,
        training_mode: str = "unknown",
        random_seed: int = 42,
        base_dir: str = "out/experiments"
    ):
        self.strategy = strategy
        self.model = model
        self.feature_mode = feature_mode
        self.dataset = dataset
        self.num_clients = num_clients
        self.num_byzantine = num_byzantine
        self.n_rounds = n_rounds
        self.distribution = distribution
        self.training_mode = training_mode
        self.random_seed = random_seed
        self.base_dir = base_dir
        
        # Determine root directory (assuming tests/experiment_tracker.py)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_strategy = strategy.replace("_", "-")
        self.exp_id = f"EXP_{self.timestamp}_{safe_strategy}_{model}"
        self.exp_dir = os.path.join(self.project_root, base_dir, self.exp_id)
        
        self.start_time = None
        self.end_time = None
        self.status = "initialized"
        self.metrics_summary = {}
        
        self._initial_files = {}
        self._manifest = {}
        
    def _get_git_hash(self) -> str:
        """Attempt to retrieve the current git commit hash."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
        except Exception:
            return "unknown"
            
    def _snapshot_files(self) -> Dict[str, float]:
        """Snapshot modification times of all files in monitored directories."""
        snapshot = {}
        for rel_dir in self.MONITORED_DIRS:
            abs_dir = os.path.join(self.project_root, rel_dir)
            if not os.path.exists(abs_dir):
                continue
                
            for root, _, files in os.walk(abs_dir):
                for f in files:
                    file_path = os.path.join(root, f)
                    try:
                        snapshot[file_path] = os.path.getmtime(file_path)
                    except OSError:
                        pass
        return snapshot

    def __enter__(self):
        """Start the experiment tracking."""
        self.start_time = time.time()
        self.status = "running"
        
        # Ensure base experiment directory exists
        os.makedirs(os.path.join(self.project_root, self.base_dir), exist_ok=True)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Set deterministic seeds
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # Snapshot the filesystem to know what already exists
        self._initial_files = self._snapshot_files()
        
        # Save config snapshot immediately
        self._save_config_snapshot()
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish the experiment tracking, copy files, and generate summary."""
        self.end_time = time.time()
        
        if exc_type is KeyboardInterrupt:
            self.status = "interrupted"
        elif exc_type is not None:
            self.status = "failed"
            self.metrics_summary["error"] = str(exc_val)
            self.metrics_summary["traceback"] = "".join(traceback.format_tb(exc_tb))
        else:
            self.status = "completed"
            
        self._collect_artifacts()
        self._save_manifest()
        self._save_summary()
        
        # Do not swallow exceptions unless we want to, so we return False
        return False

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update the tracking summary with custom metrics during/after run."""
        self.metrics_summary.update(metrics)

    def _collect_artifacts(self):
        """Find new/modified files and copy them to the experiment folder."""
        final_files = self._snapshot_files()
        
        for file_path, mtime in final_files.items():
            # If the file is new, or if its modification time changed
            if file_path not in self._initial_files or mtime > self._initial_files[file_path]:
                self._copy_artifact(file_path)

    def _copy_artifact(self, src_abs_path: str):
        """Copy a single file preserving its relative structure."""
        rel_path = os.path.relpath(src_abs_path, self.project_root)
        dest_abs_path = os.path.join(self.exp_dir, rel_path)
        
        os.makedirs(os.path.dirname(dest_abs_path), exist_ok=True)
        
        try:
            # copy2 preserves original metadata like timestamps
            shutil.copy2(src_abs_path, dest_abs_path)
            self._manifest[rel_path] = dest_abs_path
        except Exception as e:
            print(f"Warning: Failed to copy artifact {rel_path}: {e}")

    def _save_config_snapshot(self):
        """Save the experiment configuration and system metadata."""
        config = {
            "experiment_id": self.exp_id,
            "timestamp": self.timestamp,
            "system": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "git_commit": self._get_git_hash(),
            },
            "execution": {
                "cli_args": sys.argv,
                "cwd": os.getcwd()
            },
            "parameters": {
                "strategy": self.strategy,
                "model": self.model,
                "feature_mode": self.feature_mode,
                "dataset": self.dataset,
                "num_clients": self.num_clients,
                "num_byzantine": self.num_byzantine,
                "n_rounds": self.n_rounds,
                "distribution": self.distribution,
                "training_mode": self.training_mode,
                "random_seed": self.random_seed
            }
        }
        
        config_path = os.path.join(self.exp_dir, "config_snapshot.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    def _save_manifest(self):
        """Save the manifest mapping original paths to local copies."""
        manifest_path = os.path.join(self.exp_dir, "manifest.json")
        
        data = {
            "experiment_id": self.exp_id,
            "artifact_count": len(self._manifest),
            "artifacts": self._manifest
        }
        
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=4)

    def _save_summary(self):
        """Save the final execution summary including metrics and runtime."""
        duration = self.end_time - self.start_time if self.start_time else 0
        
        summary = {
            "experiment_id": self.exp_id,
            "status": self.status,
            "training_mode": self.training_mode,
            "runtime_seconds": round(duration, 2),
            "metrics": self.metrics_summary,
            "generated_artifacts": len(self._manifest)
        }
        
        summary_path = os.path.join(self.exp_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
