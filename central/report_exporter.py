import os
import json
import logging
import pandas as pd
from typing import Dict, Any

from central.state import (
    get_historical_metrics_history,
    get_dashboard_summary,
    get_all_aggregation_results,
    get_all_trust_scores
)

logger = logging.getLogger("federated_central")

def export_round_reports(timestamp: str, out_dir: str = "out/reports") -> None:
    """
    Export comprehensive experiment reports to out/reports directory.
    
    Args:
        timestamp: String timestamp to append to filenames, ensuring sync with plots.
        out_dir: The directory to save the reports to.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        
        # 1. Export Round Metrics (CSV)
        _export_round_metrics_csv(timestamp, out_dir)
        
        # 2. Export Experiment Summary (JSON)
        _export_experiment_summary_json(timestamp, out_dir)
        
        # 3. Export Training Summary (JSON)
        _export_training_summary_json(timestamp, out_dir)
        
        logger.info(f"Successfully exported experiment reports to {out_dir}")
        
    except Exception as e:
        logger.error(f"Failed to export experiment reports: {e}", exc_info=True)


def _export_round_metrics_csv(timestamp: str, out_dir: str) -> None:
    """Export round-by-round metrics history as a CSV file."""
    try:
        history = get_historical_metrics_history(last_n=1000) # Fetch all available history
        rounds = history.get('rounds', [])
        
        if not rounds:
            logger.debug("No historical metrics available to export CSV.")
            return
            
        # Flatten the data for pandas
        flat_data = []
        for r in rounds:
            row = {
                'round': r.get('round', 0),
                'timestamp': r.get('timestamp', ''),
                'participating_clients': r.get('participating_clients', 0)
            }
            # Flatten evaluation metrics if present
            eval_metrics = r.get('evaluation', {})
            row['knn_accuracy'] = eval_metrics.get('knn_accuracy', 0.0)
            
            # Flatten defense stats if present
            defense = r.get('defense_report', {})
            if defense:
                row['accepted_clients'] = defense.get('accepted_count', 0)
                row['rejected_clients'] = defense.get('rejected_count', 0)
                
            flat_data.append(row)
            
        df = pd.DataFrame(flat_data)
        out_path = os.path.join(out_dir, f"round_metrics_{timestamp}.csv")
        df.to_csv(out_path, index=False)
        logger.debug(f"Exported round metrics CSV to {out_path}")
        
    except Exception as e:
        logger.error(f"Error exporting round metrics CSV: {e}")


def _export_experiment_summary_json(timestamp: str, out_dir: str) -> None:
    """Export top-level dashboard summary as a JSON file."""
    try:
        summary = get_dashboard_summary()
        out_path = os.path.join(out_dir, f"experiment_summary_{timestamp}.json")
        
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        logger.debug(f"Exported experiment summary JSON to {out_path}")
    except Exception as e:
        logger.error(f"Error exporting experiment summary JSON: {e}")


def _export_training_summary_json(timestamp: str, out_dir: str) -> None:
    """Export detailed training and aggregation breakdown as a JSON file."""
    try:
        results = {
            "knn_aggregations": get_all_aggregation_results("knn"),
            "dt_aggregations": get_all_aggregation_results("dt"),
            "client_trust_scores": get_all_trust_scores()
        }
        
        out_path = os.path.join(out_dir, f"training_summary_{timestamp}.json")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.debug(f"Exported training summary JSON to {out_path}")
    except Exception as e:
        logger.error(f"Error exporting training summary JSON: {e}")
