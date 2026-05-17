import argparse
import os
import json
import pickle
import numpy as np
from datetime import datetime

def print_header(title):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def run_inference(model_path, features_path, labels_path=None, output_dir="out/predictions", num_to_display=10):
    print_header("RadioFed Inference Engine")
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return

    if not os.path.exists(features_path):
        print(f"ERROR: Features file not found: {features_path}")
        return

    # Load Model
    try:
        model = load_pickle(model_path)
        print(f"Model loaded successfully: {type(model).__name__}")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return

    # Load Data
    try:
        features = load_pickle(features_path)
        features = np.array(features)
        print(f"Loaded {features.shape[0]} samples with {features.shape[1] if len(features.shape) > 1 else 'unknown'} features.")
    except Exception as e:
        print(f"ERROR: Failed to load features: {e}")
        return

    labels = None
    if labels_path and os.path.exists(labels_path):
        try:
            labels = np.array(load_pickle(labels_path))
            print(f"Loaded true labels.")
        except Exception as e:
            print(f"WARNING: Failed to load labels: {e}")

    # Perform Inference
    print("\nRunning inference...")
    start_time = datetime.now()
    
    try:
        predictions = model.predict(features)
    except Exception as e:
        print(f"ERROR: Model prediction failed: {e}")
        return

    # Confidence scores (predict_proba)
    probabilities = None
    has_confidence = hasattr(model, "predict_proba")
    
    if has_confidence:
        try:
            probabilities = model.predict_proba(features)
            print("Confidence scores extracted.")
        except Exception as e:
            print(f"WARNING: predict_proba failed despite model having the attribute: {e}")
            has_confidence = False
    else:
        print("Notice: Model does not support confidence scores (predict_proba).")

    inference_duration = (datetime.now() - start_time).total_seconds()
    
    # Calculate accuracy if labels are provided
    accuracy = None
    if labels is not None and len(labels) == len(predictions):
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(labels, predictions)

    # Format output
    timestamp_str = datetime.now().isoformat()
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prediction details
    prediction_list = []
    for i in range(len(predictions)):
        pred_entry = {
            "sample_index": i,
            "predicted_class": int(predictions[i]) if isinstance(predictions[i], (np.integer, int, float)) else str(predictions[i])
        }
        
        if labels is not None:
            pred_entry["true_class"] = int(labels[i]) if isinstance(labels[i], (np.integer, int, float)) else str(labels[i])
            pred_entry["is_correct"] = bool(predictions[i] == labels[i])
            
        if has_confidence and probabilities is not None:
            # Get the confidence score for the predicted class
            # Ensure proper handling of probability arrays
            class_probs = [float(p) for p in probabilities[i]]
            pred_entry["confidence_score"] = float(np.max(probabilities[i]))
            pred_entry["all_probabilities"] = class_probs

        prediction_list.append(pred_entry)

    # Build JSON report
    report = {
        "metadata": {
            "timestamp": timestamp_str,
            "model_type": type(model).__name__,
            "total_samples": len(predictions),
            "inference_duration_seconds": float(inference_duration),
            "supports_confidence": has_confidence
        },
        "summary": {},
        "predictions": prediction_list
    }

    if accuracy is not None:
        report["summary"]["accuracy"] = float(accuracy)
        print(f"Accuracy on provided labels: {accuracy*100:.2f}%")

    if has_confidence:
        avg_confidence = float(np.mean([p["confidence_score"] for p in prediction_list]))
        report["summary"]["average_confidence"] = avg_confidence

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"inference_report_{file_timestamp}.json")
    
    with open(out_file, 'w') as f:
        json.dump(report, f, indent=2)

    # Terminal output summary
    print_header("Inference Summary")
    print(f"{'Sample':<10} | {'Prediction':<15} | " + 
          (f"{'True Label':<15} | {'Correct':<10} | " if labels is not None else "") +
          (f"{'Confidence':<15}" if has_confidence else ""))
    print("-" * (43 + (30 if labels is not None else 0) + (15 if has_confidence else 0)))
    
    display_count = min(num_to_display, len(prediction_list))
    for p in prediction_list[:display_count]:
        line = f"{p['sample_index']:<10} | {p['predicted_class']:<15} | "
        if labels is not None:
            line += f"{p['true_class']:<15} | {str(p['is_correct']):<10} | "
        if has_confidence:
            line += f"{p['confidence_score']:.4f}"
        print(line)
        
    if len(prediction_list) > display_count:
        print(f"... and {len(prediction_list) - display_count} more samples")
        
    print(f"\nTotal Samples: {len(predictions)}")
    if has_confidence:
        print(f"Average Confidence: {avg_confidence:.4f}")
        
    print(f"\nDetailed JSON report saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Inference Script for RadioFed")
    parser.add_argument("--model", required=True, help="Path to the trained model (.pkl)")
    parser.add_argument("--features", required=True, help="Path to the features file (.pkl)")
    parser.add_argument("--labels", default=None, help="Optional path to true labels file (.pkl) for accuracy validation")
    parser.add_argument("--outdir", default="out/predictions", help="Directory to save JSON reports")
    parser.add_argument("--head", type=int, default=10, help="Number of predictions to display in terminal")
    
    args = parser.parse_args()
    run_inference(args.model, args.features, args.labels, args.outdir, args.head)
