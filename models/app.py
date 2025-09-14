import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# -------------------------------
# Flask app setup
# -------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "XGBoost_Tuned.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "saved_models", "XGBoost_Tuned_metrics.json")

# -------------------------------
# Load model & metrics
# -------------------------------
print("📦 Loading model...")
model = joblib.load(MODEL_PATH)

print("📦 Loading training metrics...")
training_metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        training_metrics = json.load(f)

# -------------------------------
# Prometheus Metrics
# -------------------------------
REQUEST_COUNTER = Counter(
    "vigilix_requests_total",
    "Total requests to model",
    ["endpoint", "method", "status"]
)

INFERENCE_COUNTER = Counter(
    "vigilix_inferences_total",
    "Total inferences made"
)

INFERENCE_LATENCY = Histogram(
    "vigilix_inference_latency_seconds",
    "Inference latency seconds"
)

# Training metrics (gauges loaded from JSON)
MODEL_ACCURACY = Gauge("vigilix_model_accuracy", "Model Accuracy (set from training)")
MODEL_PRECISION = Gauge("vigilix_model_precision", "Model Precision (set from training)")
MODEL_RECALL = Gauge("vigilix_model_recall", "Model Recall (set from training)")
MODEL_F1 = Gauge("vigilix_model_f1", "Model F1-score (set from training)")

# Live anomaly detection ratio
ANOMALY_RATIO = Gauge("vigilix_anomaly_ratio", "Fraction of predictions detected as anomalies")

# Initialize gauges from training metrics
MODEL_ACCURACY.set(training_metrics.get("Accuracy", 0))
MODEL_PRECISION.set(training_metrics.get("Precision", 0))
MODEL_RECALL.set(training_metrics.get("Recall", 0))
MODEL_F1.set(training_metrics.get("F1-Score", 0))

# -------------------------------
# Routes
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        data = request.get_json()
        if not data:
            REQUEST_COUNTER.labels(endpoint="/predict", method="POST", status="400").inc()
            return jsonify({"error": "No input data provided"}), 400

        # Convert input into DataFrame
        X_input = pd.DataFrame([data])

        # Run prediction
        preds = model.predict(X_input)
        preds = preds.astype(int).tolist()

        latency = time.time() - start_time
        INFERENCE_LATENCY.observe(latency)
        INFERENCE_COUNTER.inc()
        REQUEST_COUNTER.labels(endpoint="/predict", method="POST", status="200").inc()

        # Update anomaly ratio
        anomaly_ratio = np.mean(preds)
        ANOMALY_RATIO.set(anomaly_ratio)

        return jsonify({"predictions": preds})

    except Exception as e:
        REQUEST_COUNTER.labels(endpoint="/predict", method="POST", status="500").inc()
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/")
def home():
    return jsonify({"message": "Vigilix IDS API is running!"})


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
