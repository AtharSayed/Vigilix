# models/app.py
import os
import time
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry, multiprocess

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "XGBoost_Tuned.joblib")  # Added Fine tuned XGB joblib (Best Performance compared with others)
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8000"))

# Create app
app = Flask(__name__)

# Load model pipeline (preprocessor + classifier) saved earlier
if os.path.exists(MODEL_PATH):
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
else:
    model_pipeline = None
    print(f"No model found at {MODEL_PATH}; /predict will return error until you save a model here.")

# Prometheus metrics
REQUEST_COUNTER = Counter("vigilix_requests_total", "Total requests to model", ["endpoint", "method", "status"])
INFERENCE_COUNTER = Counter("vigilix_inferences_total", "Total inferences made")
INFERENCE_LATENCY = Histogram("vigilix_inference_latency_seconds", "Inference latency seconds")
MODEL_ACCURACY = Gauge("vigilix_model_accuracy", "Model Accuracy (set from training)")
MODEL_PRECISION = Gauge("vigilix_model_precision", "Model Precision (set from training)")
MODEL_RECALL = Gauge("vigilix_model_recall", "Model Recall (set from training)")
MODEL_F1 = Gauge("vigilix_model_f1", "Model F1-score (set from training)")

# Optionally start a simple metrics server for multiprocess mode if needed
# For this simple deployment, we will expose /metrics from Flask itself.

@app.route("/health")
def health():
    REQUEST_COUNTER.labels(endpoint="/health", method="GET", status="200").inc()
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    try:
        payload = request.get_json(force=True)
        # Expecting payload to be a dict or list of dicts representing rows
        if isinstance(payload, dict):
            X = pd.DataFrame([payload])
        else:
            X = pd.DataFrame(payload)

        if model_pipeline is None:
            REQUEST_COUNTER.labels(endpoint="/predict", method="POST", status="500").inc()
            return jsonify({"error": "Model not loaded"}), 500

        preds = model_pipeline.predict(X)
        proba = None
        if hasattr(model_pipeline, "predict_proba"):
            try:
                proba = model_pipeline.predict_proba(X).tolist()
            except Exception:
                proba = None

        INFERENCE_COUNTER.inc()
        latency = time.time() - start
        INFERENCE_LATENCY.observe(latency)
        REQUEST_COUNTER.labels(endpoint="/predict", method="POST", status="200").inc()

        return jsonify({"predictions": preds.tolist(), "probabilities": proba, "latency": latency}), 200
    except Exception as e:
        REQUEST_COUNTER.labels(endpoint="/predict", method="POST", status="500").inc()
        return jsonify({"error": str(e)}), 500

@app.route("/metrics")
def metrics():
    # Expose important gauges and counters - use generate_latest
    # We rely on prometheus_client's default CollectorRegistry
    data = generate_latest()
    return data, 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # If you want a separate metrics server, start it here (but Flask /metrics is enough)
    # start_http_server(METRICS_PORT)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
