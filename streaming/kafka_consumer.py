import os
import sys
import json
import time
import joblib
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.preprocess import create_preprocessor_pipeline

# -------------------------------
# Load model & training metrics
# -------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "XGBoost_Tuned.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "saved_models", "XGBoost_Tuned_metrics.json")

print("📦 Loading model...")
model = joblib.load(MODEL_PATH)

print("📦 Loading training metrics...")
training_metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        training_metrics = json.load(f)

# -------------------------------
# Kafka setup
# -------------------------------
# Kafka consumer: listens to incoming logs
consumer = KafkaConsumer(
    "vigilix-stream",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

# Kafka producer: sends anomalies to another topic
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# -------------------------------
# Prometheus Metrics
# -------------------------------
# Live metrics from the Kafka stream
REQUESTS_TOTAL = Counter("vigilix_kafka_requests_total", "Total requests processed from Kafka topic")
ANOMALIES_TOTAL = Counter("vigilix_kafka_anomalies_total", "Total anomalies detected in Kafka stream")
ANOMALY_RATIO = Gauge("vigilix_kafka_anomaly_ratio", "Fraction of anomalies in Kafka stream")
INFERENCE_LATENCY = Histogram("vigilix_kafka_inference_latency_seconds", "Inference latency for Kafka consumer")

# Static metrics from the trained model (consistent with app.py)
MODEL_ACCURACY = Gauge("vigilix_model_accuracy", "Model Accuracy (from training)")
MODEL_PRECISION = Gauge("vigilix_model_precision", "Model Precision (from training)")
MODEL_RECALL = Gauge("vigilix_model_recall", "Model Recall (from training)")
MODEL_F1 = Gauge("vigilix_model_f1", "Model F1-score (from training)")

# Initialize gauges with values from training metrics file
MODEL_ACCURACY.set(training_metrics.get("Accuracy", 0))
MODEL_PRECISION.set(training_metrics.get("Precision", 0))
MODEL_RECALL.set(training_metrics.get("Recall", 0))
MODEL_F1.set(training_metrics.get("F1-Score", 0))

# -------------------------------
# Main consumer loop
# -------------------------------
print("✅ Kafka Consumer started. Listening on 'vigilix-stream'...")

# Start Prometheus metrics server on a different port than the main app
start_http_server(8001)

total_processed = 0
anomalies_detected = 0

for message in consumer:
    start_time = time.time()
    
    record = message.value
    df = pd.DataFrame([record])

    # Drop leakage columns if they exist
    df = df.drop(columns=["label", "attack_cat"], errors="ignore")

    # Make prediction
    pred = model.predict(df)[0]

    # Observe latency
    latency = time.time() - start_time
    INFERENCE_LATENCY.observe(latency)

    # Increment counters
    total_processed += 1
    REQUESTS_TOTAL.inc()

    if pred == 1:
        anomalies_detected += 1
        ANOMALIES_TOTAL.inc()
        # Send detected anomaly to the 'anomalies' topic
        producer.send("anomalies", {"alert": "Anomaly detected", "record": record})

    # Update anomaly ratio gauge
    ANOMALY_RATIO.set(anomalies_detected / total_processed if total_processed > 0 else 0)
    
    print(f"🔍 Processed record in {latency:.4f}s. Prediction: {'Anomaly' if pred == 1 else 'Normal'}")