import os
import sys
import json
import joblib
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import Counter, Gauge, start_http_server

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.preprocess import create_preprocessor_pipeline

# Load trained model
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "XGBoost_Tuned.joblib")
model = joblib.load(MODEL_PATH)

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

# Prometheus metrics
REQUESTS_TOTAL = Counter("vigilix_kafka_requests_total", "Total requests from Kafka")
ANOMALIES_TOTAL = Counter("vigilix_kafka_anomalies_total", "Total anomalies detected")
ANOMALY_RATIO = Gauge("vigilix_kafka_anomaly_ratio", "Fraction of anomalies in Kafka stream")

print("✅ Kafka Consumer started. Listening on 'network-traffic'...")

# Start Prometheus metrics server
start_http_server(8001)

total = 0
anomalies = 0

for message in consumer:
    record = message.value
    df = pd.DataFrame([record])

    # Drop leakage cols if present
    df = df.drop(columns=["label", "attack_cat"], errors="ignore")

    pred = model.predict(df)[0]

    total += 1
    REQUESTS_TOTAL.inc()

    if pred == 1:
        anomalies += 1
        ANOMALIES_TOTAL.inc()
        producer.send("anomalies", {"alert": "Anomaly detected", "record": record})

    ANOMALY_RATIO.set(anomalies / total if total > 0 else 0)
    print(f"🔍 Processed record. Prediction: {'Anomaly' if pred == 1 else 'Normal'}")
