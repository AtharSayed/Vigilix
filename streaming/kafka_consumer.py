import os, sys, json, time
import joblib, pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import Counter, Gauge, Histogram, start_http_server

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# from src.preprocess import create_preprocessor_pipeline  # keep if you use it

TOPIC_IN   = os.getenv("KAFKA_TOPIC", "vigilix-stream")
TOPIC_OUT  = os.getenv("KAFKA_OUT_TOPIC", "anomalies")
BOOTSTRAP  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
GROUP_ID   = os.getenv("KAFKA_GROUP_ID", "vigilix-consumer")
METRICS_PORT = int(os.getenv("METRICS_PORT", "8001"))

MODEL_PATH   = os.path.join(BASE_DIR, "models", "saved_models", "XGBoost_Tuned.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "saved_models", "XGBoost_Tuned_metrics.json")

print("loading model…")
model = joblib.load(MODEL_PATH)

training_metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        training_metrics = json.load(f)

consumer = KafkaConsumer(
    TOPIC_IN,
    bootstrap_servers=BOOTSTRAP,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id=GROUP_ID,
)

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

REQUESTS_TOTAL    = Counter("vigilix_kafka_requests_total", "Total requests processed from Kafka topic")
ANOMALIES_TOTAL   = Counter("vigilix_kafka_anomalies_total", "Total anomalies detected in Kafka stream")
ANOMALY_RATIO     = Gauge("vigilix_kafka_anomaly_ratio", "Fraction of anomalies in Kafka stream")
INFERENCE_LATENCY = Histogram("vigilix_kafka_inference_latency_seconds", "Inference latency for Kafka consumer")

MODEL_ACCURACY  = Gauge("vigilix_model_accuracy", "Model Accuracy (from training)")
MODEL_PRECISION = Gauge("vigilix_model_precision", "Model Precision (from training)")
MODEL_RECALL    = Gauge("vigilix_model_recall", "Model Recall (from training)")
MODEL_F1        = Gauge("vigilix_model_f1", "Model F1-score (from training)")

MODEL_ACCURACY.set(training_metrics.get("Accuracy", 0))
MODEL_PRECISION.set(training_metrics.get("Precision", 0))
MODEL_RECALL.set(training_metrics.get("Recall", 0))
MODEL_F1.set(training_metrics.get("F1-Score", 0))

print(f"✅ consumer starting: in={TOPIC_IN} out={TOPIC_OUT} bootstrap={BOOTSTRAP}")
start_http_server(METRICS_PORT)

total = 0
anom = 0

for msg in consumer:
    t0 = time.time()
    record = msg.value
    df = pd.DataFrame([record]).drop(columns=["label", "attack_cat"], errors="ignore")

    pred = model.predict(df)[0]
    INFERENCE_LATENCY.observe(time.time() - t0)

    total += 1
    REQUESTS_TOTAL.inc()

    if pred == 1:
        anom += 1
        ANOMALIES_TOTAL.inc()
        producer.send(TOPIC_OUT, {"alert": True, "record": record})

    ANOMALY_RATIO.set(anom / total if total else 0)
