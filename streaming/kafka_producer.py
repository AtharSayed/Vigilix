import os
import json
import time
import pandas as pd
from kafka import KafkaProducer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "UNSW_NB15_testing-set.parquet") # Real Time simulation from Test Data

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def main():
    print("🚀 Kafka Producer started. Sending logs to topic: vigilix-stream")

    df = pd.read_parquet(DATA_PATH)

    for _, row in df.iterrows():
        record = row.to_dict()
        producer.send("vigilix-stream", record)
        print(f"📤 Sent: {record}")
        time.sleep(0.5)  # simulate real-time flow

if __name__ == "__main__":
    main()
