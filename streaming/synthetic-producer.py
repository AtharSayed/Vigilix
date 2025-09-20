import json
import time
import random
from kafka import KafkaProducer

# Initializing the  Kafka Producer
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# A base template for a normal network log record. We will randomize some of these fields to create variety.
BASE_RECORD = {
    'dur': 0.000011, 'proto': 'udp', 'service': '-', 'state': 'INT', 'spkts': 2,
    'dpkts': 0, 'sbytes': 104, 'dbytes': 0, 'rate': 90909.0902, 'sttl': 254,
    'dttl': 0, 'sload': 37818181.82, 'dload': 0.0, 'sloss': 0, 'dloss': 0,
    'sinpkt': 0.011, 'dinpkt': 0.0, 'sjit': 0.0, 'djit': 0.0, 'swin': 0,
    'stcpb': 0, 'dtcpb': 0, 'dwin': 0, 'tcprtt': 0.0, 'synack': 0.0,
    'ackdat': 0.0, 'smean': 52, 'dmean': 0, 'trans_depth': 0,
    'response_body_len': 0, 'ct_srv_src': 2, 'ct_state_ttl': 2,
    'ct_dst_ltm': 1, 'ct_src_dport_ltm': 1, 'ct_dst_sport_ltm': 1,
    'ct_dst_src_ltm': 1, 'is_ftp_login': 0, 'ct_ftp_cmd': 0,
    'ct_flw_http_mthd': 0, 'ct_src_ltm': 1, 'ct_srv_dst': 2, 'is_sm_ips_ports': 0
}

def generate_synthetic_record():
    record = BASE_RECORD.copy()

    # --- Introducing  random variations to simulate real traffic ---

    # Vary some numerical features
    record['dur'] = max(0, record['dur'] + random.uniform(-0.000005, 0.000005))
    record['sbytes'] = record['sbytes'] + random.randint(-20, 20)
    record['rate'] = max(0, record['rate'] + random.uniform(-10000, 10000))
    record['sload'] = max(0, record['sload'] + random.uniform(-1000000, 1000000))
    record['smean'] = max(0, record['smean'] + random.randint(-10, 10))

    # Occasionally, create a potential anomaly with high packet counts
    if random.random() < 0.1: # 10% chance of being an "attack" type
        record['proto'] = 'tcp'
        record['spkts'] = random.randint(50, 200)
        record['sbytes'] = random.randint(10000, 50000)
        record['rate'] = random.uniform(500, 2500)
        record['service'] = 'dns'
    else: # 90% chance of being a "normal" type
        record['proto'] = random.choice(['udp', 'tcp', 'arp'])
        record['service'] = random.choice(['-', 'dns', 'http'])

    return record

def main():
    """Main function to generate and send data continuously."""
    print("🚀 Starting synthetic data producer...")
    try:
        while True:
            # 1. Generate a new synthetic record
            record = generate_synthetic_record()

            # 2. Send it to the Kafka topic
            producer.send("vigilix-stream", record)
            producer.flush() # Ensure message is sent immediately

            print(f"📤 Sent record with proto={record['proto']} and spkts={record['spkts']}")

            # 3. Wait for a short period to simulate a real-time stream
            time.sleep(1) # Send one record per second

    except KeyboardInterrupt:
        print("\n🛑 Producer stopped.")

if __name__ == "__main__":
    main()