import os, json, time, random
from kafka import KafkaProducer

TOPIC = os.getenv("KAFKA_TOPIC", "vigilix-stream")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

BASE = {
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

def make_record():
    r = dict(BASE)
    r['dur'] = max(0, r['dur'] + random.uniform(-0.000005, 0.000005))
    r['sbytes'] = r['sbytes'] + random.randint(-20, 20)
    r['rate'] = max(0, r['rate'] + random.uniform(-10000, 10000))
    r['sload'] = max(0, r['sload'] + random.uniform(-1_000_000, 1_000_000))
    r['smean'] = max(0, r['smean'] + random.randint(-10, 10))
    if random.random() < 0.1:
        r['proto'] = 'tcp'
        r['spkts'] = random.randint(50, 200)
        r['sbytes'] = random.randint(10_000, 50_000)
        r['rate'] = random.uniform(500, 2500)
        r['service'] = 'dns'
    else:
        r['proto'] = random.choice(['udp', 'tcp', 'arp'])
        r['service'] = random.choice(['-', 'dns', 'http'])
    return r

def main():
    print(f"[producer] topic={TOPIC} bootstrap={BOOTSTRAP}")
    sleep_s = float(os.getenv("PRODUCER_SLEEP_SEC", "1"))
    try:
        while True:
            rec = make_record()
            producer.send(TOPIC, rec)
            producer.flush()
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("producer stopped")

if __name__ == "__main__":
    main()
