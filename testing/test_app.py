import unittest
import json
import sys
import os

# Adding the root path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# Import from models package
from models.app import app

# Sample payloads
non_attack_payload_1 = {
    "proto": "udp", "service": "-", "state": "INT", "dur": 0.0,
    "sbytes": 0, "dbytes": 0, "sttl": 1, "dttl": 1, "spkts": 1,
    "dpkts": 0, "sload": 0.0, "dload": 0.0, "stcpb": 0, "dtcpb": 0
}

non_attack_payload_2 = {
    "proto": "tcp", "service": "http", "state": "ESTABLISHED", "dur": 12.5,
    "sbytes": 1500, "dbytes": 1300, "sttl": 64, "dttl": 64, "spkts": 30,
    "dpkts": 28, "sload": 120.4, "dload": 110.2, "stcpb": 123456, "dtcpb": 654321
}

attack_payload_1 = {
    "proto": "tcp", "service": "ftp", "state": "S0", "dur": 0.1,
    "sbytes": 10, "dbytes": 0, "sttl": 10, "dttl": 0, "spkts": 2,
    "dpkts": 0, "sload": 0.01, "dload": 0.0, "stcpb": 0, "dtcpb": 0
}

attack_payload_2 = {
    "proto": "udp", "service": "-", "state": "S1", "dur": 0.05,
    "sbytes": 3, "dbytes": 1, "sttl": 1, "dttl": 1, "spkts": 1,
    "dpkts": 1, "sload": 0.0, "dload": 0.0, "stcpb": 0, "dtcpb": 0
}

class TestVigilixAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_predict_valid_payload(self):
        response = self.client.post("/predict", data=json.dumps(non_attack_payload_1),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("predictions", data)
        self.assertIsInstance(data["predictions"], list)
        self.assertEqual(len(data["predictions"]), 1)

    def test_predict_non_attack_2(self):
        response = self.client.post("/predict", data=json.dumps(non_attack_payload_2),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("predictions", data)
        self.assertIsInstance(data["predictions"], list)
        self.assertEqual(len(data["predictions"]), 1)

    def test_predict_attack_1(self):
        response = self.client.post("/predict", data=json.dumps(attack_payload_1),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("predictions", data)
        self.assertIsInstance(data["predictions"], list)
        self.assertEqual(len(data["predictions"]), 1)

    def test_predict_attack_2(self):
        response = self.client.post("/predict", data=json.dumps(attack_payload_2),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("predictions", data)
        self.assertIsInstance(data["predictions"], list)
        self.assertEqual(len(data["predictions"]), 1)

    def test_predict_empty_payload(self):
        response = self.client.post("/predict", data=json.dumps({}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_home_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.get_json())

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"# HELP", response.data)

if __name__ == "__main__":
    unittest.main()
