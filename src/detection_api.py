from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from datetime import datetime
import time
import joblib
import os

app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('nids_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('nids_request_latency_seconds', 'Request latency', ['endpoint'])
PREDICTION_SCORE = Gauge('nids_prediction_score', 'Prediction confidence score')
ANOMALY_COUNT = Counter('nids_anomalies_total', 'Total anomalies detected')
BENIGN_COUNT = Counter('nids_benign_total', 'Total benign traffic detected')
PREDICTION_TIME = Histogram('nids_prediction_time_seconds', 'Time taken for prediction')

# Model path
MODEL_PATH = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\model_output\random_forest_model.joblib"

# Load Random Forest model
print("Loading Random Forest model...")
try:
    model: RandomForestClassifier = joblib.load(MODEL_PATH)
    print("✅ Random Forest model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load Random Forest model: {e}")
    raise RuntimeError("Failed to load Random Forest model")

print("ℹ️  Model trained on raw features - no scaling applied")

def predict_with_model(features: pd.DataFrame):
    """Make prediction using Random Forest"""
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]  # Returns [P(class_0), P(class_1)]
    return prediction, prediction_proba


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    endpoint = '/predict'
    
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            REQUEST_COUNT.labels('POST', endpoint, '400').inc()
            return jsonify({'error': 'Missing features in request'}), 400
        
        features = pd.DataFrame([data['features']])
        
        prediction_start = time.time()
        prediction, prediction_proba = predict_with_model(features)
        PREDICTION_TIME.observe(time.time() - prediction_start)
        
        confidence = np.max(prediction_proba)
        PREDICTION_SCORE.set(confidence)
        
        # Update counters
        if prediction == 1:
            ANOMALY_COUNT.inc()
            status = 'attack'
        else:
            BENIGN_COUNT.inc()
            status = 'benign'
        
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint).observe(latency)
        REQUEST_COUNT.labels('POST', endpoint, '200').inc()
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(confidence),
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'processing_time': latency,
            'scaler_used': False  # No scaling applied
        })
        
    except Exception as e:
        REQUEST_COUNT.labels('POST', endpoint, '500').inc()
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    return generate_latest(REGISTRY)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'model_loaded': True,
        'scaler_used': False,
        'model_type': 'random_forest'
    })

if __name__ == '__main__':
    print(f"Model path: {MODEL_PATH}")
    print("✅ API ready - using Random Forest on raw features (no scaling)")
    app.run(host='0.0.0.0', port=5000, debug=False)
