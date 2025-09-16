# 🛡️ Vigilix – Intrusion Detection & Monitoring System

Welcome to **Vigilix**, an intelligent anomaly detection pipeline built using classical ML models with support for monitoring via Grafana and CI/CD integration. This project is modular, scalable, and designed for deployment in production environments.

---

## 📚 Overview

**Vigilix** leverages multiple anomaly detection models (Isolation Forest, Random Forest, XGBoost) to identify irregular patterns in data. It includes:

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Preprocessing utilities
- ⚙️ Model training and evaluation
- 🧪 Unit testing for critical components
- 📈 Monitoring with Grafana dashboards
- 🔄 CI/CD compatibility for automation

---

## 🧭 Project Structure

```bash
atharsayed-vigilix/
├── README.md                      # Project overview, setup instructions, and usage guide
├── requirements.txt              # Python dependencies for the project
├── models/                        # Model training, tuning, and inference scripts
│   ├── app.py                     # Main entry point to run and evaluate models
│   ├── hyper-xgb.py               # Hyperparameter tuning for XGBoost
│   ├── isolation-forest.py        # Isolation Forest anomaly detection implementation
│   ├── random-forest.py           # Random Forest classification model
│   └── xgboost_model.py           # XGBoost classification model
├── monitoring/                    # Monitoring stack configuration
│   ├── grafana/                   # Grafana dashboard setup
│   │   ├── datasource.yaml        # Grafana data source configuration
│   │   └── vigilix_dashboard.json # Predefined Grafana dashboard layout
│   └── prometheus/                # Prometheus configuration
│       └── prometheus.yml         # Prometheus scrape targets and settings
├── results/                       # Output results from EDA and model evaluations
│   ├── eda/                       # Exploratory Data Analysis outputs
│   │   └── eda_summary.txt        # Summary statistics and insights from EDA
│   └── models/                    # Model performance metrics
│       ├── isolationforest_results.txt   # Results from Isolation Forest
│       ├── RandomForest_results.txt      # Results from Random Forest
│       ├── XGBoost_results.txt           # Results from XGBoost
│       └── XGBoost_Tuned_results.txt     # Results from tuned XGBoost
├── scripts/                       # Automation and environment setup scripts
│   └── start-kafka.bat            # Script to start Zookeeper, Kafka broker, and create topics
├── src/                           # Core data processing and utility scripts
│   ├── eda.py                     # Script for performing EDA
│   ├── preprocess.py              # Data cleaning and preprocessing logic
│   └── utils.py                   # Common helper functions used across modules
├── streaming/                     # Kafka streaming components
│   ├── kafka_consumer.py          # Kafka consumer to ingest and process data
│   └── kafka_producer.py          # Kafka producer to send data to topics
└── testing/                       # Unit and integration tests
    └── test_app.py                # Tests for model pipeline and app logic

