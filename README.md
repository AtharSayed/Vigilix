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
├── monitoring/                    # Monitoring stack configuration for model/data pipeline
│   ├── grafana/                   # Grafana setup for data visualization
│   │   ├── datasource.yaml        # Grafana data source configuration (e.g., Prometheus)
│   │   └── vigilix_dashboard.json # Predefined dashboard for model and system metrics
│   └── prometheus/                # Prometheus setup for metrics scraping
│       └── prometheus.yml         # Configuration file for Prometheus scrape jobs
├── results/                       # Output directory for EDA summaries and model evaluations
│   ├── eda/                       # EDA result storage
│   │   └── eda_summary.txt        # Summary of statistical and visual data insights
│   └── models/                    # Model evaluation metrics and performance logs
│       ├── isolationforest_results.txt   # Evaluation results for Isolation Forest
│       ├── RandomForest_results.txt      # Evaluation results for Random Forest
│       ├── XGBoost_results.txt           # Evaluation results for XGBoost
│       └── XGBoost_Tuned_results.txt     # Evaluation results after XGBoost tuning
├── scripts/                       # Scripts to automate environment or service startup
│   ├── start-kafka.bat            # Script to launch Zookeeper, Kafka broker, and topics
│   └── start-prometheus.bat       # NEW: Script to start Prometheus monitoring service
├── src/                           # Core data processing logic and helper utilities
│   ├── eda.py                     # Script to perform Exploratory Data Analysis
│   ├── preprocess.py              # Data cleaning and transformation logic
│   └── utils.py                   # Common helper functions used across modules
├── streaming/                     # Kafka-based streaming components
│   ├── kafka_consumer.py          # Kafka consumer to receive and process streaming data
│   └── kafka_producer.py          # Kafka producer to send data to topics
└── testing/                       # Unit and integration tests
    └── test_app.py                # Tests for model pipeline and app logic

