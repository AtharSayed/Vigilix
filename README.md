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
├── requirements.txt               # Python dependencies
├── models/                        # Model training & inference scripts
│   ├── app.py                     # Main entry point to run models
│   ├── hyper-xgb.py               # XGBoost tuning
│   ├── isolation-forest.py        # Isolation Forest implementation
│   ├── random-forest.py           # Random Forest implementation
│   └── xgboost_model.py           # XGBoost implementation
├── monitoring/
│   └── grafana/
│       └── vigilix_dashboard.json # Grafana dashboard config
├── results/
│   ├── eda/
│   │   └── eda_summary.txt        # EDA output
│   └── models/
│       ├── isolationforest_results.txt
│       ├── RandomForest_results.txt
│       ├── XGBoost_results.txt
│       └── XGBoost_Tuned_results.txt
├── src/                           # Core scripts
│   ├── eda.py                     # Exploratory data analysis
│   ├── preprocess.py              # Data cleaning and transformation
│   └── utils.py                   # Helper functions
└── testing/                       # Unit tests
    └── test_app.py                # Tests for app.py and models
