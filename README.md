# 🛡️ Vigilix – Intrusion Detection & Monitoring System

Welcome to **Vigilix**, an intelligent anomaly detection pipeline designed to identify and monitor network intrusions in real-time. This project leverages machine learning models, real-time data streaming, and monitoring tools to provide a robust and scalable solution for anomaly detection.

---

## 📚 Overview

**Vigilix** is a modular and production-ready system that combines classical machine learning models with real-time monitoring capabilities. It is designed to process network data, detect anomalies, and visualize system performance metrics in real-time.

### Key Features:
- **Anomaly Detection Models:** Implements Isolation Forest, Random Forest, and XGBoost for anomaly detection.
- **Real-Time Data Streaming:** Uses Kafka for real-time data ingestion and processing.
- **Monitoring Stack:** Integrates Prometheus and Grafana for real-time metrics visualization.
- **Exploratory Data Analysis (EDA):** Provides insights into the dataset with statistical summaries and visualizations.
- **Preprocessing Utilities:** Includes scripts for data cleaning, transformation, and feature engineering.
- **Model Evaluation:** Tracks model performance with confusion matrices, feature importance, and evaluation metrics.
- **CI/CD Ready:** Compatible with CI/CD pipelines for automated testing and deployment.

---

## 🖼️ System Architecture

### Network Intrusion Detection & Monitoring System Architecture
![System Architecture](results/images/System-Design.png)

This architecture illustrates the Vigilix pipeline:
1. **Data Ingestion:** Raw network data is ingested into Kafka topics.
2. **Data Processing:** Data is preprocessed and streamed to machine learning models for inference.
3. **Anomaly Detection:** Models classify data as normal or anomalous.
4. **Monitoring:** Prometheus collects metrics, and Grafana visualizes them in real-time dashboards.

---

## 🖥️ Dashboard Example

### Vigilix Kafka Real-Time Monitoring Dashboard
![Dashboard Screenshot](results/images/Sample-Dashboard-Screenshot.png)

The dashboard provides real-time insights, including:
- Total network requests processed.
- Number of anomalies detected.
- Anomaly detection ratio.
- Model performance metrics (e.g., accuracy, precision, recall).

---

## 🧭 Project Structure

```bash
Vigilix/
├── README.md                      # Project overview and usage guide
├── requirements.txt               # Python dependencies
├── docker-compose.yml             # Docker Compose file for service orchestration
├── models/                        # Model training, tuning, and inference scripts
│   ├── app.py                     # Main entry point for model evaluation
│   ├── isolation-forest.py        # Isolation Forest implementation
│   ├── random-forest.py           # Random Forest implementation
│   ├── xgboost_model.py           # XGBoost implementation
│   └── saved_models/              # Pre-trained models
├── monitoring/                    # Monitoring stack configuration
│   ├── grafana/                   # Grafana setup
│   └── prometheus/                # Prometheus setup
├── results/                       # EDA summaries and model evaluation results
├── scripts/                       # Scripts to start Kafka and Prometheus
├── src/                           # Core data processing and orchestration logic
│   ├── main.py                    # Main orchestrator script
│   ├── preprocess.py              # Data preprocessing logic
│   └── utils.py                   # Helper functions
├── streaming/                     # Kafka producer and consumer scripts
└── testing/                       # Unit and integration tests
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Docker and Docker Compose
- Java Runtime Environment (for Kafka)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vigilix.git
   cd vigilix
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the services using Docker Compose:
   ```bash
   docker-compose up
   ```

4. Start the orchestrator:
   ```bash
   python src/main.py
   ```

---

## 🛠️ Components

### 1. **Data Pipeline**
- **Kafka Producer:** Streams synthetic or real network data to Kafka topics.
- **Kafka Consumer:** Consumes data from Kafka topics and sends it to the ML models for inference.

### 2. **Machine Learning Models**
- **Isolation Forest:** Unsupervised anomaly detection model.
- **Random Forest:** Supervised classification model.
- **XGBoost:** Gradient boosting model for high-performance classification.

### 3. **Monitoring Stack**
- **Prometheus:** Collects metrics from the pipeline and models.
- **Grafana:** Visualizes metrics in real-time dashboards.

---

## 📊 Results

### Exploratory Data Analysis (EDA)
- **Correlation Heatmap:** Visualizes relationships between features.
- **Label Distribution:** Shows the distribution of normal vs. anomalous data.
- **Attack Category Distribution:** Highlights the types of attacks in the dataset.

### Model Evaluation
- **Confusion Matrices:** Evaluate model performance.
- **Feature Importance:** Identify the most important features for classification.
- **Metrics:** Accuracy, precision, recall, and F1-score.

---

## 🧪 Testing

### Unit Tests
Run unit tests to ensure the integrity of the pipeline:
```bash
pytest testing/
```

### Test Coverage
- **Model Pipeline:** Tests for data preprocessing, model inference, and evaluation.
- **Streaming Components:** Tests for Kafka producer and consumer.

---

## 📈 Future Enhancements

- **Model Registry:** Implement a model registry (e.g., MLflow) to track model versions and metrics.
- **Data Validation:** Add schema validation for incoming data.
- **Scalability:** Deploy the pipeline on Kubernetes for horizontal scaling.
- **Advanced Models:** Integrate deep learning models for anomaly detection.
- **Security:** Add authentication and authorization for APIs and dashboards.

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

---

## 📞 Support

For any questions or issues, please contact:
- **Name:** Your Name
- **Email:** your.email@example.com
- **GitHub:** [your-username](https://github.com/your-username)

---

## 🌟 Acknowledgments

- **Kafka:** For real-time data streaming.
- **Prometheus & Grafana:** For monitoring and visualization.
- **Scikit-learn & XGBoost:** For machine learning models.
- **UNSW-NB15 Dataset:** For providing the dataset used in this project.
