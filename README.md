# Vigilix

Vigilix is an advanced network intrusion detection system (NIDS) designed to identify and classify various types of network attacks, including DDoS, DoS, and more, using machine learning models. The system is built to process large-scale network traffic data, with a primary focus on the CIC-IDS 2017 dataset.

The project leverages powerful and scalable machine learning algorithms to achieve high-accuracy detection and provide insights into the nature of network threats.

---

### 🚀 Project Structure

The repository is organized to separate different stages of the machine learning pipeline: data, models, results, and scripts.

```

Vigilix/
├── data/                  \# Stores raw and processed network traffic data
│   └── raw/
│       └── cic-collection.parquet/
├── model/                 \# Contains model training scripts and notebooks
│   ├── train-iso.py       \# Training script for Isolation Forest
│   └── train-xgb.py       \# Training script for XGBoost (from original repo)
├── model\_output/          \# Directory to save trained models
├── results/               \# Directory to save evaluation reports and plots
├── .gitignore             \# Git ignore file
└── README.md              \# This file

````

---

### ✨ Features

* **Supervised Anomaly Detection**: Uses a robust XGBoost model to classify network traffic with high accuracy.
* **Unsupervised Anomaly Detection**: Includes an Isolation Forest model as an alternative for unsupervised detection.
* **Performance Evaluation**: Comprehensive evaluation using metrics like **ROC-AUC**, **PR-AUC**, and **Classification Reports** to assess model effectiveness.
* **Feature Importance Analysis**: Identifies the most critical network features contributing to attack detection.
* **Scalable Data Handling**: Designed to process large datasets like the CIC-IDS 2017 collection efficiently.

---

### ⚙️ Technologies

* **Python**: The core programming language.
* **scikit-learn**: For Isolation Forest and evaluation metrics.
* **XGBoost**: For the supervised learning model.
* **pandas**: For data manipulation and analysis.
* **matplotlib**: For data visualization and plotting.
* **joblib**: For saving and loading machine learning models.

---

### 📝 Getting Started

#### Prerequisites

* Python 3.8+
* pip

#### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/AtharSayed/Vigilix.git](https://github.com/AtharSayed/Vigilix.git)
    cd Vigilix
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    (Note: You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt`).

#### Usage

1.  **Place your dataset**: Ensure your `cic-collection.parquet` file is placed in the `data/raw/` directory.

2.  **Train a model**:
    * To train the **XGBoost** model:
        ```bash
        python model/train-xgb.py
        ```
    * To train the **Isolation Forest** model:
        ```bash
        python model/train-iso.py
        ```

3.  **View results**:
    * Trained models will be saved in the `model_output/` directory.
    * Evaluation reports and plots will be saved in the `results/` directory.

---

### 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/AtharSayed/Vigilix/issues).

---

### 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

### 📧 Contact

Athar Sayed - [sayedathar242@gmail.com]
Project Link: [https://github.com/AtharSayed/Vigilix](https://github.com/AtharSayed/Vigilix)
````
