import os
import joblib
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# ========================
# Configuration
# ========================
BASE_DIR = "C:/Users/sayed/Desktop/L&T-Project/Vigilix/data/processed/cicids2017_features"
MODEL_PATH = "C:/Users/sayed/Desktop/L&T-Project/Vigilix/model_output/isolation_forest_tuned.pkl"
USE_SAMPLE = False       # Avoid OOM, set to False for full data
SAMPLE_FRACTION = 0.2   # 20% sample (tune as needed)

# ========================
# Spark Init
# ========================
def init_spark():
    return (
        SparkSession.builder
        .appName("IsolationForestAnomalyDetection")
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

# ========================
# Data Load
# ========================
def load_data(spark, path):
    df = spark.read.parquet(path)
    if USE_SAMPLE:
        df = df.sample(fraction=SAMPLE_FRACTION, seed=42)
    return df.withColumn("features_array", vector_to_array(col("features")))

def convert_to_numpy(df):
    pdf = df.select("features_array", "label_index").toPandas()
    X = np.vstack(pdf["features_array"].to_numpy())
    y = pdf["label_index"].to_numpy()
    return X, y

# ========================
# Main Training
# ========================
def main():
    spark = init_spark()
    print("🔹 Reading training & testing datasets...")

    train_df = load_data(spark, os.path.join(BASE_DIR, "train"))
    test_df = load_data(spark, os.path.join(BASE_DIR, "test"))

    print("🔹 Converting Spark DataFrames to NumPy arrays...")
    X_train, y_train = convert_to_numpy(train_df)
    X_test, y_test = convert_to_numpy(test_df)

    # Binary labels (0 = normal, 1 = anomaly/attack)
    y_train_binary = np.where(y_train == 0, 0, 1)
    y_test_binary = np.where(y_test == 0, 0, 1)

    print("🔹 Training tuned Isolation Forest model...")

    model = IsolationForest(
        n_estimators=500,       # more trees → stability
        max_samples=0.8,        # sub-sample for diversity
        contamination=0.02,     # expected anomaly rate
        max_features=1.0,       # use all features
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    print("🔹 Predicting anomalies...")
    preds = model.predict(X_test)
    preds = np.where(preds == 1, 0, 1)  # Map {1=normal, -1=anomaly} → {0=normal, 1=attack}

    print("\n🎯 Evaluation Results (Tuned Isolation Forest):")
    print(confusion_matrix(y_test_binary, preds))
    print(classification_report(y_test_binary, preds, target_names=["Normal", "Attack"]))

    print(f"🔹 Saving model to: {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    spark.stop()
    print("✅ Training completed successfully.")

# ========================
# Run
# ========================
if __name__ == "__main__":
    main()


# model result: (Isolation Forest) (Training results from  20% Dataset)

# | Metric        | Normal (0) | Attack (1) |
# | ------------- | ---------- | ---------- |
# | **Precision** | 0.82       | 0.46       |
# | **Recall**    | 0.97       | 0.12       |
# | **F1-score**  | 0.89       | 0.19       |
# | **Support**   | 91,428     | 22,079     |



# model result: (Isolation Forest) (Training results from  100% Dataset)
# | Class        | Precision | Recall | F1-score | Support |
# | ------------ | --------- | ------ | -------- | ------- |
# | Normal       | 0.80      | 0.98   | 0.88     | 454,721 |
# | Attack       | 0.26      | 0.03   | 0.05     | 111,494 |
# | **Accuracy** |           |        | 0.79     | 566,215 |
