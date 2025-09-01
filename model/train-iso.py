import os
import joblib
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
BASE_DIR = "C:/Users/sayed/Desktop/L&T-Project/Vigilix/data/processed/cicids2017_features"
MODEL_PATH = "C:/Users/sayed/Desktop/L&T-Project/Vigilix/models/isolation_forest.pkl"
USE_SAMPLE = True  # Set to False if you want to use full data (may crash with large sets)
SAMPLE_FRACTION = 0.2  # 20% of data to avoid Spark memory crash

def init_spark():
    return SparkSession.builder \
        .appName("IsolationForestAnomalyDetection") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

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

def main():
    spark = init_spark()
    print("🔹 Reading training & testing datasets...")

    train_df = load_data(spark, os.path.join(BASE_DIR, "train"))
    test_df = load_data(spark, os.path.join(BASE_DIR, "test"))

    print("🔹 Converting Spark DataFrames to NumPy arrays...")
    X_train, y_train = convert_to_numpy(train_df)
    X_test, y_test = convert_to_numpy(test_df)

    # Convert to binary labels for anomaly detection (0 = normal, 1 = anomaly)
    y_train_binary = np.where(y_train == 0, 0, 1)
    y_test_binary = np.where(y_test == 0, 0, 1)

    print("🔹 Training Isolation Forest model...")
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_train)

    print("🔹 Predicting anomalies...")
    preds = model.predict(X_test)
    preds = np.where(preds == 1, 0, 1)  # Convert IsolationForest output to 0 = normal, 1 = anomaly

    print("🔹 Evaluation Results:")
    print(confusion_matrix(y_test_binary, preds))
    print(classification_report(y_test_binary, preds))

    print(f"🔹 Saving model to: {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    spark.stop()
    print("✅ Training completed successfully.")

if __name__ == "__main__":
    main()
