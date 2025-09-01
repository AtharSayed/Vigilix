import os
import joblib
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
BASE_DIR = "C:/Users/sayed/Desktop/L&T-Project/Vigilix/data/processed/cicids2017_features"
MODEL_PATH = "C:/Users/sayed/Desktop/L&T-Project/Vigilix/model_output/random_forest.pkl"
USE_SAMPLE = True   # Keep True for faster debugging, False for full dataset
SAMPLE_FRACTION = 0.2

def init_spark():
    return SparkSession.builder \
        .appName("RandomForestIntrusionDetection") \
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

    print("🔹 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,       # number of trees
        max_depth=20,          # prevent overfitting
        class_weight="balanced", # handle imbalance
        n_jobs=-1,             # use all CPU cores
        random_state=42
    )
    model.fit(X_train, y_train)

    print("🔹 Predicting on test set...")
    preds = model.predict(X_test)

    print("🔹 Evaluation Results:")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    print(f"🔹 Saving model to: {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    spark.stop()
    print("✅ Random Forest training completed successfully.")

if __name__ == "__main__":
    main()
