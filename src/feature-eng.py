import os
import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

# File paths
BASE_DIR = "C:/Users/sayed/Desktop/L&T-Project/Vigilix"
INPUT_FILE = os.path.join(BASE_DIR, "data/processed/cicids2017_preprocessed.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/cicids2017_features")

def clean_colname(col_name: str) -> str:
    """Cleans column names for Spark compatibility."""
    col_name = col_name.strip()
    col_name = col_name.replace(" ", "_").replace("/", "_").replace(".", "_")
    return re.sub(r"[^a-zA-Z0-9_]", "", col_name)

def main():
    # 🔧 Spark configuration to increase memory
    spark = SparkSession.builder \
        .appName("FeatureEngineering") \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "6g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.default.parallelism", "8") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print("🔹 Loading preprocessed dataset...")
    df = spark.read.parquet(INPUT_FILE)
    print(f"✅ Dataset loaded with {df.count()} rows and {len(df.columns)} columns")

    # 🧹 Clean column names
    df = df.toDF(*[clean_colname(c) for c in df.columns])

    # 🎯 Encode label
    label_col = "Label"  # Change if your label column is named differently
    indexer = StringIndexer(inputCol=label_col, outputCol="label_index")
    df = indexer.fit(df).transform(df)
    print("✅ Label column encoded")

    # ⚙️ Assemble features
    feature_cols = [col for col in df.columns if col not in [label_col, "label_index"]]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    df = assembler.transform(df)

    # 📏 Scale features
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    df = scaler.fit(df).transform(df)
    print(f"✅ Features assembled & scaled: {len(feature_cols)} columns")

    # ✂️ Keep only necessary columns
    df = df.select("features", "label_index")

    # 🔎 Show label distribution
    print("🔎 Label distribution after feature engineering:")
    df.groupBy("label_index").count().show()

    # 🔄 Repartition to reduce memory per partition
    df = df.repartition(8)

    # 💾 Save to Parquet
    print("💾 Saving processed data to Parquet...")
    df.write.mode("overwrite").parquet(OUTPUT_DIR)
    print("✅ Feature-engineered data saved.")

    spark.stop()

if __name__ == "__main__":
    main()
