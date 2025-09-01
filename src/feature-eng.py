# feature-eng.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

# ========================
# 1. Spark Session
# ========================
spark = SparkSession.builder \
    .appName("CICIDS2017-FeatureEngineering") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# ========================
# 2. File Paths
# ========================
BASE_DIR = "C:/Users/sayed/Desktop/L&T-Project/Vigilix/data/processed"
file_path = f"file:///{BASE_DIR}/cicids2017_preprocessed.csv"
output_path = f"file:///{BASE_DIR}/cicids2017_features"

print("🔹 Loading preprocessed dataset...")
df = spark.read.csv(file_path, header=True, inferSchema=True)
print(f"✅ Dataset loaded with {df.count()} rows and {len(df.columns)} columns")

# ========================
# 3. Clean column names
# ========================
df = df.toDF(*[c.strip().replace(" ", "_").replace(".", "_").replace("/", "_") for c in df.columns])
print("✅ Column names cleaned")

# ========================
# 4. Encode Label
# ========================
target_col = "Label"
if target_col not in df.columns:
    raise ValueError(f"❌ Target column '{target_col}' not found in dataset. Available columns: {df.columns}")

indexer = StringIndexer(inputCol=target_col, outputCol="label_index")
df = indexer.fit(df).transform(df)
print("✅ Label column encoded")

# ========================
# 5. Assemble Features
# ========================
feature_cols = [c for c in df.columns if c not in [target_col, "label_index"]]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
df = assembler.transform(df)
print(f"✅ Features assembled: {len(feature_cols)} columns")

# ========================
# 6. Scale Features
# ========================
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
df = scaler.fit(df).transform(df)
print("✅ Features scaled")

# ========================
# 7. Train/Test Split
# ========================
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"✅ Train size: {train_df.count()}, Test size: {test_df.count()}")

# ========================
# 8. Save Processed Data
# ========================
if os.path.exists(output_path.replace("file:///", "")):
    print("⚠️ Output path already exists, overwriting...")

train_df.write.mode("overwrite").parquet(f"{output_path}/train")
test_df.write.mode("overwrite").parquet(f"{output_path}/test")

print(f"🎉 Feature engineering completed! Saved to {output_path}")
