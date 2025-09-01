# feature-eng.py (fixed)

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation
import os

# Initialize Spark
spark = SparkSession.builder \
    .appName("FeatureEngineering_CICIDS2017") \
    .getOrCreate()

# File paths
BASE_PATH = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\data"
INPUT_FILE = os.path.join(BASE_PATH, "processed", "cicids2017_preprocessed.csv")
OUTPUT_FILE = os.path.join(BASE_PATH, "processed", "cicids2017_features.csv")

print("🔹 Loading preprocessed dataset...")
df = spark.read.csv(INPUT_FILE, header=True, inferSchema=True)

print(f"Dataset loaded with {df.count()} rows and {len(df.columns)} columns")
print("🔹 Columns in dataset:")
print(df.columns[:20])  # print first 20 columns to inspect

# -------------------------------
# 1. Fix Label Column Name
# -------------------------------
# Standardize column names (strip spaces, lowercase)
df = df.toDF(*[c.strip().replace(" ", "_") for c in df.columns])

# Find label column
label_col_candidates = [c for c in df.columns if c.lower() in ["label", "attack", "class", "target"]]
if not label_col_candidates:
    raise ValueError("❌ No Label column found. Please check your dataset!")
else:
    LABEL_COL = label_col_candidates[0]
    print(f"✅ Using '{LABEL_COL}' as target column")

# -------------------------------
# 2. Encode Label
# -------------------------------
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol=LABEL_COL, outputCol="Label_Indexed")
df = indexer.fit(df).transform(df)

# -------------------------------
# 3. Assemble Features
# -------------------------------
from pyspark.sql.types import NumericType

# Keep only numeric columns (float, int, double)
numeric_cols = [c for (c, dtype) in df.dtypes if isinstance(df.schema[c].dataType, NumericType)]

# Drop target column from features
feature_cols = [c for c in numeric_cols if c not in [LABEL_COL, "Label_Indexed"]]

print(f"✅ Using {len(feature_cols)} numeric feature columns")

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
df = assembler.transform(df)

# -------------------------------
# 4. Scale Features
# -------------------------------
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
df = scaler.fit(df).transform(df)

# -------------------------------
# 5. Save Feature Engineered Dataset
# -------------------------------
print("🔹 Saving feature-engineered dataset...")
df.select(feature_cols + ["Label_Indexed"]).write.csv(OUTPUT_FILE, header=True, mode="overwrite")

print(f"✅ Feature engineering completed! Final dataset saved at: {OUTPUT_FILE}")

spark.stop()
