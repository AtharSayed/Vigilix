import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when, mean, stddev, skewness
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\data\raw\cic-collection.parquet\cic-collection.parquet"
RESULTS_PATH = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\results\eda_results.txt"
SAMPLE_SIZE = 100_000

# -------------------------------
# INIT SPARK
# -------------------------------
spark = SparkSession.builder \
    .appName("NetworkEDA") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# -------------------------------
# LOAD DATA
# -------------------------------
print("📦 Loading data with Spark...")
df = spark.read.parquet(DATA_PATH)
num_rows = df.count()
num_cols = len(df.columns)
print(f"✅ Loaded {num_rows:,} rows and {num_cols} columns")

# -------------------------------
# Start Writing to File
# -------------------------------
with open(RESULTS_PATH, "a", encoding="utf-8") as f:
    f.write("\n" + "="*80 + "\n")
    f.write("🔍 SPARK EDA REPORT\n")
    f.write("="*80 + "\n")
    f.write(f"✅ Data shape: {num_rows:,} rows × {num_cols} columns\n")

# -------------------------------
# LABEL DISTRIBUTION
# -------------------------------
label_counts = df.groupBy("Label").count().orderBy("count", ascending=False)
label_str = label_counts.toPandas().to_string(index=False)

with open(RESULTS_PATH, "a", encoding="utf-8") as f:
    f.write("\n📊 Label Distribution:\n")
    f.write(label_str + "\n")

# -------------------------------
# MISSING VALUES
# -------------------------------
missing = df.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in df.columns
])
missing_str = missing.toPandas().T
missing_str.columns = ["MissingCount"]
missing_str = missing_str[missing_str["MissingCount"] > 0]

with open(RESULTS_PATH, "a", encoding="utf-8") as f:
    f.write("\n❓ Missing Values (non-zero only):\n")
    if not missing_str.empty:
        f.write(missing_str.to_string() + "\n")
    else:
        f.write("✅ No missing values found.\n")

# -------------------------------
# BASIC STATS
# -------------------------------
stats = df.describe().toPandas().set_index("summary").T
with open(RESULTS_PATH, "a", encoding="utf-8") as f:
    f.write("\n📈 Basic Statistics:\n")
    f.write(stats.head(10).to_string() + "\n")  # only first 10 features for brevity

# -------------------------------
# SKEWNESS
# -------------------------------
numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ['double', 'float', 'int', 'bigint']]
skew_vals = df.select([skewness(c).alias(c) for c in numeric_cols[:10]])
skew_pd = skew_vals.toPandas().T
skew_pd.columns = ["Skewness"]

with open(RESULTS_PATH, "a", encoding="utf-8") as f:
    f.write("\n📉 Skewness (Top 10 numerical columns):\n")
    f.write(skew_pd.to_string() + "\n")

# -------------------------------
# SAMPLE FOR PLOTTING
# -------------------------------
print("\n📌 Sampling data for plotting...")
sample_df = df.sample(fraction=0.01, seed=42).limit(SAMPLE_SIZE).toPandas()

# -------------------------------
# PLOTS
# -------------------------------
# Label distribution plot
sns.countplot(data=sample_df, x="Label", order=sample_df["Label"].value_counts().index)
plt.title("Label Distribution (Sampled)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("spark_label_distribution.png")
plt.close()

# Correlation heatmap (top features)
top_features = sample_df.select_dtypes(include=["float64", "int64"]).corr().abs().sum().sort_values(ascending=False).head(10).index
sns.heatmap(sample_df[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Top Feature Correlations (Sampled)")
plt.tight_layout()
plt.savefig("spark_correlation_heatmap.png")
plt.close()

# -------------------------------
# DONE
# -------------------------------
print("✅ EDA completed. Results appended to:")
print(RESULTS_PATH)
