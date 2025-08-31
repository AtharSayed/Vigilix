import sys
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, when

def main():
    # 'spark' is created here and is accessible within this function
    try:
        spark = SparkSession.builder \
            .appName("NetworkIntrusionEDA") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
    except pyspark.sql.utils.SparkCoreMemoryException:
        print("Failed to get or create Spark Session due to memory constraints. Trying again with more memory.")
        spark.stop()
        spark = SparkSession.builder \
            .appName("NetworkIntrusionEDA") \
            .config("spark.driver.memory", "8g") \
            .getOrCreate()

    # Define the dataset path. Replace with your actual path if different.
    dataset_path = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\data\processed\cicids2017_preprocessed.csv"

    print("📂 Loading dataset...")
    df = spark.read.csv(dataset_path, header=True, inferSchema=True)

    print("🧹 Cleaning column names...")
    cleaned_columns = [c.strip().replace(" ", "_").replace("/", "_").replace(".", "_") for c in df.columns]
    df_cleaned = df.toDF(*cleaned_columns)

    print("📑 Schema (with cleaned column names):")
    df_cleaned.printSchema()

    print(f"\n✅ Data loaded with {df_cleaned.count()} rows and {len(df_cleaned.columns)} columns\n")
    print("👀 Sample Data:")
    df_cleaned.show(5)

    print("🚫 Null Value Counts (Corrected):")
    null_counts_expr = [spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) for c in df_cleaned.columns]
    null_counts_row = df_cleaned.agg(*null_counts_expr).collect()[0]
    
    null_counts = null_counts_row.asDict()

    for column, count in null_counts.items():
        if count > 0:
            print(f"  - {column}: {count} null values")
    
    if all(count == 0 for count in null_counts.values()):
        print("  - No null values found in any column.")

    spark.stop()
    print("\n✅ Spark session stopped.")

if __name__ == "__main__":
    main()