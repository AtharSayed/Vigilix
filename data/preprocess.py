import pandas as pd
import numpy as np
import os
import sys
from contextlib import redirect_stdout

# === File Paths ===
raw_data = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\data\raw\cic-collection.parquet\cic-collection.parquet"
processed_data = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\data\processed\cic-collection-cleaned.parquet"
log_file_path = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\results\preprocess.txt"

# === Features to Drop ===

# Contaminated (leaky) features
contaminated_features = [
    'PSH Flag Count', 'ECE Flag Count', 'RST Flag Count', 'ACK Flag Count',
    'Fwd Packet Length Min', 'Bwd Packet Length Min', 'Packet Length Min',
    'Protocol', 'Down/Up Ratio'
]

# Zero-predictive (useless) features
zero_predictive_features = [
    'Bwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
    'Bwd PSH Flags', 'Bwd URG Flags', 'CWE Flag Count', 'FIN Flag Count',
    'Fwd Avg Bulk Rate', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
    'Fwd URG Flags'
]

# === Create output directories if needed ===
os.makedirs(os.path.dirname(processed_data), exist_ok=True)
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# === Redirect all print output to log file ===
with open(log_file_path, "a") as log_file:
    with redirect_stdout(log_file):

        print("\n========== 🧹 Data Preprocessing Started ==========\n")

        # Step 1: Load dataset
        print("📥 Loading dataset...")
        df = pd.read_parquet(raw_data)
        print(f"✅ Initial shape: {df.shape}\n")

        # Step 2: Drop contaminated features
        contaminated_to_drop = [col for col in contaminated_features if col in df.columns]
        df.drop(columns=contaminated_to_drop, inplace=True)
        print(f"🧹 Dropped contaminated features: {contaminated_to_drop}")
        print(f"📏 Shape after contaminated drop: {df.shape}\n")

        # Step 3: Drop zero-predictive features
        zero_to_drop = [col for col in zero_predictive_features if col in df.columns]
        df.drop(columns=zero_to_drop, inplace=True)
        print(f"🧽 Dropped zero-predictive features: {zero_to_drop}")
        print(f"📏 Shape after zero-predictive drop: {df.shape}\n")

        # Step 4: Save cleaned file
        df.to_parquet(processed_data, index=False)
        print(f"📤 Cleaned dataset saved to: {processed_data}\n")

        # Step 5: Summary of remaining data
        print("📌 Remaining Features:")
        print(df.columns.tolist())
        print(f"\n📏 Final shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Optional: Data type info
        print("\n🧪 Column Data Types and Missing Values:")
        df.info()

        print("\n========== ✅ Preprocessing Complete ==========\n")
