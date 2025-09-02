import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = "C:/Users/sayed/Desktop/L&T-Project/Vigilix"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw/")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data/processed/")
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "cicids2017_preprocessed.parquet")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def preprocess_dataframe(df, scaler=None, label_encoders=None):
    """Clean, encode, scale a single DataFrame."""
    # Handle inf/nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Encode categorical features
    if label_encoders is None:
        label_encoders = {}

    for col in ["Protocol", "Label"]:
        if col in df.columns:
            if col not in label_encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
            else:
                df[col] = label_encoders[col].transform(df[col].astype(str))

    # Scale numeric columns (except Label)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Label" in numeric_cols:
        numeric_cols.remove("Label")

    if scaler is None:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, scaler, label_encoders


def main():
    scaler, label_encoders = None, {}
    all_dfs = []

    for file in os.listdir(RAW_DATA_DIR):
        if file.endswith(".csv"):
            path = os.path.join(RAW_DATA_DIR, file)
            print(f"Processing {path} ...")
            
            # Load one file at a time
            df = pd.read_csv(path, low_memory=False)

            df, scaler, label_encoders = preprocess_dataframe(df, scaler, label_encoders)

            all_dfs.append(df)

    # Concatenate all processed chunks
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"✅ Preprocessed dataset saved at {OUTPUT_FILE}")
    else:
        print("⚠️ No CSV files found in raw data directory.")


if __name__ == "__main__":
    main()
