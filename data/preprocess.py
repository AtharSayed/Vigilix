import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = "C:/Users/sayed/Desktop/L&T-Project/Vigilix"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw/")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data/processed/")
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "cicids2017_preprocessed.csv")

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

    # Scale numeric columns
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
    first = True
    scaler, label_encoders = None, {}

    for file in os.listdir(RAW_DATA_DIR):
        if file.endswith(".csv"):
            path = os.path.join(RAW_DATA_DIR, file)
            print(f"Processing {path} ...")
            
            # Load one file at a time
            df = pd.read_csv(path, low_memory=False)

            df, scaler, label_encoders = preprocess_dataframe(df, scaler, label_encoders)

            # Append to final file
            mode = "w" if first else "a"
            header = first
            df.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)
            first = False

    print(f"✅ Preprocessed dataset saved at {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
