import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix
)
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================
# 🔹 CONFIG
# ==============================
DATA_PATH = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\data\raw\cic-collection.parquet\cic-collection.parquet"
CLASSIFICATION_MODE = "binary"   # Isolation Forest is primarily for unsupervised anomaly detection
TEST_SIZE = 0.2
RANDOM_STATE = 42
# Contamination is the proportion of outliers in the dataset.
# This is a critical hyperparameter for Isolation Forest.
CONTAMINATION = 0.01 

MODEL_DIR = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\model_output"
RESULTS_DIR = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\results"
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest_model.joblib")
RESULTS_FILE = os.path.join(RESULTS_DIR, "IsolationForest_evals.txt")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# 🔹 LOAD DATA
# ==============================
print("🔹 Loading dataset...")
df = pd.read_parquet(DATA_PATH, engine="pyarrow")
print(f"✅ Dataset loaded: {df.shape}")
print(f"✅ Columns: {list(df.columns)[:10]} ...")

# ==============================
# 🔹 PREPROCESS LABELS
# ==============================
print("🔹 Checking label distribution...")
print(df["Label"].value_counts().head(20))

if CLASSIFICATION_MODE == "binary":
    print("🔹 Converting labels to binary (Benign=0, Attack=1)...")
    df["Label"] = df["Label"].apply(lambda x: 0 if str(x).lower().strip() == "benign" else 1)
    num_classes = 2
else:
    raise ValueError("Isolation Forest is best suited for binary (unsupervised) anomaly detection. Set CLASSIFICATION_MODE to 'binary'.")

# ==============================
# 🔹 SPLIT DATA
# ==============================
non_numeric_cols = df.select_dtypes(include=["object"]).columns.tolist()
if "Label" in non_numeric_cols:
    non_numeric_cols.remove("Label")

X = df.drop(columns=["Label"] + non_numeric_cols)
y = df["Label"]

assert all(dtype.kind in "ifb" for dtype in X.dtypes), "❌ Non-numeric columns still present in features!"

print("🔹 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"✅ Train size: {X_train.shape}, Test size: {X_test.shape}")

# ==============================
# 🔹 TRAIN ISOLATION FOREST
# ==============================
print("🔹 Training Isolation Forest Classifier...")
# The Isolation Forest model is an unsupervised algorithm. It does not require 'y_train' during training.
# It learns to identify anomalies from the structure of the data itself.
clf = IsolationForest(
    n_estimators=100, 
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    max_features=1.0,
    n_jobs=-1 # Use all available cores
)

# Fit the model on the training data. Note: y_train is not used for unsupervised training.
clf.fit(X_train)

# ==============================
# 🔹 EVALUATE MODEL
# ==============================
print("\n🔹 Evaluating model...")

# predict returns -1 for anomalies and 1 for normal instances.
# We convert them to 1 and 0 to match our label scheme (1=Attack, 0=Benign).
y_pred = clf.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)

# Metrics
print("\n✅ Classification Report:")
report = classification_report(y_test, y_pred)
print(report)

print("\n✅ Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# ==============================
# 🔹 SAVE MODEL & RESULTS
# ==============================
import joblib # joblib is preferred for sklearn models

joblib.dump(clf, MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(RESULTS_FILE, "a", encoding='utf-8') as f:
    f.write(f"\n\n==== Training Results [{timestamp}] ====\n")
    f.write(f"Mode: {CLASSIFICATION_MODE}\n")
    f.write(f"Contamination: {CONTAMINATION}\n")
    f.write("🔹 Classification Report:\n")
    f.write(report)
    f.write("\n🔹 Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix, separator=", "))
    f.write("\n" + "=" * 50 + "\n")

print(f"✅ Results appended to {RESULTS_FILE}")