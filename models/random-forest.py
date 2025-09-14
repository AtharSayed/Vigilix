# random_forest.py

import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# Import custom modules
from src.preprocess import create_preprocessor_pipeline
from src.utils import evaluate_and_save

# --- Paths ---
data_dir = os.path.join(base_dir, "data", "processed")
train_path = os.path.join(data_dir, "UNSW_NB15_training-set.parquet")
test_path = os.path.join(data_dir, "UNSW_NB15_testing-set.parquet")

results_dir = os.path.join(base_dir, "results", "models")
models_dir = os.path.join(base_dir, "models", "saved_models")

# --- Load Data ---
print("📦 Loading datasets...")
train_df = pd.read_parquet(train_path, engine="fastparquet")
test_df = pd.read_parquet(test_path, engine="fastparquet")

# --- Drop label leakage columns ---
leakage_cols = ["label", "attack_cat"]   # To be used for multi - class classification to categorize the attacks 
X_train = train_df.drop(columns=leakage_cols)
y_train = train_df["label"]

X_test = test_df.drop(columns=leakage_cols)
y_test = test_df["label"]

print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- Preprocessing ---
preprocessor = create_preprocessor_pipeline(X_train)

# --- Model ---
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# --- Pipeline ---
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", rf_model)
])

# --- Train ---
print("🧠 Training Random Forest...")
pipeline.fit(X_train, y_train)

# --- Evaluate & Save ---
evaluate_and_save(
    model_name="RandomForest",
    pipeline=pipeline,
    X_test=X_test,
    y_test=y_test,
    results_dir=results_dir,
    models_dir=models_dir
)

print("✅ Random Forest training complete.")
