# XgboostModel.py

import os
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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

# --- Drop leakage columns ---
leakage_cols = ["label", "attack_cat"] # Removed the multi-class attack feature since we are doing binary classifn
X_train = train_df.drop(columns=leakage_cols)
y_train = train_df["label"]

X_test = test_df.drop(columns=leakage_cols)
y_test = test_df["label"]

print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- Preprocessing ---
preprocessor = create_preprocessor_pipeline(X_train)

# --- Model ---
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1,
    random_state=42
)

# --- Pipeline ---
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

# --- Train ---
print("🧠 Training XGBoost...")
pipeline.fit(X_train, y_train)

# --- Evaluate & Save ---
evaluate_and_save(
    model_name="XGBoost",
    pipeline=pipeline,
    X_test=X_test,
    y_test=y_test,
    results_dir=results_dir,
    models_dir=models_dir
)

print("✅ XGBoost training complete.")
