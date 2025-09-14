import os
import sys
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Ensure project root is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# Import your preprocessing pipeline and evaluation utility
from src.preprocess import create_preprocessor_pipeline
from src.utils import evaluate_and_save  # we will modify this call to support unsupervised case or write a separate function

# --- Paths ---
data_dir = os.path.join(base_dir, "data", "processed")
train_path = os.path.join(data_dir, "UNSW_NB15_training-set.parquet")
test_path = os.path.join(data_dir, "UNSW_NB15_testing-set.parquet")

results_dir = os.path.join(base_dir, "results", "models")
models_dir = os.path.join(base_dir, "models", "saved_models")

def main():
    print("📦 Loading datasets...")
    train_df = pd.read_parquet(train_path, engine="fastparquet")
    test_df = pd.read_parquet(test_path, engine="fastparquet")

    # Drop label and attack_cat columns for features
    leakage_cols = ["label", "attack_cat"]
    X_train = train_df.drop(columns=leakage_cols)
    y_train = train_df["label"]

    X_test = test_df.drop(columns=leakage_cols)
    y_test = test_df["label"]

    print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Preprocessing
    preprocessor = create_preprocessor_pipeline(X_train)

    # Isolation Forest (unsupervised)
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.1,  # You can tune this to expected anomaly fraction
        random_state=42,
        n_jobs=-1
    )

    # Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", iso_forest)
    ])

    print("🧠 Training Isolation Forest...")
    pipeline.fit(X_train)

    # Predict: Isolation Forest predicts -1 for anomalies and 1 for normal
    y_pred = pipeline.predict(X_test)
    # Convert to binary labels: attack=1, normal=0
    y_pred_binary = [0 if x == 1 else 1 for x in y_pred]

    # Evaluate manually
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)

    print("--- Isolation Forest Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Save metrics to file
    metrics_file = os.path.join(results_dir, "isolationforest_results.txt")
    os.makedirs(results_dir, exist_ok=True)
    with open(metrics_file, "w") as f:
        f.write("--- Isolation Forest Evaluation Metrics ---\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title("Isolation Forest Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "isolationforest_confusion_matrix.png"))
    plt.close()

    # Save model
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(models_dir, "isolation_forest.joblib"))

    print("✅ Isolation Forest training complete.")

if __name__ == "__main__":
    main()
