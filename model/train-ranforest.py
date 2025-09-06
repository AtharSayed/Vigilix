import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

# ==============================
# 🔹 CONFIG
# ==============================
DATA_PATH = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\data\raw\cic-collection.parquet\cic-collection.parquet"
CLASSIFICATION_MODE = "binary"  
TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.5   # Default decision threshold (can tune later)

MODEL_DIR = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\model_output"
RESULTS_DIR = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\results"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.joblib")
RESULTS_FILE = os.path.join(RESULTS_DIR, "RandomForest_evals.txt")

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
    print("🔹 Using original labels for multi-class classification...")
    df["Label"] = df["Label"].astype('category').cat.codes
    num_classes = len(df["Label"].unique())

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
# 🔹 TRAIN RANDOM FOREST
# ==============================
print("🔹 Training Random Forest Classifier...")
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,  # Use all available cores
    class_weight='balanced',  # Addresses class imbalance
    max_samples=100000  # Reduces training time by using a subset of the data for each tree
)

clf.fit(X_train, y_train)
print("✅ Training complete.")

# ==============================
# 🔹 EVALUATE MODEL
# ==============================
print("\n🔹 Evaluating model...")

if CLASSIFICATION_MODE == "binary":
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"✅ ROC-AUC: {roc_auc:.4f}")

    # PR-AUC
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"✅ PR-AUC: {pr_auc:.4f}")

    # Plot ROC and PR curves
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.show()

else:
    y_pred = clf.predict(X_test)
    y_proba = None  # multi-class handled differently

# Metrics
print("\n✅ Classification Report:")
report = classification_report(y_test, y_pred)
print(report)

print("\n✅ Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# ==============================
# 🔹 FEATURE IMPORTANCE
# ==============================
print("\n🔍 Plotting Top 15 Important Features...")
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
top_15_features = feature_importances.nlargest(15)

plt.figure(figsize=(10, 8))
top_15_features.sort_values().plot(kind='barh')
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ==============================
# 🔹 SAVE MODEL & RESULTS
# ==============================
joblib.dump(clf, MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(RESULTS_FILE, "a", encoding='utf-8') as f:
    f.write(f"\n\n==== Training Results [{timestamp}] ====\n")
    f.write(f"Mode: {CLASSIFICATION_MODE}\n")
    if CLASSIFICATION_MODE == "binary":
        f.write(f"Threshold: {THRESHOLD}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}\n")
    f.write("🔹 Classification Report:\n")
    f.write(report)
    f.write("\n🔹 Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix, separator=", "))
    f.write("\n" + "=" * 50 + "\n")

print(f"✅ Results appended to {RESULTS_FILE}")