import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================
# 🔹 CONFIG
# ==============================
DATA_PATH = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\data\raw\cic-collection.parquet\cic-collection.parquet"
CLASSIFICATION_MODE = "binary"   # "binary" or "multi"
TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.5   # Default decision threshold (can tune later)

MODEL_DIR = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\model_output"
RESULTS_DIR = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\results"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
RESULTS_FILE = os.path.join(RESULTS_DIR, "Xgboost_evals.txt")

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
    objective = "binary:logistic"

    # Compute scale_pos_weight for imbalance
    neg, pos = np.bincount(df["Label"])
    scale_pos_weight = neg / pos
    print(f"⚖️ Auto class weight (scale_pos_weight): {scale_pos_weight:.2f}")

else:
    print("🔹 Using multi-class labels (ClassLabel column)...")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["ClassLabel"].astype(str))
    num_classes = df["Label"].nunique()
    objective = "multi:softmax"
    scale_pos_weight = None
    print(f"✅ Number of classes: {num_classes}")

# ==============================
# 🔹 SPLIT DATA
# ==============================
non_numeric_cols = df.select_dtypes(include=["object"]).columns.tolist()
if "Label" in non_numeric_cols:
    non_numeric_cols.remove("Label")

X = df.drop(columns=["Label"] + non_numeric_cols)
y = df["Label"]

assert all(dtype.kind in "ifb" for dtype in X.dtypes), "❌ Non-numeric columns still present in features!"

print("🔹 Splitting dataset (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"✅ Train size: {X_train.shape}, Test size: {X_test.shape}")

# ==============================
# 🔹 TRAIN XGBOOST
# ==============================
print("🔹 Training XGBoost Classifier...")

params = dict(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    objective=objective,
    eval_metric="logloss",
    use_label_encoder=False
)
if CLASSIFICATION_MODE == "binary":
    params["scale_pos_weight"] = scale_pos_weight
if CLASSIFICATION_MODE == "multi":
    params["num_class"] = num_classes

clf = xgb.XGBClassifier(**params)
clf.fit(X_train, y_train)

# ==============================
# 🔹 EVALUATE MODEL
# ==============================
print("\n🔹 Evaluating model...")

if CLASSIFICATION_MODE == "binary":
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    # ROC-AUC & PR-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"✅ ROC-AUC: {roc_auc:.4f}")
    print(f"✅ PR-AUC: {pr_auc:.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(rec, prec, label=f"PR curve (AUC = {pr_auc:.4f})")
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
xgb.plot_importance(clf, max_num_features=15, importance_type="gain")
plt.show()

# ==============================
# 🔹 SAVE MODEL & RESULTS
# ==============================
clf.save_model(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(RESULTS_FILE, "a") as f:
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
