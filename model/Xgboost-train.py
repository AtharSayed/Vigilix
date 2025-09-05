import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt

# ==============================
# 🔹 CONFIG
# ==============================
DATA_PATH = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\data\raw\cic-collection.parquet\cic-collection.parquet"
CLASSIFICATION_MODE = "binary"   
TEST_SIZE = 0.2
RANDOM_STATE = 42

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
    print("🔹 Using multi-class labels (ClassLabel column)...")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["ClassLabel"].astype(str))
    num_classes = df["Label"].nunique()
    print(f"✅ Number of classes: {num_classes}")

# ==============================
# 🔹 SPLIT DATA
# ==============================
# Drop non-numeric (object) columns except 'Label'
non_numeric_cols = df.select_dtypes(include=["object"]).columns.tolist()
if "Label" in non_numeric_cols:
    non_numeric_cols.remove("Label")  # keep the label column

X = df.drop(columns=["Label"] + non_numeric_cols)
y = df["Label"]

# Optional: Ensure no non-numeric data remains
assert all(dtype.kind in 'ifb' for dtype in X.dtypes), "❌ Non-numeric columns still present in features!"

print("🔹 Splitting dataset (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"✅ Train size: {X_train.shape}, Test size: {X_test.shape}")

# ==============================
# 🔹 TRAIN XGBOOST
# ==============================
print("🔹 Training XGBoost Classifier...")

clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",  # faster on large data
    objective="binary:logistic" if CLASSIFICATION_MODE == "binary" else "multi:softmax",
    num_class=num_classes if CLASSIFICATION_MODE == "multi" else None,
    eval_metric="logloss",
    use_label_encoder=False  # for compatibility
)

clf.fit(X_train, y_train)

# ==============================
# 🔹 EVALUATE MODEL
# ==============================
print("\n🔹 Evaluating model...")

y_pred = clf.predict(X_test)

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 🔹 FEATURE IMPORTANCE
# ==============================
print("\n🔍 Plotting Top 15 Important Features...")
xgb.plot_importance(clf, max_num_features=15, importance_type="gain")
plt.show()

# ==============================
# 🔹 SAVE MODEL
# ==============================
MODEL_PATH = r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\model_output\xgboost_model.json"
clf.save_model(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
