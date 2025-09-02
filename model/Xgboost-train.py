import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

BASE_DIR = "C:/Users/sayed/Desktop/L&T-Project/Vigilix"
FEATURE_PATH = os.path.join(BASE_DIR, "data/processed/cicids2017_features")

print("🔹 Loading feature-engineered dataset...")
df = pd.read_parquet(FEATURE_PATH)
print(f"✅ Dataset loaded: {df.shape}")
print(f"✅ Columns: {df.columns.tolist()}")

# --- FIXED FEATURE EXPANSION ---
print("🔹 Expanding Spark 'features' column into numeric array...")

# If features are dicts, unpack their 'values'
def extract_vector(x):
    if isinstance(x, dict):
        return x.get("values", [])
    elif isinstance(x, (list, np.ndarray)):
        return x
    else:
        return []

df_features = pd.DataFrame(df["features"].apply(extract_vector).tolist())

# Assign proper column names
df_features.columns = [f"f{i}" for i in range(df_features.shape[1])]

# Force all numeric
df_features = df_features.apply(pd.to_numeric, errors="coerce")
df_features.dropna(inplace=True)

# Merge back label
df = pd.concat([df_features, df["label_index"].reset_index(drop=True)], axis=1)

X = df.drop(columns=["label_index"])
y = (df["label_index"] != 0).astype(int)

print(f"✅ Features shape: {X.shape}, Labels shape: {y.shape}")
print(f"✅ Feature dtypes:\n{X.dtypes.value_counts()}")

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"✅ Train size: {X_train.shape}, Test size: {X_test.shape}")

# --- XGBoost Training ---
print("🔹 Training XGBoost Binary Classifier...")
clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist"
)
clf.fit(X_train, y_train)

# --- Evaluation ---
print("\n🔹 Evaluating model...\n")
y_pred = clf.predict(X_test)
print("✅ Classification Report:\n", classification_report(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Feature Importance ---
print("\n🔍 Top 15 Important Features:")
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1][:15]
for i in indices:
    print(f"{X.columns[i]}: {importances[i]:.4f}")

plt.figure(figsize=(10,6))
xgb.plot_importance(clf, max_num_features=15, importance_type="gain")
plt.title("Top 15 Feature Importances")
plt.show()
