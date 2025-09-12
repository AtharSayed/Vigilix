import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "processed")   # <-- FIXED
train_path = os.path.join(data_dir, "UNSW_NB15_training-set.parquet")
test_path = os.path.join(data_dir, "UNSW_NB15_testing-set.parquet")


results_dir = os.path.join(base_dir, "results", "eda")
plots_dir = os.path.join(results_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

summary_file_path = os.path.join(results_dir, "eda_summary.txt")

# --- Load Data ---
print("Loading training dataset...")
train_df = pd.read_parquet(train_path, engine="fastparquet")
print("Dataset loaded.")

# --- EDA Summary ---
original_stdout = sys.stdout
with open(summary_file_path, "w") as f:
    sys.stdout = f

    print("--- Exploratory Data Analysis (EDA) ---\n")
    print("Training Data Info:\n")
    train_df.info()

    print("\nDuplicate Rows:", train_df.duplicated().sum())
    train_df.drop_duplicates(inplace=True)
    print("Duplicates removed.")

    print("\nMissing Values:\n", train_df.isnull().sum())
    print("\nLabel Distribution:\n", train_df["label"].value_counts())
    print("\nLabel Distribution (%):\n", train_df["label"].value_counts(normalize=True) * 100)

    print("\n--- Categorical Features ---")
    categorical_features = ["proto", "service", "state"]
    for feature in categorical_features:
        if feature in train_df.columns:
            print(f"\n{feature}:")
            print("Unique:", train_df[feature].nunique())
            print("Top 5:\n", train_df[feature].value_counts().head())

    print("\n--- Numerical Features ---")
    num_features = train_df.select_dtypes(include=np.number).columns.tolist()
    if "label" in num_features:
        num_features.remove("label")
    print(train_df[num_features].describe().T)

sys.stdout = original_stdout
print(f"EDA summary saved to {summary_file_path}")

# --- Plots ---
print("Generating plots...")

# Label distribution
plt.figure(figsize=(6,4))
sns.countplot(x="label", data=train_df)
plt.title("Normal (0) vs Attack (1)")
plt.savefig(os.path.join(plots_dir, "label_distribution.png"))
plt.close()

# Attack category distribution
if "attack_cat" in train_df.columns:
    plt.figure(figsize=(8,6))
    sns.countplot(y="attack_cat", data=train_df, order=train_df["attack_cat"].value_counts().index)
    plt.title("Attack Categories")
    plt.savefig(os.path.join(plots_dir, "attack_category_distribution.png"))
    plt.close()

# Correlation heatmap
if num_features:
    plt.figure(figsize=(10,8))
    sns.heatmap(train_df[num_features].corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"))
    plt.close()

print("EDA plots saved.")
