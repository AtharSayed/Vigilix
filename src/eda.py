# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Configuration ---
# Define the dataset directory and file names
# Note: The EDA is performed on the training set, as is standard practice.
data_dir = r'E:\Network_Intrusion_Detection\data'
train_path = os.path.join(data_dir, 'UNSW_NB15_training-set.parquet')

# Define the directory to save EDA plots and the summary file
results_dir = 'results'
plots_dir = os.path.join(results_dir, 'eda_plots')
os.makedirs(plots_dir, exist_ok=True)

# Define the path for the EDA summary text file
summary_file_path = os.path.join(results_dir, 'eda_summary.txt')

# --- Data Loading ---
try:
    print("Loading datasets...")
    train_df = pd.read_parquet(train_path, engine='fastparquet')
    print("Datasets loaded successfully.")
    print("-" * 50)
except FileNotFoundError:
    print(f"Error: Parquet file not found at {train_path}. Please check the path and file name.")
    sys.exit()

# --- EDA Text Summary (Redirected to a file) ---
original_stdout = sys.stdout
with open(summary_file_path, 'w') as f:
    sys.stdout = f

    print("--- Starting Exploratory Data Analysis (EDA) ---")

    # Display basic info for the training data
    print("\nTraining set info:")
    train_df.info()

    # Check for duplicate rows
    print(f"\nNumber of duplicate rows: {train_df.duplicated().sum()}")
    train_df.drop_duplicates(inplace=True)
    print("Duplicate rows removed.")

    # Check for missing values
    print("\nMissing values per column:")
    print(train_df.isnull().sum())

    # Check the distribution of the target variable 'label'
    print("\nDistribution of 'label' in the training set:")
    print(train_df['label'].value_counts())
    print("\nPercentage distribution:")
    print(train_df['label'].value_counts(normalize=True) * 100)

    # Analyze categorical features
    print("\n--- Categorical Feature Analysis ---")
    categorical_features = ['proto', 'service', 'state']
    for feature in categorical_features:
        if feature in train_df.columns:
            print(f"\nUnique values in '{feature}': {train_df[feature].nunique()}")
            print(f"Top 5 most frequent values in '{feature}':")
            print(train_df[feature].value_counts().head())
        else:
            print(f"\nWarning: Column '{feature}' not found in the dataset.")

    # Analyze numerical features
    print("\n--- Numerical Feature Analysis ---")
    numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
    # Exclude 'label' from numerical features for descriptive stats
    if 'label' in numerical_features:
        numerical_features.remove('label')
    if numerical_features:
        print("\nDescriptive statistics for numerical features:")
        print(train_df[numerical_features].describe().T)
    else:
        print("\nNo numerical features found for analysis.")

    print("--- EDA Text Summary Complete ---")
    print("-" * 50)

# Restore stdout to the console
sys.stdout = original_stdout
print(f"EDA text summary has been saved to: {summary_file_path}")

# --- Visualizations (Saved to a folder) ---
print("Generating and saving visualizations...")

# 1. Distribution of the target variable 'label'
if 'label' in train_df.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=train_df)
    plt.title('Distribution of Normal (0) vs. Attack (1) Traffic')
    plt.xlabel('Traffic Type')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Normal', 'Attack'])
    plt.savefig(os.path.join(plots_dir, 'label_distribution.png'))
    plt.show()

# 2. Distribution of attacks by category
if 'attack_cat' in train_df.columns:
    plt.figure(figsize=(12, 8))
    sns.countplot(y='attack_cat', data=train_df, order=train_df['attack_cat'].value_counts().index)
    plt.title('Distribution of Attack Categories')
    plt.xlabel('Count')
    plt.ylabel('Attack Category')
    plt.savefig(os.path.join(plots_dir, 'attack_category_distribution.png'))
    plt.show()

# 3. Correlation heatmap of numerical features
if numerical_features:
    plt.figure(figsize=(15, 12))
    correlation_matrix = train_df[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
    plt.show()

# 4. Box plots for a few key numerical features to check for outliers
key_numerical_features = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl']
plt.figure(figsize=(18, 10))
plot_count = 0
for i, feature in enumerate(key_numerical_features, 1):
    if feature in train_df.columns:
        plt.subplot(2, 3, i)
        sns.boxplot(x='label', y=feature, data=train_df)
        plt.title(f'Box Plot of {feature} by Traffic Type')
        plot_count += 1
    else:
        print(f"Warning: Column '{feature}' not found, skipping boxplot.")

if plot_count > 0:
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'numerical_feature_boxplots.png'))
    plt.show()
else:
    print("No numerical features to plot box plots for.")

print("All visualizations have been generated and saved.")
print("-" * 50)