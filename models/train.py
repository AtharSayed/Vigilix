import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# --- Fix for import error: Add the 'src' directory to the system path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(src_dir)

# Import the preprocessor pipeline from preprocess.py
try:
    from preprocess import create_preprocessor_pipeline
except ImportError as e:
    print("Error: preprocess.py not found or failed to import. Please ensure it is located in the 'src' directory.")
    print("Detailed error:", e)
    sys.exit()

def save_evaluation_results(model_name, metrics, output_dir='results'):
    """
    Saves the model's evaluation metrics to a text file.

    Args:
        model_name (str): The name of the model.
        metrics (dict): A dictionary containing the evaluation metrics.
        output_dir (str): The directory to save the output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{model_name}_evaluations.txt')

    with open(file_path, 'w') as f:
        f.write(f"--- Model: {model_name} ---\n\n")
        f.write("--- Evaluation Metrics ---\n")
        for metric, value in metrics.items():
            f.write(f"{metric.ljust(10)}: {value:.4f}\n")

    print(f"Evaluation results saved to: {file_path}")

# --- Configuration ---
data_dir = r'E:\Network_Intrusion_Detection\data'
train_path = os.path.join(data_dir, 'UNSW_NB15_training-set.parquet')
test_path = os.path.join(data_dir, 'UNSW_NB15_testing-set.parquet')

# --- Data Loading ---
try:
    print("Loading datasets...")
    train_df = pd.read_parquet(train_path, engine='fastparquet')
    test_df = pd.read_parquet(test_path, engine='fastparquet')
    print("Datasets loaded successfully.")
    print("-" * 50)
except FileNotFoundError:
    print(f"Error: One or both parquet files not found in {data_dir}. Please check the path and file names.")
    sys.exit()
except ImportError:
    print("Error: Required 'fastparquet' library not found. Please install it: pip install fastparquet")
    sys.exit()

# --- Split Features and Target ---
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)

X_train = train_df.drop(columns=['label', 'attack_cat'], errors='ignore')
y_train = train_df['label']

X_test = test_df.drop(columns=['label', 'attack_cat'], errors='ignore')
y_test = test_df['label']

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print("-" * 50)

# --- Preprocessing and Model Pipeline ---
preprocessor = create_preprocessor_pipeline(X_train)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# --- Training ---
print("Starting model training...")
full_pipeline.fit(X_train, y_train)
print("Model training complete.")
print("-" * 50)

# --- Evaluation ---
print("Making predictions on the test set...")
y_pred = full_pipeline.predict(X_test)
print("Predictions complete.")
print("-" * 50)

print("--- Evaluation Metrics ---")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

metrics_dict = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
}

save_evaluation_results(model.__class__.__name__, metrics_dict)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print("-" * 50)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])

print("Confusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

print("Script finished.")
