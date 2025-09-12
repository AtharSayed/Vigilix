import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

def evaluate_and_save(model_name, pipeline, X_test, y_test, results_dir, models_dir):
    """Evaluate model, save metrics, confusion matrix, and trained model."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0)
    }

    # Save metrics
    metrics_file = os.path.join(results_dir, f"{model_name}_results.txt")
    with open(metrics_file, "w") as f:
        f.write(f"--- {model_name} Evaluation Metrics ---\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(results_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # Save trained model
    joblib.dump(pipeline, os.path.join(models_dir, f"{model_name}.joblib"))

    print(f"{model_name}: Results saved to {results_dir}")
