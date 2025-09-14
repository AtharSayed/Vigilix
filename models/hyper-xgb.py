import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Ensure project root is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# Import your preprocessing pipeline and evaluation utility
from src.preprocess import create_preprocessor_pipeline
from src.utils import evaluate_and_save

# Skewed numerical columns identified from EDA
SKEWED_FEATURES = [
    "dur", "sbytes", "dbytes", "rate", "sload", "dload",
    "sinpkt", "dinpkt", "sjit", "djit"
]

def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p transform to skewed features for better model accuracy."""
    df = df.copy()
    for col in SKEWED_FEATURES:
        if col in df.columns:
            df[col] = np.log1p(df[col])  # log(1+x) to handle zeros safely
    return df

def load_data():
    """Load processed train and test data."""
    train_path = os.path.join(base_dir, "data", "processed", "UNSW_NB15_training-set.parquet")
    test_path = os.path.join(base_dir, "data", "processed", "UNSW_NB15_testing-set.parquet")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df

def main():
    print("📦 Loading data...")
    train_df, test_df = load_data()

    # Drop label and attack_cat columns to avoid leakage
    leakage_cols = ["label", "attack_cat"]
    X_train = train_df.drop(columns=leakage_cols)
    y_train = train_df["label"]
    X_test = test_df.drop(columns=leakage_cols)
    y_test = test_df["label"]

    print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Log-transform skewed features
    print("🔄 Applying log-transform to skewed features...")
    X_train = log_transform(X_train)
    X_test = log_transform(X_test)

    # Create preprocessing pipeline
    print("🔄 Creating preprocessing pipeline...")
    preprocessor = create_preprocessor_pipeline(X_train)

    # Compute scale_pos_weight to handle imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"⚖️ scale_pos_weight set to: {scale_pos_weight:.2f}")

    # Setup XGBoost classifier
    print("⚙️ Setting up XGBoost classifier...")
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb)
    ])

    # Hyperparameter search space
    param_dist = {
        'classifier__n_estimators': [100, 200, 300, 400],
        'classifier__max_depth': [3, 5, 7, 10],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__gamma': [0, 1, 5],
        'classifier__reg_alpha': [0, 0.1, 1],
        'classifier__reg_lambda': [1, 1.5, 2]
    }

    # Multi-metric scoring
    scoring = {"f1": "f1", "roc_auc": "roc_auc"}

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=30,
        scoring=scoring,
        refit="f1",  # optimize for F1
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    print("🚀 Starting hyperparameter tuning...")
    random_search.fit(X_train, y_train)

    print("🎯 Best parameters found:")
    print(random_search.best_params_)

    best_model = random_search.best_estimator_

    print("📊 Evaluating best model on test set...")
    evaluate_and_save(
        model_name="XGBoost_Tuned",
        pipeline=best_model,
        X_test=X_test,
        y_test=y_test,
        results_dir=os.path.join(base_dir, "results", "models"),
        models_dir=os.path.join(base_dir, "models", "saved_models")
    )

    # Save feature importances for interpretability
    print("📈 Saving feature importances plot...")
    classifier = best_model.named_steps["classifier"]
    importances = classifier.feature_importances_

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances)
    plt.title("XGBoost Feature Importances (Top 30)")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "results", "models", "XGBoost_feature_importances.png"))
    plt.close()

    print("✅ Hyperparameter tuning and evaluation complete.")

if __name__ == "__main__":
    main()
