import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ======================================================
# 1️⃣ Create Preprocessor Pipeline
# ======================================================
def create_preprocessor_pipeline(data: pd.DataFrame):
    """
    Create and fit a preprocessing pipeline with scaling + one-hot encoding.
    Used both for model training and for live SSH log data.
    """
    num_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = data.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    # ⚠️ Fit the preprocessor immediately (since we don't have one saved)
    preprocessor.fit(data)
    return preprocessor


