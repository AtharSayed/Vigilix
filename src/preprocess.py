import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_preprocessor_pipeline(data: pd.DataFrame):
    """Create preprocessing pipeline with scaling + one-hot encoding."""
    num_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = data.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])
    return preprocessor
