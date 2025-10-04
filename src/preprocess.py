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

def preprocess_ssh_logs(df: pd.DataFrame, preprocessor: ColumnTransformer):
    """
    Preprocess SSH log data to match UNSW-NB15 dataset feature format.
    Transforms the features and returns the processed data ready for prediction.
    """
    # Feature transformation for 'proto' column (protocol mapping)
    df['protocol'] = df['protocol_type'].map({'tcp': 1, 'udp': 2, 'icmp': 3}).fillna(0).astype(int)  # Adjust as per protocol types

    # Handle other features (assuming these columns are available in your SSH logs)
    df['bytes'] = df['bytes_sent'] + df['bytes_received']  # Aggregate bytes
    df['packets'] = df['packets_sent'] + df['packets_received']  # Aggregate packets
    
    # Handle missing values (example)
    df.fillna(0, inplace=True)
    
    # Ensure categorical columns are encoded correctly
    # (assuming you want to use OneHotEncoding for 'protocol', 'state', etc.)
    
    # Run preprocessor if available (for scaling or encoding)
    df_processed = preprocessor.fit_transform(df)

    return df_processed
    
