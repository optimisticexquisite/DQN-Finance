import pandas as pd
import torch
from features.bitcoin_features import extract_all_features

def prepare_features(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = extract_all_features(df)
    return df


def to_model_tensor(df, lookback=128, feature_cols=None):
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("timestamp")]

    data = df[feature_cols].values.astype("float32")

    windows = []
    for i in range(len(data) - lookback):
        windows.append(data[i:i+lookback])

    return torch.tensor(windows)  # shape: (num_windows, lookback, num_features)
