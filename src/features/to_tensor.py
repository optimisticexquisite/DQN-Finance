import numpy as np
import pandas as pd
import torch

def to_model_tensor(df: pd.DataFrame, lookback: int = 128):
    """Convert a feature-engineered dataframe into a 3D tensor:
    (num_sequences, lookback, num_features)
    for Transformer / LSTM models.
    """
    # Drop non-numeric columns (timestamp etc.)
    df = df.select_dtypes(include=[np.number]).copy()
    data = df.to_numpy().astype(np.float32)
    N, F = data.shape
    if N < lookback:
        raise ValueError(f"Not enough rows ({N}) for lookback={lookback}")
    sequences = []
    for i in range(N - lookback):
        window = data[i : i + lookback]
        sequences.append(window)
    arr = np.stack(sequences)
    return torch.tensor(arr, dtype=torch.float32)