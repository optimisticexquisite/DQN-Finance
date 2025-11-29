import pandas as pd
import torch
from features.bitcoin_features import extract_all_features

def prepare_features(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = extract_all_features(df)
    return df


