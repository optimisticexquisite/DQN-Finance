import pandas as pd
from features.bitcoin_features import extract_all_features

def prepare_features(data):
    """Accepts either a CSV path or a DataFrame."""
    
    if isinstance(data, str):
        # it's a file path
        df = pd.read_csv(data, parse_dates=["timestamp"])
    elif isinstance(data, pd.DataFrame):
        # it's already a dataframe
        df = data.copy()
    else:
        raise TypeError("prepare_features expects a file path or DataFrame")

    df = extract_all_features(df)
    return df
