import numpy as np
import pandas as pd


def add_basic_features(df):
    df["return"] = df["close"].pct_change().fillna(0)
    df["log_return"] = np.log(df["close"]).diff().fillna(0)
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["co_range"] = (df["close"] - df["open"]) / df["open"]
    df["volume_delta"] = df["volume"].diff().fillna(0)
    return df


def add_rolling_stats(df, windows=[5, 10, 20, 50, 100]):
    for w in windows:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
        df[f"std_{w}"] = df["close"].rolling(w).std()
        df[f"vol_{w}"] = df["return"].rolling(w).std()
        df[f"zscore_{w}"] = (df["close"] - df[f"ma_{w}"]) / (df[f"std_{w}"] + 1e-9)
    return df


def add_rsi(df, period=14):
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period).mean()
    ma_down = down.ewm(alpha=1/period).mean()
    rs = ma_up / (ma_down + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df):
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    return df


def add_bollinger(df, window=20):
    ma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    df["bb_upper"] = ma + 2 * std
    df["bb_lower"] = ma - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    return df


def add_time_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def extract_all_features(df):
    df = df.copy()

    add_basic_features(df)
    add_rolling_stats(df)
    add_rsi(df)
    add_macd(df)
    add_bollinger(df)
    add_time_features(df)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df
