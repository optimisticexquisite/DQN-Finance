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

def add_volatility_estimators(df):
    # Parkinson volatility
    df["vol_parkinson"] = ((df["high"] / df["low"]).apply(np.log)) ** 2
    # Garman–Klass volatility
    df["vol_gk"] = (
        0.5 * ((df["high"] / df["low"]).apply(np.log)) ** 2
        - (2 * np.log(2) - 1) * ((df["close"] / df["open"]).apply(np.log)) ** 2
    )
    # Rogers–Satchell volatility
    df["vol_rs"] = (
        np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"])
        + np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"])
    )
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

def add_stochastic(df, k=14, d=3):
    low_min = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(d).mean()
    return df

def add_roc(df, period=10):
    df["roc"] = df["close"].pct_change(period).fillna(0)
    return df

def add_cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    df["cci"] = (tp - ma) / (0.015 * md + 1e-9)
    return df

def add_williams_r(df, period=14):
    highest = df["high"].rolling(period).max()
    lowest = df["low"].rolling(period).min()
    df["williams_r"] = -100 * (highest - df["close"]) / (highest - lowest + 1e-9)
    return df

def add_macd(df):
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df

def add_adx(df, period=14):
    # Directional movement
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    tr = np.max([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=0)
    tr_s = pd.Series(tr).rolling(period).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).sum() / (tr_s + 1e-9)
    minus_di = 100 * pd.Series(minus_dm).rolling(period).sum() / (tr_s + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["adx"] = dx.rolling(period).mean()
    return df


def add_aroon(df, period=25):
    df["aroon_up"] = df["high"].rolling(period).apply(lambda x: float(np.argmax(x)) / period)
    df["aroon_down"] = df["low"].rolling(period).apply(lambda x: float(np.argmin(x)) / period)
    return df

def add_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    return df


def add_accum_dist(df):
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-9)
    df["accdist"] = (clv * df["volume"]).cumsum()
    return df

def add_autocorr(df, lags=[1, 2, 3, 5, 10]):
    for lag in lags:
        df[f"autocorr_{lag}"] = df["return"].rolling(200).apply(
            lambda x: np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag else 0,
            raw=False
        )
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
    df["weekofyear"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["month"] = df["timestamp"].dt.month
    df["weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def extract_all_features(df):
    df = df.copy()

    add_basic_features(df)
    add_rolling_stats(df)
    add_volatility_estimators(df)
    add_rsi(df)
    add_stochastic(df)
    add_roc(df)
    add_cci(df)
    add_williams_r(df)
    add_macd(df)
    add_adx(df)
    add_aroon(df)
    add_obv(df)
    add_accum_dist(df)
    add_autocorr(df)
    add_bollinger(df)
    add_time_features(df)

    # Clean any numerical issues
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df
    
