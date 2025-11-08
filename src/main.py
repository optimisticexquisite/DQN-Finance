from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from dqn_finance import DEFAULT_AGENT_PRESETS, MarketEnvironment, create_agent

FEATURE_COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
AGENT_SPECS: Dict[str, Dict[str, int]] = {
    "1D": {"aggregation": 1, "epochs": 25},
    "3D": {"aggregation": 3, "epochs": 25},
    "12D": {"aggregation": 12, "epochs": 25},
}


def _normalise_column_name(name: str) -> str:
    return name.strip().lower().replace(".", "").replace("%", "").replace(" ", "_")


def _clean_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            continue
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("M", "e6", regex=False)
            .str.replace("K", "e3", regex=False)
            .str.replace("B", "e9", regex=False)
            .str.replace("%", "", regex=False)
        )
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_nifty_ohlcv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    raw_df = pd.read_csv(csv_path)
    raw_df.columns = [_normalise_column_name(col) for col in raw_df.columns]

    rename_map = {
        "price": "close",
        "close_price": "close",
        "vol": "volume",
        "vol_": "volume",
        "date": "timestamp",
    }
    for source, target in rename_map.items():
        if source in raw_df.columns:
            raw_df = raw_df.rename(columns={source: target})

    missing = [col for col in FEATURE_COLUMNS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Input CSV must contain OHLCV columns. Missing: {missing}")

    cleaned_df = _clean_numeric_columns(raw_df, FEATURE_COLUMNS)
    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
    cleaned_df = cleaned_df.fillna(method="ffill").dropna().reset_index(drop=True)

    if "timestamp" in cleaned_df.columns:
        cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["timestamp"], errors="coerce")
        cleaned_df = cleaned_df.dropna(subset=["timestamp"])
        cleaned_df = cleaned_df.sort_values("timestamp").reset_index(drop=True)

    cleaned_df = cleaned_df.reset_index(drop=True)
    cleaned_df[list(FEATURE_COLUMNS)] = cleaned_df[list(FEATURE_COLUMNS)].astype(np.float32)

    return cleaned_df[list(FEATURE_COLUMNS) + (["timestamp"] if "timestamp" in cleaned_df.columns else [])]


def aggregate_ohlcv(data: pd.DataFrame, span: int) -> pd.DataFrame:
    if span <= 0:
        raise ValueError("Aggregation span must be positive")
    if span == 1:
        return data.copy()
    if len(data) < span:
        raise ValueError("Data length must exceed aggregation span")

    records: List[Dict[str, float]] = []
    include_timestamp = "timestamp" in data.columns

    for idx in range(span - 1, len(data)):
        window = data.iloc[idx - span + 1 : idx + 1]
        record = {
            "open": float(window.iloc[0]["open"]),
            "high": float(window["high"].max()),
            "low": float(window["low"].min()),
            "close": float(window.iloc[-1]["close"]),
            "volume": float(window["volume"].sum()),
        }
        if include_timestamp:
            record["timestamp"] = window.iloc[-1]["timestamp"]
        records.append(record)

    aggregated = pd.DataFrame(records)
    columns: List[str] = list(FEATURE_COLUMNS)
    if include_timestamp:
        columns.append("timestamp")
    return aggregated[columns]


def split_dataset(
    data: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Train/validation ratios must be in (0,1) and sum to less than 1.")

    n_samples = len(data)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_df = data.iloc[:train_end].reset_index(drop=True)
    val_df = data.iloc[train_end:val_end].reset_index(drop=True)
    test_df = data.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def build_environment_splits(
    data: pd.DataFrame,
    lookback: int,
    time_window: int,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> Tuple[MarketEnvironment, MarketEnvironment, MarketEnvironment]:
    train_df, val_df, test_df = split_dataset(data)

    minimum_required = lookback + time_window + 1
    for name, split_df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        if len(split_df) <= minimum_required:
            raise ValueError(
                f"{name.capitalize()} split is too short (len={len(split_df)}) for lookback={lookback} "
                f"and stabilization_window={time_window}. Need at least {minimum_required + 1} rows."
            )

    feature_only = list(feature_columns)
    train_env = MarketEnvironment(train_df[feature_only], lookback, time_window)
    mean = train_env.normalization_mean
    std = train_env.normalization_std

    val_env = MarketEnvironment(
        val_df[feature_only],
        lookback,
        time_window,
        normalization_mean=mean,
        normalization_std=std,
    )
    test_env = MarketEnvironment(
        test_df[feature_only],
        lookback,
        time_window,
        normalization_mean=mean,
        normalization_std=std,
    )

    return train_env, val_env, test_env


def train_agent_on_dataframe(
    agent_name: str,
    data: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    log_prefix: str | None = None,
    log_metrics: bool = True,
) -> Dict[str, object]:
    config = DEFAULT_AGENT_PRESETS[agent_name]
    epochs = AGENT_SPECS[agent_name]["epochs"]
    feature_columns = tuple(feature_columns)

    for column in feature_columns:
        if column not in data.columns:
            raise ValueError(f"Expected feature column '{column}' in dataframe for agent '{agent_name}'")

    working_data = data.reset_index(drop=True).copy()
    working_data[list(feature_columns)] = working_data[list(feature_columns)].astype(np.float32)

    train_env, val_env, test_env = build_environment_splits(
        working_data, config.lookback, config.time_window, feature_columns=feature_columns
    )

    agent = create_agent(train_env, agent_name)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        agent.optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=5e-6,
        verbose=False,
    )

    history: List[Dict[str, float]] = []
    prefix = log_prefix or agent_name

    for epoch in range(epochs):
        train_metrics = agent.train_epoch(train_env)
        val_metrics = agent.evaluate_epoch(val_env)

        scheduler.step(val_metrics["mse"])
        current_lr = agent.optimizer.param_groups[0]["lr"]

        epoch_summary = {
            "train_loss": train_metrics["avg_loss"],
            "train_epsilon": train_metrics["epsilon"],
            "val_mse": val_metrics["mse"],
            "val_avg_reward": val_metrics["avg_reward"],
            "lr": current_lr,
        }
        history.append(epoch_summary)

        if log_metrics:
            print(
                f"[{prefix}] epoch={epoch + 1:02d} "
                f"train_loss={epoch_summary['train_loss']:.6f} "
                f"val_mse={epoch_summary['val_mse']:.6f} "
                f"val_reward={epoch_summary['val_avg_reward']:.6f} "
                f"lr={current_lr:.2e}"
            )

    test_metrics = agent.evaluate_epoch(test_env)
    if log_metrics:
        print(
            f"[{prefix}] test_mse={test_metrics['mse']:.6f} "
            f"test_reward={test_metrics['avg_reward']:.6f}"
        )

    return {
        "history": history,
        "test": test_metrics,
        "agent": agent,
        "normalization": {
            "mean": train_env.normalization_mean,
            "std": train_env.normalization_std,
        },
        "feature_columns": feature_columns,
    }


def train_agent_for_view(
    agent_name: str,
    aggregation_span: int,
    base_data: pd.DataFrame,
) -> Dict[str, object]:
    aggregated_data = aggregate_ohlcv(base_data, aggregation_span)
    return train_agent_on_dataframe(
        agent_name,
        aggregated_data,
        feature_columns=FEATURE_COLUMNS,
        log_prefix=agent_name,
        log_metrics=True,
    )


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    data_path = Path(__file__).resolve().parent / "data" / "Nifty_50.csv"
    nifty_data = load_nifty_ohlcv(data_path)

    results: Dict[str, Dict[str, object]] = {}
    for agent_name, spec in AGENT_SPECS.items():
        print(f"\n=== Training {agent_name} agent (aggregation span={spec['aggregation']} days) ===")
        results[agent_name] = train_agent_for_view(
            agent_name=agent_name,
            aggregation_span=spec["aggregation"],
            base_data=nifty_data,
        )

    print("\n=== Summary ===")
    for agent_name, metrics in results.items():
        final_epoch = metrics["history"][-1]
        test_metrics = metrics["test"]
        print(
            f"{agent_name}: "
            f"train_loss={final_epoch['train_loss']:.6f}, "
            f"val_mse={final_epoch['val_mse']:.6f}, "
            f"test_mse={test_metrics['mse']:.6f}, "
            f"test_reward={test_metrics['avg_reward']:.6f}"
        )


if __name__ == "__main__":
    main()