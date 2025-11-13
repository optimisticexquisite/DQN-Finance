from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

from dqn_finance import DEFAULT_AGENT_PRESETS, DQNAgent, MarketEnvironment, create_agent

FEATURE_COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
AGENT_SPECS: Dict[str, Dict[str, int]] = {
    # "1D": {"aggregation": 1, "epochs": 30},
    # "3D": {"aggregation": 3, "epochs": 60},
    # "12D": {"aggregation": 12, "epochs": 200},
    "H1": {"aggregation": 12, "epochs": 200},
}

INITIAL_BALANCE: float = 100.0
MODEL_DIR: Path = Path(__file__).resolve().parent / "models"

ACTION_STYLES: Tuple[Tuple[float, str, str], ...] = (
    (-1.0, "Sell", "#d62728"),
    (0.0, "Hold", "#1f77b4"),
    (1.0, "Buy", "#2ca02c"),
)
ACTION_COLOR_MAP: Dict[float, str] = {value: color for value, _, color in ACTION_STYLES}
ACTION_LABEL_MAP: Dict[float, str] = {value: label for value, label, _ in ACTION_STYLES}


def _current_timestamp_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _annotate_plot_with_timestamp(ax: Axes, *, timestamp: str | None = None) -> str:
    label = timestamp or _current_timestamp_label()
    ax.text(
        0.995,
        0.02,
        f"Generated {label}",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
        ha="right",
        va="bottom",
        alpha=0.85,
    )
    return label


def _save_agent_checkpoint(
    agent: DQNAgent,
    checkpoint_path: Path,
    *,
    normalization: Dict[str, np.ndarray],
    feature_columns: Sequence[str],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: tensor.detach().cpu()
        for key, tensor in agent.policy_net.state_dict().items()
    }
    payload = {
        "state_dict": state_dict,
        "config": asdict(agent.config),
        "action_space": list(agent.action_space),
        "feature_columns": list(feature_columns),
        "normalization": {
            "mean": normalization["mean"],
            "std": normalization["std"],
        },
    }
    torch.save(payload, checkpoint_path)


def _balance_trajectory_from_rewards(
    rewards: Sequence[float], *, initial_balance: float = INITIAL_BALANCE
) -> np.ndarray:
    reward_array = np.asarray(rewards, dtype=np.float64)
    balances = np.empty(reward_array.size + 1, dtype=np.float64)
    balances[0] = initial_balance
    if reward_array.size:
        balances[1:] = initial_balance * np.cumprod(1.0 + reward_array, dtype=np.float64)
    else:
        balances[1:] = initial_balance
    return balances


def _plot_balance_with_actions(
    balances: np.ndarray,
    actions: Sequence[float],
    output_path: Path,
    *,
    title: str,
) -> None:
    if balances.size != len(actions) + 1:
        raise ValueError("Number of actions must be exactly one less than balance points")

    steps = np.arange(balances.size)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, balances, color="black", linewidth=1.5, label="Balance")
    action_colors = [ACTION_COLOR_MAP.get(float(action), "#7f7f7f") for action in actions]
    ax.scatter(steps[1:], balances[1:], c=action_colors, s=25, alpha=0.9)

    legend_handles = [Line2D([0], [0], color="black", linewidth=1.5, label="Balance")]
    for value, label, color in ACTION_STYLES:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=color,
                label=label,
                markersize=8,
            )
        )

    ax.legend(handles=legend_handles, loc="best")
    ax.set_xlabel("Step")
    ax.set_ylabel("Balance")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    _annotate_plot_with_timestamp(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


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
    # Fast path: parse timestamp as datetime; enforce float32 on prices/volume
    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],
        dtype={
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "volume": "float32",
        },
    )

    # Sanity: expected columns present
    expected = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Ensure no NaNs were introduced
    bad_ts = int(df["timestamp"].isna().sum())
    bad_any = int(df[expected].isna().any(axis=1).sum())
    if bad_ts or bad_any:
        raise ValueError(
            f"Parsing created NaNs: timestamp NaT={bad_ts}, rows with any NaN={bad_any}"
        )

    # Sort chronologically and return canonical order
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[expected]


def aggregate_ohlcv(data: pd.DataFrame, span: int) -> pd.DataFrame:
    if span <= 0:
        raise ValueError("Aggregation span must be positive")
    if span == 1:
        return data.copy()
    if len(data) < span:
        raise ValueError("Data length must exceed aggregation span")

    records: List[Dict[str, float]] = []
    include_timestamp = "timestamp" in data.columns

    for idx in range(span - 1, len(data), span):
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
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

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
    plot_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> Dict[str, object]:
    config = DEFAULT_AGENT_PRESETS[agent_name]
    epochs = AGENT_SPECS[agent_name]["epochs"]
    feature_columns = tuple(feature_columns)

    agent_plot_dir = plot_dir / agent_name if plot_dir is not None else None
    training_plot_dir = None
    validation_plot_dir = None
    if agent_plot_dir is not None:
        agent_plot_dir.mkdir(parents=True, exist_ok=True)
        training_plot_dir = agent_plot_dir / "training"
        training_plot_dir.mkdir(parents=True, exist_ok=True)
        validation_plot_dir = agent_plot_dir / "validation"
        validation_plot_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for column in feature_columns:
        if column not in data.columns:
            raise ValueError(f"Expected feature column '{column}' in dataframe for agent '{agent_name}'")
    # print(len(data))
    working_data = data.reset_index(drop=True).copy()
    working_data[list(feature_columns)] = working_data[list(feature_columns)].astype(np.float32)
    # print(len(working_data))

    train_env, val_env, test_env = build_environment_splits(
        working_data, config.lookback, config.time_window, feature_columns=feature_columns
    )
    normalization = {
        "mean": train_env.normalization_mean,
        "std": train_env.normalization_std,
    }

    agent = create_agent(train_env, agent_name)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        agent.optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=5e-6,
    )

    history: List[Dict[str, float]] = []
    prefix = log_prefix or agent_name

    for epoch in range(epochs):
        train_metrics = agent.train_epoch(
            train_env,
            track_rewards=training_plot_dir is not None,
            track_actions=training_plot_dir is not None,
        )
        val_metrics = agent.evaluate_epoch(
            val_env,
            track_rewards=True,
            track_actions=validation_plot_dir is not None,
        )

        scheduler.step(val_metrics["mse"])
        current_lr = agent.optimizer.param_groups[0]["lr"]

        if training_plot_dir is not None:
            train_rewards = train_metrics.get("rewards")
            train_actions = train_metrics.get("actions")
            if (
                isinstance(train_rewards, list)
                and isinstance(train_actions, list)
                and train_rewards
                and len(train_rewards) == len(train_actions)
            ):
                train_balances = _balance_trajectory_from_rewards(
                    train_rewards,
                    initial_balance=INITIAL_BALANCE,
                )
                train_plot_path = training_plot_dir / f"epoch_{epoch + 1:02d}_training_balance.png"
                _plot_balance_with_actions(
                    train_balances,
                    train_actions,
                    train_plot_path,
                    title=f"{agent_name} Training Balance (Epoch {epoch + 1})",
                )

        if validation_plot_dir is not None:
            val_rewards = val_metrics.get("rewards")
            val_actions = val_metrics.get("actions")
            if (
                isinstance(val_rewards, list)
                and isinstance(val_actions, list)
                and val_rewards
                and len(val_rewards) == len(val_actions)
            ):
                balances = _balance_trajectory_from_rewards(val_rewards, initial_balance=INITIAL_BALANCE)
                plot_path = validation_plot_dir / f"epoch_{epoch + 1:02d}_validation_balance.png"
                _plot_balance_with_actions(
                    balances,
                    val_actions,
                    plot_path,
                    title=f"{agent_name} Validation Balance (Epoch {epoch + 1})",
                )
        val_avg_reward = val_metrics["avg_reward"]
        epoch_summary = {
            "train_loss": train_metrics["avg_loss"],
            "temperature": train_metrics["temperature"],
            "val_mse": val_metrics["mse"],
            "val_avg_reward": val_avg_reward,
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
        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
            _save_agent_checkpoint(
                agent,
                checkpoint_path,
                normalization=normalization,
                feature_columns=feature_columns,
            )
            if log_metrics:
                print(f"[{prefix}] saved checkpoint to {checkpoint_path}")

    test_metrics = agent.evaluate_epoch(
        test_env,
        track_rewards=True,
        track_actions=agent_plot_dir is not None,
    )
    if log_metrics:
        print(
            f"[{prefix}] test_mse={test_metrics['mse']:.6f} "
            f"test_reward={test_metrics['avg_reward']:.6f}"
        )

    return {
        "history": history,
        "test": test_metrics,
        "agent": agent,
        "normalization": normalization,
        "feature_columns": feature_columns,
    }


def train_agent_for_view(
    agent_name: str,
    aggregation_span: int,
    base_data: pd.DataFrame,
    *,
    plot_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> Dict[str, object]:
    # print(len(base_data))
    aggregated_data = aggregate_ohlcv(base_data, aggregation_span)
    # print(len(aggregated_data))
    return train_agent_on_dataframe(
        agent_name,
        aggregated_data,
        feature_columns=FEATURE_COLUMNS,
        log_prefix=agent_name,
        log_metrics=True,
        plot_dir=plot_dir,
        checkpoint_dir=checkpoint_dir,
    )


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    data_path = Path(__file__).resolve().parent / "data" / "SP_5min.csv"
    nifty_data = load_nifty_ohlcv(data_path)
    # print(len(nifty_data))

    plot_dir = Path(__file__).resolve().parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, object]] = {}
    for agent_name, spec in AGENT_SPECS.items():
        print(f"\n=== Training {agent_name} agent (aggregation span={spec['aggregation']} days) ===")
        agent_checkpoint_dir = MODEL_DIR / agent_name
        results[agent_name] = train_agent_for_view(
            agent_name=agent_name,
            aggregation_span=spec["aggregation"],
            base_data=nifty_data,
            plot_dir=plot_dir,
            checkpoint_dir=agent_checkpoint_dir,
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
        rewards = test_metrics.get("rewards")
        if not isinstance(rewards, list) or not rewards:
            print(f"{agent_name}: no test rewards recorded; skipping plot.")
            continue

        balances = _balance_trajectory_from_rewards(rewards, initial_balance=INITIAL_BALANCE)
        actions = test_metrics.get("actions")
        agent_plot_dir = plot_dir / agent_name
        agent_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = agent_plot_dir / "test_balance.png"

        if isinstance(actions, list) and len(actions) == len(rewards):
            _plot_balance_with_actions(
                balances,
                actions,
                plot_path,
                title=f"{agent_name} Test Balance Trajectory",
            )
        else:
            steps = np.arange(balances.size)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, balances, color="black", linewidth=1.5, label="Balance")
            ax.set_title(f"{agent_name} Test Balance Trajectory")
            ax.set_xlabel("Step")
            ax.set_ylabel("Balance")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            _annotate_plot_with_timestamp(ax)
            fig.tight_layout()
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)
        print(f"{agent_name}: saved test balance plot to {plot_path}")


if __name__ == "__main__":
    main()
