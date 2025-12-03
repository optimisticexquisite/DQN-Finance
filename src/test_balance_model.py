from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from dqn_finance import (
    DEFAULT_AGENT_PRESETS,
    DQNAgent,
    MarketEnvironment,
    create_agent,
)

from main import (
    load_nifty_ohlcv,
    split_dataset,
    FEATURE_COLUMNS,
)
N = 58
INITIAL_BALANCE: float = 100.0

CHECKPOINT_PATH: Path = (
    Path(__file__).resolve().parent
    / "models_128_4_128"
    / "H1"
    / f"epoch_0{N}.pt"
)

DATA_PATH: Path = (
    Path(__file__).resolve().parent
    / "data"
    / "btc_hourly_with_sentiment.csv"
)

PLOT_PATH: Path = (
    Path(__file__).resolve().parent
    / "plots"
    / "H1"
    / f"test_balance_epoch_0{N}.png"
)


def balance_from_rewards(
    rewards: Sequence[float],
    initial_balance: float = INITIAL_BALANCE,
) -> np.ndarray:
    """
    Convert per-step returns (reward_t, interpreted as fractional returns)
    into a balance trajectory starting from initial_balance.
    """
    reward_array = np.asarray(rewards, dtype=np.float64)
    balances = np.empty(reward_array.size + 1, dtype=np.float64)
    balances[0] = initial_balance
    if reward_array.size:
        balances[1:] = initial_balance * np.cumprod(1.0 + reward_array, dtype=np.float64)
    else:
        balances[1:] = initial_balance
    return balances


def plot_balance(balances: np.ndarray, output_path: Path, title: str) -> None:
    steps = np.arange(balances.size)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, balances, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Balance")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load raw hourly data (same as in your training script)
    df = load_nifty_ohlcv(DATA_PATH)

    # 2. Use the same split as training: train / val / test
    _, test_df, _ = split_dataset(df)  # ratios default to 0.7 / 0.15 / 0.15

    # 3. Load checkpoint
    checkpoint: Dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
        weights_only=False,  # <- important for old-style checkpoints with numpy objects
    )

    state_dict = checkpoint["state_dict"]
    feature_columns: List[str] = list(checkpoint["feature_columns"])
    normalization = checkpoint["normalization"]
    config_dict: Dict = checkpoint["config"]
    print("Loaded checkpoint config:", config_dict)

    lookback = config_dict["lookback"]
    time_window = config_dict["time_window"]

    # 4. Build test environment using saved normalization and feature columns
    missing = [c for c in feature_columns if c not in test_df.columns]
    if missing:
        raise ValueError(f"Test dataframe is missing feature columns: {missing}")

    test_env = MarketEnvironment(
        test_df[feature_columns],
        lookback,
        time_window,
        normalization_mean=normalization["mean"],
        normalization_std=normalization["std"],
    )

    # 5. Re-create agent and load weights
    agent_name = "H1"
    agent: DQNAgent = create_agent(test_env, agent_name)
    agent.policy_net.load_state_dict(state_dict)
    agent.policy_net.eval()

    # 6. Evaluate on test env and capture per-step rewards
    test_metrics = agent.evaluate_epoch(
        test_env,
        track_rewards=True,
        track_actions=False,  # not needed for simple balance plot
    )

    rewards = test_metrics.get("rewards")
    if not isinstance(rewards, list) or not rewards:
        print("No rewards recorded during test evaluation; cannot plot balance.")
        return

    # 7. Compute balance trajectory starting from 100 units
    balances = balance_from_rewards(rewards, initial_balance=INITIAL_BALANCE)

    # 8. Plot balance vs steps
    plot_balance(
        balances,
        PLOT_PATH,
        title="H1 Test Balance Trajectory (epoch 10, start=100)",
    )

    print(f"Saved test balance plot to {PLOT_PATH}")
    print(f"Final balance: {balances[-1]:.4f}")


if __name__ == "__main__":
    main()
