"""Evaluate saved H1 checkpoints on the test split and plot balance curves."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from dqn_finance import create_agent
from dqn_finance.agent import DEFAULT_AGENT_PRESETS
from main import (
    AGENT_SPECS,
    FEATURE_COLUMNS,
    INITIAL_BALANCE,
    MODEL_DIR,
    _balance_trajectory_from_rewards,
    _plot_balance_with_actions,
    aggregate_ohlcv,
    build_environment_splits,
    load_nifty_ohlcv,
)


CHECKPOINT_DIR = MODEL_DIR / "H1"
OUTPUT_DIR = Path(__file__).resolve().parent / "plots" / "H1_eval"


def _load_checkpoints() -> Iterable[Path]:
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(p for p in CHECKPOINT_DIR.glob("*.pt") if p.is_file())


def _prepare_environments():
    data_path = Path(__file__).resolve().parent / "data" / "SP_5min.csv"
    base_data = load_nifty_ohlcv(data_path)
    aggregation = AGENT_SPECS["H1"]["aggregation"]
    aggregated = aggregate_ohlcv(base_data, aggregation)

    config = DEFAULT_AGENT_PRESETS["H1"]
    train_env, _, test_env = build_environment_splits(
        aggregated,
        lookback=config.lookback,
        time_window=config.time_window,
        feature_columns=FEATURE_COLUMNS,
    )
    return train_env, test_env


def evaluate_checkpoints() -> None:
    checkpoints = list(_load_checkpoints())
    if not checkpoints:
        print(f"No checkpoints found in {CHECKPOINT_DIR}")
        return

    train_env, test_env = _prepare_environments()
    agent = create_agent(train_env, "H1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ckpt_path in checkpoints:
        payload = torch.load(ckpt_path, map_location=agent.device, weights_only=False)
        state_dict = payload.get("state_dict")
        if state_dict is None:
            print(f"Skipping {ckpt_path}: missing state_dict")
            continue

        agent.policy_net.load_state_dict(state_dict)
        agent.target_net.load_state_dict(state_dict)

        metrics = agent.evaluate_epoch(
            test_env,
            track_rewards=True,
            track_actions=True,
        )
        rewards = metrics.get("rewards")
        actions = metrics.get("actions")
        if not isinstance(rewards, list) or not rewards:
            print(f"{ckpt_path.name}: no rewards logged; skipping plot.")
            continue
        if not isinstance(actions, list) or len(actions) != len(rewards):
            print(f"{ckpt_path.name}: action log missing or mismatched; skipping plot.")
            continue

        balances = _balance_trajectory_from_rewards(
            rewards,
            initial_balance=INITIAL_BALANCE,
        )
        run_dir = OUTPUT_DIR / ckpt_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        plot_path = run_dir / "test_balance.png"
        _plot_balance_with_actions(
            balances,
            actions,
            plot_path,
            title=f"H1 Test Balance ({ckpt_path.stem})",
        )
        print(
            f"{ckpt_path.name}: mse={metrics['mse']:.6f} "
            f"reward={metrics['avg_reward']:.6f} plot={plot_path}"
        )


if __name__ == "__main__":
    evaluate_checkpoints()
