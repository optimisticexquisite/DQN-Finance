#!/usr/bin/env python
"""
Evaluate a trained DQN trading agent on the test split and save
action probabilities at each timestep to a CSV.

- Loads checkpoint produced by _save_agent_checkpoint(...)
- Rebuilds test environment using same CSV, feature columns, lookback, etc.
- Runs greedy policy (argmax Q) with stabilization window.
- Writes one row per environment step:
    [date_time, env_index, action, reward, prob_<action_0>, prob_<action_1>, ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------
# Imports from your existing codebase
# ---------------------------------------------------------------------
# Adjust these to your actual package/module names.
from environment import MarketEnvironment  # your MarketEnvironment class
from agent import DQNAgent           # your DQNAgent class
from agent import AgentConfig        # your AgentConfig dataclass

# If load_nifty_ohlcv / split_dataset are in a different module,
# adjust imports accordingly.


def load_nifty_ohlcv(csv_path: Path) -> pd.DataFrame:
    # Fast path: parse timestamp as datetime; enforce float32 on prices/volume
    df = pd.read_csv(
        csv_path,
        parse_dates=["date_time"],
    )

    # Sanity: expected columns present
    expected = ["date_time","open","low","high","close","volume","sentiment_core__prob_positive_1","sentiment_core__prob_negative_1","sentiment_core__prob_neutral_1","sentiment_core__compound_score_1","sentiment_core__sentiment_confidence_1","market_psychology__fud_index_1","market_psychology__fomo_index_1","market_psychology__hype_index_1","topic_category_confidence__topic_regulation_1","topic_category_confidence__topic_institutional_1","topic_category_confidence__topic_security_breach_1","topic_category_confidence__topic_fed_macro_1","entity_targeted_sentiment__sent_bitcoin_1","entity_targeted_sentiment__sent_exchanges_1","entity_targeted_sentiment__sent_regulators_1","metadata__clickbait_score_1","metadata__has_price_prediction_1","sentiment_core__prob_positive_2","sentiment_core__prob_negative_2","sentiment_core__prob_neutral_2","sentiment_core__compound_score_2","sentiment_core__sentiment_confidence_2","market_psychology__fud_index_2","market_psychology__fomo_index_2","market_psychology__hype_index_2","topic_category_confidence__topic_regulation_2","topic_category_confidence__topic_institutional_2","topic_category_confidence__topic_security_breach_2","topic_category_confidence__topic_fed_macro_2","entity_targeted_sentiment__sent_bitcoin_2","entity_targeted_sentiment__sent_exchanges_2","entity_targeted_sentiment__sent_regulators_2","metadata__clickbait_score_2","metadata__has_price_prediction_2"]

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Ensure no NaNs were introduced
    bad_ts = int(df["date_time"].isna().sum())
    bad_any = int(df[expected].isna().any(axis=1).sum())
    if bad_ts or bad_any:
        raise ValueError(
            f"Parsing created NaNs: timestamp NaT={bad_ts}, rows with any NaN={bad_any}"
        )

    # Sort chronologically and return canonical order
    df = df.sort_values("date_time").reset_index(drop=True)
    return df[expected]

# ---------------------------------------------------------------------
# If split_dataset is not importable, copy it here (identical to training)
# ---------------------------------------------------------------------
def split_dataset(
    data: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
):
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Train/validation ratios must be in (0,1) and sum to less than 1.")

    n_samples = len(data)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_df = data.iloc[:train_end].reset_index(drop=True)
    val_df = data.iloc[train_end:val_end].reset_index(drop=True)
    test_df = data.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------
@dataclass
class LoadedCheckpoint:
    config: AgentConfig
    action_space: Sequence[float]
    feature_columns: Sequence[str]
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    state_dict: Dict[str, torch.Tensor]


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedCheckpoint:
    # Important: weights_only=False because this checkpoint contains more than just raw tensors
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = AgentConfig(**payload["config"])
    action_space = list(payload["action_space"])
    feature_columns = list(payload["feature_columns"])
    normalization = payload["normalization"]
    normalization_mean = np.asarray(normalization["mean"], dtype=np.float32)
    normalization_std = np.asarray(normalization["std"], dtype=np.float32)

    state_dict = payload["state_dict"]

    return LoadedCheckpoint(
        config=config,
        action_space=action_space,
        feature_columns=feature_columns,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        state_dict=state_dict,
    )



# ---------------------------------------------------------------------
# Build test environment from CSV + checkpoint metadata
# ---------------------------------------------------------------------
def build_test_environment(
    csv_path: Path,
    feature_columns: Sequence[str],
    config: AgentConfig,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
) -> tuple[MarketEnvironment, pd.DataFrame]:
    """
    - Loads the full CSV (same as during training).
    - Splits into train/val/test with same ratios.
    - Builds a MarketEnvironment only on the test split, using
      normalization stats from the checkpoint.
    - Returns (test_env, test_df_with_datetime).
    """
    full_df = load_nifty_ohlcv(csv_path)
    train_df, val_df, test_df = split_dataset(full_df)

    # Environment sees only feature columns; we keep full test_df to access date_time
    feature_only_test = test_df[list(feature_columns)].astype(np.float32)

    test_env = MarketEnvironment(
        data=feature_only_test,
        lookback=config.lookback,
        stabilization_window=config.time_window,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )

    return test_env, test_df


# ---------------------------------------------------------------------
# Run agent on test environment and collect action probabilities
# ---------------------------------------------------------------------
def run_agent_and_collect_probs(
    agent: DQNAgent,
    environment: MarketEnvironment,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Runs the agent greedily over the test environment, respecting
    the stabilization window, and records at each step:

    - date_time (from test_df)
    - env_index (cursor)
    - chosen action (float)
    - reward received
    - action probabilities (softmax over Q-values)
    """
    device = agent.device

    records: List[Dict[str, float]] = []

    state = environment.reset()
    stabilization_counter = 0
    cached_action_idx = 0

    # For probability column names
    action_space = list(agent.action_space)
    prob_col_names = [f"prob_action_{a:g}" for a in action_space]

    while not environment.done:
        current_index = environment.current_index  # index in test_df

        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.policy_net(state_tensor).squeeze(0)  # (n_actions,)
            probs = torch.softmax(q_values, dim=-1).cpu().numpy()

        # Greedy action selection with stabilization
        if stabilization_counter == 0:
            action_idx = int(torch.argmax(q_values).item())
            cached_action_idx = action_idx
            stabilization_counter = agent.config.time_window
        else:
            action_idx = cached_action_idx

        action_value = action_space[action_idx]

        # Step environment
        step_result = environment.step(action_value)
        reward = step_result.reward
        next_state = step_result.next_state

        # Build record for this timestep
        row = {
            "env_index": current_index,
            "action": float(action_value),
            "reward": float(reward),
        }

        # attach timestamp if present
        if "date_time" in test_df.columns:
            row["date_time"] = test_df.iloc[current_index]["date_time"]

        # add probabilities per action
        for col_name, p in zip(prob_col_names, probs):
            row[col_name] = p

        records.append(row)

        # prepare for next step
        state = next_state
        stabilization_counter = max(stabilization_counter - 1, 0)

    df = pd.DataFrame.from_records(records)

    # Optional: sort by env_index / date_time just to be safe
    if "date_time" in df.columns:
        df = df.sort_values(["env_index"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def main() -> None:
    # === CONFIGURE THESE PATHS/VALUES ===
    # Path to the CSV used for training
    data_path = Path("/home/rchethan1/DQN-Finance/src/data/btc_hourly_with_sentiment.csv")

    # Path to a specific checkpoint you saved during training
    checkpoint_path = Path("/home/rchethan1/DQN-Finance/src/models/H1/epoch_035.pt")  # adjust as needed

    # Where to write the CSV with action probabilities
    output_csv_path = Path("plots/H1/test_actions_with_probs.csv")

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load checkpoint
    ckpt = load_checkpoint(checkpoint_path, device=device)

    # 2) Build test environment (using same CSV, split, feature columns, etc.)
    test_env, test_df = build_test_environment(
        csv_path=data_path,
        feature_columns=ckpt.feature_columns,
        config=ckpt.config,
        normalization_mean=ckpt.normalization_mean,
        normalization_std=ckpt.normalization_std,
    )

    # 3) Recreate agent and load weights
    state_dim = int(test_env.state_shape[0])
    agent = DQNAgent(
        state_dim=state_dim,
        action_space=ckpt.action_space,
        config=ckpt.config,
        device=str(device),
        seed=None,
    )
    agent.policy_net.load_state_dict(ckpt.state_dict)
    agent.target_net.load_state_dict(ckpt.state_dict)
    agent.policy_net.eval()
    agent.target_net.eval()

    # 4) Run agent on test env and collect probabilities
    results_df = run_agent_and_collect_probs(agent, test_env, test_df)

    # 5) Save to CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Saved test actions and probabilities to: {output_csv_path}")


if __name__ == "__main__":
    main()
