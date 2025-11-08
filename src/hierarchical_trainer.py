from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch

from dqn_finance.agent import DQNAgent
from dqn_finance.environment import MarketEnvironment

from main import (
    AGENT_SPECS,
    FEATURE_COLUMNS,
    aggregate_ohlcv,
    load_nifty_ohlcv,
    train_agent_on_dataframe,
)


MODELS_DIR = Path(__file__).resolve().parent / "models"


def ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_agent_weights(agent: DQNAgent, filename: str, *, label: str) -> None:
    ensure_models_dir()
    path = MODELS_DIR / filename

    state_dict = {key: tensor.detach().cpu() for key, tensor in agent.policy_net.state_dict().items()}
    payload = {
        "state_dict": state_dict,
        "config": asdict(agent.config),
        "action_space": list(agent.action_space),
    }

    torch.save(payload, path)
    print(f"Saved {label} weights to {path}")


def compute_action_series(
    agent: DQNAgent,
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    normalization: Dict[str, np.ndarray],
) -> np.ndarray:
    evaluation_mode = agent.policy_net.training
    agent.policy_net.eval()

    env = MarketEnvironment(
        data[feature_columns],
        agent.config.lookback,
        agent.config.time_window,
        normalization_mean=normalization["mean"],
        normalization_std=normalization["std"],
    )

    actions = np.zeros(len(data), dtype=np.float32)
    state = env.reset()
    stabilization_counter = 0
    cached_action_idx = 0

    while not env.done:
        current_index = env.current_index
        if stabilization_counter == 0:
            action_value = agent.select_action(state, epsilon=0.0)
            cached_action_idx = agent._action_to_index[action_value]
            stabilization_counter = agent.config.time_window
        else:
            action_value = agent.action_space[cached_action_idx]

        actions[current_index] = action_value
        step_result = env.step(action_value)
        state = step_result.next_state
        stabilization_counter = max(stabilization_counter - 1, 0)

    if evaluation_mode:
        agent.policy_net.train()

    return actions


def add_prev_action_column(data: pd.DataFrame, actions: np.ndarray, column_name: str) -> pd.DataFrame:
    result = data.copy()
    shifted = np.roll(actions, 1).astype(np.float32)
    if len(shifted) > 0:
        shifted[0] = 0.0
    result[column_name] = shifted
    return result


def merge_feature_by_timestamp(
    target: pd.DataFrame,
    source: pd.DataFrame,
    source_column: str,
    dest_column: str,
    *,
    default_value: float = 0.0,
) -> pd.DataFrame:
    result = target.copy()
    if "timestamp" in target.columns and "timestamp" in source.columns:
        lookup = source[["timestamp", source_column]].drop_duplicates(subset="timestamp", keep="last")
        merged = result.merge(lookup, on="timestamp", how="left")
        merged[dest_column] = merged[source_column].fillna(default_value).astype(np.float32)
        if source_column != dest_column:
            merged = merged.drop(columns=[source_column])
        return merged

    values = np.full(len(result), default_value, dtype=np.float32)
    src_values = source[source_column].to_numpy(dtype=np.float32)
    overlap = min(len(values), len(src_values))
    if overlap > 0:
        values[-overlap:] = src_values[-overlap:]
    result[dest_column] = values
    return result


def train_individual_agents(base_data: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    individual_results: Dict[str, Dict[str, object]] = {}
    for agent_name, spec in AGENT_SPECS.items():
        print(
            f"\n=== Training independent {agent_name} agent "
            f"(aggregation span={spec['aggregation']} days) ==="
        )
        aggregated = aggregate_ohlcv(base_data, spec["aggregation"])
        result = train_agent_on_dataframe(
            agent_name,
            aggregated,
            feature_columns=FEATURE_COLUMNS,
            log_prefix=f"{agent_name}-IND",
            log_metrics=True,
        )
        result["data"] = aggregated
        individual_results[agent_name] = result
        save_agent_weights(result["agent"], f"individual_{agent_name}.pt", label=f"{agent_name} (independent)")
    return individual_results


def train_hierarchical_agents(
    base_data: pd.DataFrame, individual_results: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, object]]:
    hierarchical_results: Dict[str, Dict[str, object]] = {}

    print("\n=== Building hierarchical stack: 12D ➜ 3D ➜ 1D ===")

    # Stage 1: 12D agent (reuse independent training)
    twelve_result = individual_results["12D"]
    hierarchical_results["12D"] = twelve_result
    save_agent_weights(twelve_result["agent"], "hierarchical_12D.pt", label="12D (hierarchical)")

    aggregated_12d = aggregate_ohlcv(base_data, AGENT_SPECS["12D"]["aggregation"])
    actions_12d = compute_action_series(
        twelve_result["agent"],
        aggregated_12d,
        twelve_result["feature_columns"],
        twelve_result["normalization"],
    )
    aggregated_12d = add_prev_action_column(aggregated_12d, actions_12d, "prev_action_12d")

    # Stage 2: 3D agent conditioned on previous 12D action
    aggregated_3d = aggregate_ohlcv(base_data, AGENT_SPECS["3D"]["aggregation"])
    aggregated_3d = merge_feature_by_timestamp(
        aggregated_3d,
        aggregated_12d,
        "prev_action_12d",
        "prev_action_12d",
        default_value=0.0,
    )
    feature_columns_3d = list(FEATURE_COLUMNS) + ["prev_action_12d"]

    print("\n=== Training hierarchical 3D agent (with 12D context) ===")
    result_3d = train_agent_on_dataframe(
        "3D",
        aggregated_3d,
        feature_columns=feature_columns_3d,
        log_prefix="3D-HIER",
        log_metrics=True,
    )
    hierarchical_results["3D"] = result_3d
    save_agent_weights(result_3d["agent"], "hierarchical_3D.pt", label="3D (hierarchical)")

    actions_3d = compute_action_series(
        result_3d["agent"],
        aggregated_3d,
        result_3d["feature_columns"],
        result_3d["normalization"],
    )
    aggregated_3d = add_prev_action_column(aggregated_3d, actions_3d, "prev_action_3d")

    # Stage 3: 1D agent conditioned on previous 3D and 12D actions
    daily_data = merge_feature_by_timestamp(
        base_data,
        aggregated_12d,
        "prev_action_12d",
        "prev_action_12d",
        default_value=0.0,
    )
    daily_data = merge_feature_by_timestamp(
        daily_data,
        aggregated_3d,
        "prev_action_3d",
        "prev_action_3d",
        default_value=0.0,
    )

    feature_columns_1d = list(FEATURE_COLUMNS) + ["prev_action_3d", "prev_action_12d"]

    print("\n=== Training hierarchical 1D agent (with 3D & 12D context) ===")
    result_1d = train_agent_on_dataframe(
        "1D",
        daily_data,
        feature_columns=feature_columns_1d,
        log_prefix="1D-HIER",
        log_metrics=True,
    )
    hierarchical_results["1D"] = result_1d
    save_agent_weights(result_1d["agent"], "hierarchical_1D.pt", label="1D (hierarchical)")

    return hierarchical_results


def summarize_results(title: str, results: Dict[str, Dict[str, object]]) -> None:
    print(f"\n=== {title} Summary ===")
    for agent_name, metrics in results.items():
        history = metrics["history"]
        if not history:
            print(f"{agent_name}: no training history recorded.")
            continue
        final_epoch = history[-1]
        test_metrics = metrics["test"]
        print(
            f"{agent_name}: "
            f"train_loss={final_epoch['train_loss']:.6f}, "
            f"val_mse={final_epoch['val_mse']:.6f}, "
            f"test_mse={test_metrics['mse']:.6f}, "
            f"test_reward={test_metrics['avg_reward']:.6f}"
        )


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    data_path = Path(__file__).resolve().parent / "data" / "Nifty_50.csv"
    base_data = load_nifty_ohlcv(data_path)

    individual_results = train_individual_agents(base_data)
    summarize_results("Independent Agents", individual_results)

    hierarchical_results = train_hierarchical_agents(base_data, individual_results)
    summarize_results("Hierarchical Agents", hierarchical_results)


if __name__ == "__main__":
    main()

