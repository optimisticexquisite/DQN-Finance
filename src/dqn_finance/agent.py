"""DQN agent tailored for financial trading tasks."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm.auto import tqdm

from .environment import MarketEnvironment
from .memory import ReplayMemory
from .network import QNetwork


@dataclass(frozen=True)
class AgentConfig:
    """Hyperparameter bundle for configuring a :class:`DQNAgent`."""

    layer_sizes: Sequence[int]
    activations: Sequence[str]
    learning_rate: float
    replay_memory_size: int
    discount_factor: float
    lookback: int
    time_window: int
    target_update_interval: int
    batch_size: int
    warmup_steps: int = 1_000
    gradient_clip_norm: Optional[float] = 1.0
    temperature_start: float = 1.0
    temperature_decay: float = 0.9995
    temperature_min: float = 0.05


class DQNAgent:
    """Implements the Deep Q-Network algorithm with target network updates."""

    def __init__(
        self,
        state_dim: int,
        action_space: Sequence[float],
        config: AgentConfig,
        *,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if len(action_space) == 0:
            raise ValueError("action_space must contain at least one action")
        if config.layer_sizes[-1] != len(action_space):
            raise ValueError("Output layer size must match the number of actions")

        self.config = config
        self.action_space = tuple(float(a) for a in action_space)
        self._action_to_index: Dict[float, int] = {action: idx for idx, action in enumerate(self.action_space)}

        if len(self._action_to_index) != len(self.action_space):
            raise ValueError("action_space must contain unique values")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)

        self.policy_net = QNetwork(state_dim, config.layer_sizes, config.activations).to(self.device)
        self.target_net = QNetwork(state_dim, config.layer_sizes, config.activations).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayMemory(config.replay_memory_size)

        self._steps_done = 0
        self._temperature = config.temperature_start
        self._cached_action_idx = 0
        self._stabilization_counter = 0

    # ---------------------------------------------------------------------
    # Action selection utilities
    # ---------------------------------------------------------------------
    def select_action(
        self,
        state: np.ndarray,
        *,
        deterministic: bool = False,
        temperature: Optional[float] = None,
    ) -> float:
        """Sample an action according to the softmax distribution over Q-values."""

        state_tensor = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)

        if deterministic:
            action_idx = int(torch.argmax(q_values).item())
        else:
            temp = float(self._temperature if temperature is None else temperature)
            temp = max(temp, 1e-6)
            stabilized = (q_values - torch.max(q_values)) / temp
            probabilities = torch.softmax(stabilized, dim=0).cpu().numpy()
            probabilities = np.clip(probabilities, 1e-9, 1.0)
            probabilities /= probabilities.sum()
            action_idx = int(np.random.choice(len(self.action_space), p=probabilities))

        return self.action_space[action_idx]

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, dtype=torch.float32, device=self.device)

    def _optimize_model(self) -> float:
        return self._optimize_with(
            memory=self.memory,
            policy_net=self.policy_net,
            target_net=self.target_net,
            optimizer=self.optimizer,
        )

    def _optimize_with(
        self,
        *,
        memory: ReplayMemory,
        policy_net: QNetwork,
        target_net: QNetwork,
        optimizer: Optimizer,
    ) -> float:
        transitions = memory.sample(self.config.batch_size)

        states = self._to_tensor(np.stack([t.state for t in transitions]))
        actions = torch.tensor(
            [t.action for t in transitions], dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = self._to_tensor(np.array([t.reward for t in transitions], dtype=np.float32))
        next_states = self._to_tensor(np.stack([t.next_state for t in transitions]))
        dones = torch.tensor([t.done for t in transitions], dtype=torch.bool, device=self.device)

        state_action_values = policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_state_values = target_net(next_states).max(1).values
            next_state_values = next_state_values.masked_fill(dones, 0.0)

        expected_state_action_values = rewards + self.config.discount_factor * next_state_values

        loss = self.loss_fn(state_action_values, expected_state_action_values)

        optimizer.zero_grad()
        loss.backward()
        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), self.config.gradient_clip_norm)
        optimizer.step()

        return float(loss.item())

    def _sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _temperature_decay(self) -> None:
        self._temperature = max(
            self.config.temperature_min,
            self._temperature * self.config.temperature_decay,
        )

    def train_epoch(
        self,
        environment: MarketEnvironment,
        *,
        track_rewards: bool = False,
        track_actions: bool = False,
    ) -> Dict[str, Union[float, List[float]]]:
        """Run a full pass over the environment data.

        Returns
        -------
        Dict[str, Union[float, List[float]]]
            Dictionary containing average training loss, cumulative steps, and
            the current sampling temperature. When ``track_rewards`` is
            ``True``, the per-step realized rewards are included under the
            ``"rewards"`` key. When ``track_actions`` is ``True``, the executed
            actions are returned under ``"actions"``.
        """

        if environment.lookback_period != self.config.lookback:
            raise ValueError("Environment lookback period does not match agent configuration")
        if environment.action_window != self.config.time_window:
            raise ValueError("Environment action window does not match agent configuration")

        state = environment.reset()
        self._stabilization_counter = 0
        epoch_loss: List[float] = []
        reward_log: Optional[List[float]] = [] if (track_rewards or track_actions) else None
        action_log: Optional[List[float]] = [] if track_actions else None
        
        pbar = tqdm(total=getattr(environment, "max_steps", None), desc="Episode", leave=False)
        while not environment.done:
            if self._stabilization_counter == 0:
                action_value = self.select_action(state)
                self._cached_action_idx = self._action_to_index[action_value]
                self._stabilization_counter = self.config.time_window
            else:
                action_value = self.action_space[self._cached_action_idx]

            step_result = environment.step(action_value)

            self.memory.push(
                state,
                self._cached_action_idx,
                step_result.reward,
                step_result.next_state,
                step_result.done,
            )

            state = step_result.next_state
            self._stabilization_counter = max(self._stabilization_counter - 1, 0)

            if self.memory.is_ready(self.config.batch_size) and self._steps_done >= self.config.warmup_steps:
                loss_value = self._optimize_model()
                epoch_loss.append(loss_value)

            self._steps_done += 1
            if self._steps_done % self.config.target_update_interval == 0:
                self._sync_target_network()
            self._temperature_decay()

            if reward_log is not None:
                reward_log.append(step_result.reward)
            if action_log is not None:
                action_log.append(float(action_value))
            # --- tqdm updates ---
            pbar.update(1)

        pbar.close()
        result: Dict[str, Union[float, List[float]]] = {
            "avg_loss": float(np.mean(epoch_loss)) if epoch_loss else 0.0,
            "steps": float(self._steps_done),
            "temperature": float(self._temperature),
        }
        if track_rewards and reward_log is not None:
            result["rewards"] = list(reward_log)
        if track_actions and action_log is not None:
            result["actions"] = list(action_log)
        return result

    def evaluate_epoch(
        self,
        environment: MarketEnvironment,
        *,
        track_rewards: bool = False,
        track_actions: bool = False,
    ) -> Dict[str, Union[float, List[float]]]:
        """Roll out a validation/test episode using the current policy network.

        Returns
        -------
        Dict[str, Union[float, List[float]]]
            Dictionary containing the mean-squared temporal-difference error,
            average realized reward and number of steps traversed. When
            ``track_rewards`` is ``True``, the per-step rewards are included
            under the ``"rewards"`` key. When ``track_actions`` is ``True``,
            the greedy actions taken at each step are returned under the
            ``"actions"`` key.
        """

        if environment.lookback_period != self.config.lookback:
            raise ValueError("Environment lookback period does not match agent configuration")
        if environment.action_window != self.config.time_window:
            raise ValueError("Environment action window does not match agent configuration")

        state = environment.reset()
        stabilization_counter = 0
        cached_action_idx = 0
        td_errors: List[float] = []
        rewards: List[float] = []
        actions: List[float] = [] if track_actions else []

        while not environment.done:
            if stabilization_counter == 0:
                action_value = self.select_action(state, deterministic=False)
                cached_action_idx = self._action_to_index[action_value]
                stabilization_counter = self.config.time_window
            else:
                action_value = self.action_space[cached_action_idx]

            step_result = environment.step(action_value)

            with torch.no_grad():
                state_tensor = self._to_tensor(state).unsqueeze(0)
                q_sa = self.policy_net(state_tensor)[0, cached_action_idx].item()
                if step_result.done:
                    target = step_result.reward
                else:
                    next_tensor = self._to_tensor(step_result.next_state).unsqueeze(0)
                    next_max = self.target_net(next_tensor).max(1).values.item()
                    target = step_result.reward + self.config.discount_factor * next_max
                td_errors.append(float((q_sa - target) ** 2))

            rewards.append(step_result.reward)
            if track_actions:
                actions.append(float(action_value))
            state = step_result.next_state
            stabilization_counter = max(stabilization_counter - 1, 0)

        avg_reward = np.array(rewards) + 1.0
        avg_reward = np.prod(avg_reward)
        result: Dict[str, Union[float, List[float]]] = {
            "mse": float(np.mean(td_errors)) if td_errors else 0.0,
            "avg_reward": float(avg_reward) if rewards else 0.0,
            "steps": float(len(rewards)),
        }
        if track_rewards:
            result["rewards"] = list(rewards)
        if track_actions:
            result["actions"] = list(actions)
        return result

    def fit(self, environment: MarketEnvironment, epochs: int) -> List[Dict[str, float]]:
        """Train the agent for a number of epochs over the environment."""

        metrics: List[Dict[str, float]] = []
        for _ in tqdm(range(epochs)):
            metrics.append(self.train_epoch(environment))
        return metrics


def build_default_action_space() -> Tuple[float, float, float]:
    """Default discrete action set: short, flat, and long positions."""

    return (-1.0, 0.0, 1.0)


DEFAULT_AGENT_PRESETS: Dict[str, AgentConfig] = {
    "1D": AgentConfig(
        layer_sizes=(400, 200, 100, 3),
        activations=("Tanh", "Tanh", "Tanh", "Linear"),
        learning_rate=1e-4,
        replay_memory_size=120_000,
        discount_factor=0.6,
        lookback=300,
        time_window=20,
        target_update_interval=1000,
        batch_size=64,
        warmup_steps=1_500,
        temperature_start=1.5,
        temperature_decay=0.9995,
        temperature_min=0.05,
    ),
    "3D": AgentConfig(
        layer_sizes=(300, 200, 100, 3),
        activations=("Tanh", "Tanh", "Tanh", "Linear"),
        learning_rate=1e-4,
        replay_memory_size=40_000,
        discount_factor=0.5,
        lookback=200,
        time_window=10,
        target_update_interval=600,
        batch_size=64,
        warmup_steps=1_200,
        temperature_start=1.5,
        temperature_decay=0.9995,
        temperature_min=0.05,
    ),
    "12D": AgentConfig(
        layer_sizes=(400, 200, 100, 3),
        activations=("tanh", "tanh", "tanh", "Linear"),
        learning_rate=1e-4,
        replay_memory_size=800,
        discount_factor=0.45,
        lookback=200,
        time_window=5,
        target_update_interval=400,
        batch_size=64,
        warmup_steps=0,
        temperature_start=1.25,
        temperature_decay=0.999,
        temperature_min=0.05,
    ),
    "H1": AgentConfig(
        layer_sizes=(200, 100, 3),
        activations=("tanh", "tanh", "Linear"),
        learning_rate=1e-4,
        replay_memory_size = 800,
        discount_factor=0.45,
        lookback=200,
        time_window=5,
        target_update_interval=400,
        batch_size=64,
        warmup_steps=0,
        temperature_start=1.25,
        temperature_decay=0.999,
        temperature_min=0.05,
    ),
    "M15": AgentConfig(
        layer_sizes=(300, 200, 100, 3),
        activations=("Tanh", "Tanh", "Tanh", "Linear"),
        learning_rate=2.5e-4,
        replay_memory_size=40_000,
        discount_factor=0.5,
        lookback=200,
        time_window=10,
        target_update_interval=1_000,
        batch_size=64,
        warmup_steps=5_000,
        temperature_start=1.25,
        temperature_decay=0.9995,
        temperature_min=0.05,
    ),
    "M5": AgentConfig(
        layer_sizes=(400, 200, 100, 3),
        activations=("Tanh", "Tanh", "Tanh", "Linear"),
        learning_rate=2.5e-4,
        replay_memory_size=120_000,
        discount_factor=0.6,
        lookback=300,
        time_window=20,
        target_update_interval=1_000,
        batch_size=128,
        warmup_steps=10_000,
        temperature_start=1.5,
        temperature_decay=0.9997,
        temperature_min=0.05,
    ),
}


def create_agent(
    environment: MarketEnvironment,
    preset: str,
    *,
    action_space: Optional[Sequence[float]] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> DQNAgent:
    """Instantiate an agent using one of the predefined presets."""

    if preset not in DEFAULT_AGENT_PRESETS:
        known = ", ".join(DEFAULT_AGENT_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available options: {known}")

    config = DEFAULT_AGENT_PRESETS[preset]
    if environment.lookback_period != config.lookback:
        raise ValueError(
            "Environment lookback period does not match the preset configuration. "
            "Consider rebuilding the environment with the preset's lookback value."
        )
    if environment.action_window != config.time_window:
        raise ValueError(
            "Environment stabilization window does not match the preset configuration. "
            "Adjust the environment or choose a matching preset."
        )

    chosen_action_space = action_space if action_space is not None else build_default_action_space()
    state_dim = int(environment.state_shape[0])
    return DQNAgent(
        state_dim=state_dim,
        action_space=chosen_action_space,
        config=config,
        device=device,
        seed=seed,
    )
