"""DQN agent tailored for financial trading tasks."""

from __future__ import annotations
from tqdm import tqdm

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

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
    epsilon_start: float
    epsilon_decay: float
    epsilon_min: float
    lookback: int
    time_window: int
    target_update_interval: int
    batch_size: int
    warmup_steps: int = 1_000
    gradient_clip_norm: Optional[float] = 1.0


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
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = ReplayMemory(config.replay_memory_size)

        self._steps_done = 0
        self._epsilon = config.epsilon_start
        self._cached_action_idx = 0
        self._stabilization_counter = 0

    # ---------------------------------------------------------------------
    # Action selection utilities
    # ---------------------------------------------------------------------
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> float:
        """Return an action following an epsilon-greedy policy."""

        eps = self._epsilon if epsilon is None else epsilon
        if np.random.rand() < eps:
            action_idx = np.random.randint(len(self.action_space))
        else:
            state_tensor = self._to_tensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = int(torch.argmax(q_values, dim=1).item())
        return self.action_space[action_idx]

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, dtype=torch.float32, device=self.device)

    def _optimize_model(self) -> float:
        transitions = self.memory.sample(self.config.batch_size)

        states = self._to_tensor(np.stack([t.state for t in transitions]))
        actions = torch.tensor(
            [t.action for t in transitions], dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = self._to_tensor(np.array([t.reward for t in transitions], dtype=np.float32))
        next_states = self._to_tensor(np.stack([t.next_state for t in transitions]))
        dones = torch.tensor([t.done for t in transitions], dtype=torch.bool, device=self.device)

        state_action_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1).values
            next_state_values = next_state_values.masked_fill(dones, 0.0)

        expected_state_action_values = rewards + self.config.discount_factor * next_state_values

        loss = self.loss_fn(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()

        return float(loss.item())

    def _sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _epsilon_decay(self) -> None:
        self._epsilon = max(self.config.epsilon_min, self._epsilon * self.config.epsilon_decay)

    def train_epoch(self, environment: MarketEnvironment) -> Dict[str, float]:
        """Run a full pass over the environment data."""

        if environment.lookback_period != self.config.lookback:
            raise ValueError("Environment lookback period does not match agent configuration")
        if environment.action_window != self.config.time_window:
            raise ValueError("Environment action window does not match agent configuration")

        state = environment.reset()
        self._stabilization_counter = 0
        epoch_loss: List[float] = []

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

            self._epsilon_decay()

        return {
            "avg_loss": float(np.mean(epoch_loss)) if epoch_loss else 0.0,
            "steps": float(self._steps_done),
            "epsilon": float(self._epsilon),
        }

    def evaluate_epoch(self, environment: MarketEnvironment) -> Dict[str, float]:
        """Run a greedy rollout over the environment without updating parameters.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the mean-squared temporal-difference error,
            average realized reward and number of steps traversed.
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

        while not environment.done:
            if stabilization_counter == 0:
                action_value = self.select_action(state, epsilon=0.0)
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
            state = step_result.next_state
            stabilization_counter = max(stabilization_counter - 1, 0)

        return {
            "mse": float(np.mean(td_errors)) if td_errors else 0.0,
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "steps": float(len(rewards)),
        }

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
        epsilon_start=1.0,
        epsilon_decay=0.99995,
        epsilon_min=1e-6,
        lookback=300,
        time_window=20,
        target_update_interval=750,
        batch_size=64,
        warmup_steps=1_500,
    ),
    "3D": AgentConfig(
        layer_sizes=(300, 200, 100, 3),
        activations=("Tanh", "Tanh", "Tanh", "Linear"),
        learning_rate=1e-4,
        replay_memory_size=40_000,
        discount_factor=0.5,
        epsilon_start=1.0,
        epsilon_decay=0.9999,
        epsilon_min=1e-6,
        lookback=200,
        time_window=10,
        target_update_interval=600,
        batch_size=64,
        warmup_steps=1_200,
    ),
    "12D": AgentConfig(
        layer_sizes=(300, 150, 3),
        activations=("Tanh", "Tanh", "Linear"),
        learning_rate=1e-4,
        replay_memory_size=10_000,
        discount_factor=0.45,
        epsilon_start=1.0,
        epsilon_decay=0.999,
        epsilon_min=1e-6,
        lookback=60,
        time_window=2,
        target_update_interval=500,
        batch_size=64,
        warmup_steps=900,
    ),
    "H1": AgentConfig(
        layer_sizes=(300, 150, 3),
        activations=("Tanh", "Tanh", "Linear"),
        learning_rate=2.5e-4,
        replay_memory_size=10_000,
        discount_factor=0.45,
        epsilon_start=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.01,
        lookback=60,
        time_window=5,
        target_update_interval=1_000,
        batch_size=64,
        warmup_steps=2_000,
    ),
    "M15": AgentConfig(
        layer_sizes=(300, 200, 100, 3),
        activations=("Tanh", "Tanh", "Tanh", "Linear"),
        learning_rate=2.5e-4,
        replay_memory_size=40_000,
        discount_factor=0.5,
        epsilon_start=1.0,
        epsilon_decay=0.9999,
        epsilon_min=0.01,
        lookback=200,
        time_window=10,
        target_update_interval=1_000,
        batch_size=64,
        warmup_steps=5_000,
    ),
    "M5": AgentConfig(
        layer_sizes=(400, 200, 100, 3),
        activations=("Tanh", "Tanh", "Tanh", "Linear"),
        learning_rate=2.5e-4,
        replay_memory_size=120_000,
        discount_factor=0.6,
        epsilon_start=1.0,
        epsilon_decay=0.99995,
        epsilon_min=0.01,
        lookback=300,
        time_window=20,
        target_update_interval=1_000,
        batch_size=128,
        warmup_steps=10_000,
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


