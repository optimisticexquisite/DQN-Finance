"""Experience replay memory utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence

import numpy as np


@dataclass
class Transition:
    """Container for a single environment interaction."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayMemory:
    """Fixed-size buffer that stores experience tuples for random sampling."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay memory capacity must be positive")
        self._capacity: int = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a completed transition in the replay memory."""

        self._buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int, *, rng: Optional[np.random.Generator] = None) -> Sequence[Transition]:
        """Sample a random minibatch of transitions."""

        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if len(self._buffer) < batch_size:
            raise ValueError("Not enough elements in replay memory to sample the requested batch")

        if rng is None:
            indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        else:
            indices = rng.choice(len(self._buffer), size=batch_size, replace=False)

        return [self._buffer[idx] for idx in indices]

    def is_ready(self, batch_size: int) -> bool:
        """Return ``True`` if the buffer contains enough samples for a minibatch."""

        return len(self._buffer) >= batch_size


