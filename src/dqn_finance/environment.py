"""Market environment wrapper tailored for OHLCV-based DQN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

try:  # Optional dependency; the environment also works without pandas.
    import pandas as pd
except ImportError:  # pragma: no cover - handled gracefully when pandas absent
    pd = None  # type: ignore


ArrayLike = Union[np.ndarray, "pd.DataFrame"]


@dataclass
class EnvironmentStep:
    """Outcome of a single environment step."""

    next_state: np.ndarray
    reward: float
    done: bool


class MarketEnvironment:
    """Sliding-window environment that exposes OHLCV states and trading rewards.

    Parameters
    ----------
    data:
        Historical OHLCV samples ordered chronologically. Accepts a NumPy array
        of shape ``(T, F)`` or a pandas DataFrame with at least five columns
        corresponding to ``(open, high, low, close, volume)``.
    lookback:
        Number of past OHLCV samples constituting the state representation.
    stabilization_window:
        Time window ``t_w`` for action stabilization and reward computation.
    price_column:
        Column name (for DataFrame input) or positional index (for array input)
        pointing to the close price. Defaults to ``"close"`` for DataFrame
        inputs and index ``3`` for arrays.

    Notes
    -----
    The returned state is a flattened vector containing lookback consecutive
    OHLCV samples ordered from oldest to newest. Rewards follow the definition
    ``r_t = ((p_{t+t_w} - p_t) / p_t) * a_t``.
    """

    def __init__(
        self,
        data: ArrayLike,
        lookback: int,
        stabilization_window: int,
        *,
        price_column: Union[str, int] = "close",
        normalization_mean: Optional[np.ndarray] = None,
        normalization_std: Optional[np.ndarray] = None,
    ) -> None:
        if lookback <= 0:
            raise ValueError("Lookback period must be positive")
        if stabilization_window <= 0:
            raise ValueError("Stabilization window must be positive")

        self.lookback = lookback
        self.stabilization_window = stabilization_window

        self._data = self._to_numpy(data)

        if (normalization_mean is None) != (normalization_std is None):
            raise ValueError("Provide both normalization_mean and normalization_std or neither.")

        if normalization_mean is None:
            self._mean = np.mean(self._data, axis=0, dtype=np.float32)
            self._std = np.std(self._data, axis=0, dtype=np.float32)
        else:
            self._mean = np.asarray(normalization_mean, dtype=np.float32)
            self._std = np.asarray(normalization_std, dtype=np.float32)

        if self._std.shape != (self._data.shape[1],) or self._mean.shape != (self._data.shape[1],):
            raise ValueError("Normalization statistics must match the number of features in data.")

        self._std = np.where(self._std <= 0.0, 1e-9, self._std + 1e-9)  # Stabilize divisions

        if self._data.ndim != 2 or self._data.shape[0] <= self.lookback:
            raise ValueError("Input data must be a 2D array with more rows than the lookback period")

        self.num_features = int(self._data.shape[1])
        self.state_size = int(self.lookback * self.num_features)

        self._price_column = self._resolve_price_column(data, price_column)
        if not 0 <= self._price_column < self.num_features:
            raise ValueError("Resolved price column index is out of bounds for the provided data")
        self._prices = self._data[:, self._price_column]

        self._max_index = self._data.shape[0] - self.stabilization_window - 1
        if self._max_index < self.lookback - 1:
            raise ValueError("Dataset is too short for the requested lookback and stabilization window")

        self._terminal_state = np.zeros(self.state_size, dtype=np.float32)

        self._cursor = self.lookback - 1
        self._done = False

    @staticmethod
    def _to_numpy(data: ArrayLike) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return data.astype(np.float32, copy=False)

        if pd is not None and isinstance(data, pd.DataFrame):
            return data.to_numpy(dtype=np.float32, copy=True)

        raise TypeError("data must be a numpy.ndarray or a pandas.DataFrame")

    @staticmethod
    def _resolve_price_column(data: ArrayLike, price_column: Union[str, int]) -> int:
        if isinstance(data, np.ndarray):
            if isinstance(price_column, int):
                return price_column
            # Default to the fourth column (close price) when names are unavailable.
            return 3

        if pd is None or not isinstance(data, pd.DataFrame):  # pragma: no cover - defensive branch
            raise TypeError("Pandas DataFrame required to resolve price column by name")

        if isinstance(price_column, str):
            if price_column not in data.columns:
                raise ValueError(f"Column '{price_column}' not present in the DataFrame")
            return int(data.columns.get_loc(price_column))

        if isinstance(price_column, int):
            return price_column

        raise TypeError("price_column must be a string or integer")

    def reset(self) -> np.ndarray:
        """Reset the environment to its initial sliding window state."""

        self._cursor = self.lookback - 1
        self._done = False
        return self._build_state(self._cursor)

    def _build_state(self, index: int) -> np.ndarray:
        window = self._data[index - self.lookback + 1 : index + 1]
        # Apply normalization
        normalized_window = (window - self._mean) / self._std
        # Return the flattened normalized window
        return np.array(normalized_window.reshape(-1), dtype=np.float32, copy=True)

    def _compute_reward(self, index: int, action: float) -> float:
        current_price = float(self._prices[index])
        future_price = float(self._prices[index + self.stabilization_window])
        # Safeguard against any nan/inf prices
        if not np.isfinite(current_price) or not np.isfinite(future_price):
            return 0.0
        if current_price == 0:
            return 0.0
        return ((future_price - current_price) / current_price) * float(action)

    def step(self, action: float) -> EnvironmentStep:
        """Advance the environment by one time step."""

        if self._done:
            raise RuntimeError("Environment is done; call reset() before stepping again")

        reward = self._compute_reward(self._cursor, action)

        self._cursor += 1
        if self._cursor > self._max_index:
            self._done = True
            next_state = self._terminal_state.copy()
        else:
            next_state = self._build_state(self._cursor)

        return EnvironmentStep(next_state = next_state, reward = reward, done = self._done)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def state_shape(self) -> Sequence[int]:
        return (self.state_size,)

    @property
    def current_index(self) -> int:
        return self._cursor

    @property
    def max_index(self) -> int:
        return self._max_index

    @property
    def lookback_period(self) -> int:
        return self.lookback

    @property
    def action_window(self) -> int:
        return self.stabilization_window

    @property
    def normalization_mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def normalization_std(self) -> np.ndarray:
        return self._std.copy()


