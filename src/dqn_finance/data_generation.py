"""Utilities for generating synthetic OHLCV data for testing DQN agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

try:  # Optional dependency: results can be returned as numpy arrays when pandas is unavailable.
    import pandas as pd
except ImportError:  # pragma: no cover - explicitly handled at runtime
    pd = None  # type: ignore


ArrayLike = np.ndarray
DataFrameLike = Union[ArrayLike, "pd.DataFrame"]


@dataclass(frozen=True)
class GBMGARCHParams:
    """Parameter bundle for GBM price dynamics with GARCH(1,1) volatility."""

    drift: float = 0.05
    omega: float = 1e-6
    alpha: float = 0.05
    beta: float = 0.9
    initial_volatility: float = 0.02


def _simulate_garch_volatility(
    n_steps: int,
    *,
    omega: float,
    alpha: float,
    beta: float,
    initial_volatility: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not 0 <= alpha < 1 or not 0 <= beta < 1 or alpha + beta >= 1:
        raise ValueError("GARCH parameters must satisfy 0 <= alpha, beta < 1 and alpha + beta < 1")
    if omega <= 0:
        raise ValueError("omega must be positive")
    if initial_volatility <= 0:
        raise ValueError("initial_volatility must be positive")

    variances = np.empty(n_steps, dtype=np.float64)
    variances[0] = initial_volatility ** 2
    shocks = np.zeros(n_steps, dtype=np.float64)

    for t in range(1, n_steps):
        shocks[t - 1] = np.sqrt(max(variances[t - 1], 1e-12)) * rng.standard_normal()
        variances[t] = omega + alpha * shocks[t - 1] ** 2 + beta * variances[t - 1]

    shocks[-1] = np.sqrt(max(variances[-1], 1e-12)) * rng.standard_normal()
    return np.sqrt(np.maximum(variances, 1e-12))


def _generate_price_path(
    start_price: float,
    drift: float,
    dt: float,
    volatilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if start_price <= 0:
        raise ValueError("start_price must be positive")

    n_steps = volatilities.shape[0]
    log_returns = drift * dt + volatilities * rng.standard_normal(n_steps)
    prices = np.empty(n_steps + 1, dtype=np.float64)
    prices[0] = start_price
    for t in range(1, n_steps + 1):
        prices[t] = prices[t - 1] * np.exp(log_returns[t - 1])
    return prices


def _synthesize_intraperiod_extremes(
    opens: np.ndarray,
    closes: np.ndarray,
    volatilities: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    span = np.clip(0.5 * volatilities, 1e-4, 0.2)
    high_noise = rng.uniform(0.1, 1.0, size=opens.size)
    low_noise = rng.uniform(0.1, 1.0, size=opens.size)

    highs = np.maximum(opens, closes) * (1 + span * high_noise)
    lows = np.minimum(opens, closes) * (1 - span * low_noise)
    lows = np.maximum(lows, 1e-3)
    return highs, lows


def _simulate_volume(volatilities: np.ndarray, base_volume: float, rng: np.random.Generator) -> np.ndarray:
    vol_scaled = volatilities / np.maximum(volatilities.mean(), 1e-6)
    lognormal_component = rng.lognormal(mean=0.0, sigma=0.3, size=volatilities.size)
    volume = base_volume * vol_scaled * lognormal_component
    return np.maximum(volume, 1.0)


def generate_mock_ohlcv(
    n_periods: int,
    *,
    start_price: float = 100.0,
    params: Optional[GBMGARCHParams] = None,
    dt: float = 1 / 252,
    freq: str = "1H",
    base_volume: float = 1e6,
    seed: Optional[int] = None,
    as_pandas: bool = True,
) -> DataFrameLike:
    """Generate synthetic OHLCV data using GBM with GARCH(1,1) volatility.

    Parameters
    ----------
    n_periods:
        Number of OHLCV rows to generate.
    start_price:
        Initial asset price, strictly positive.
    params:
        Optional :class:`GBMGARCHParams` instance. Defaults to sensible market parameters.
    dt:
        Time increment (in years) used for the drift component.
    freq:
        Frequency string used when returning a pandas DataFrame. Ignored when pandas is unavailable or
        ``as_pandas`` is ``False``.
    base_volume:
        Baseline traded volume level. Adjust to scale the magnitude of volume numbers.
    seed:
        Seed for NumPy's random number generator to obtain reproducible simulations.
    as_pandas:
        When ``True`` and pandas is installed, returns a DataFrame with OHLCV columns; otherwise returns
        a NumPy array of shape ``(n_periods, 5)``.
    """

    if n_periods <= 0:
        raise ValueError("n_periods must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if base_volume <= 0:
        raise ValueError("base_volume must be positive")

    rng = np.random.default_rng(seed)
    cfg = params or GBMGARCHParams()

    volatilities = _simulate_garch_volatility(
        n_periods,
        omega=cfg.omega,
        alpha=cfg.alpha,
        beta=cfg.beta,
        initial_volatility=cfg.initial_volatility,
        rng=rng,
    )

    prices = _generate_price_path(start_price, cfg.drift, dt, volatilities, rng)
    opens = prices[:-1]
    closes = prices[1:]
    highs, lows = _synthesize_intraperiod_extremes(opens, closes, volatilities, rng)
    volume = _simulate_volume(volatilities, base_volume, rng)

    ohlcv = np.column_stack(
        [
            opens.astype(np.float32),
            highs.astype(np.float32),
            lows.astype(np.float32),
            closes.astype(np.float32),
            volume.astype(np.float32),
        ]
    )

    if pd is not None and as_pandas:
        index = pd.date_range(start="2000-01-01", periods=n_periods, freq=freq)
        return pd.DataFrame(ohlcv, index=index, columns=["open", "high", "low", "close", "volume"])

    return ohlcv


__all__ = [
    "GBMGARCHParams",
    "generate_mock_ohlcv",
]


