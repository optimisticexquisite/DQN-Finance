"""Render a candlestick chart from a CSV file containing OHLCV data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def _infer_timestamps(df: pd.DataFrame, start: str, freq: str) -> pd.Series:
    timestamp_columns = [col for col in df.columns if col.lower() in {"timestamp", "time", "date", "datetime"}]
    if timestamp_columns:
        ts_col = timestamp_columns[0]
        return pd.to_datetime(df[ts_col], utc=False, infer_datetime_format=True)

    return pd.date_range(start=start, periods=len(df), freq=freq, name="timestamp")


def _candlestick(ax: plt.Axes, dates: np.ndarray, opens: Iterable[float], highs: Iterable[float], lows: Iterable[float], closes: Iterable[float]) -> None:
    if len(dates) == 0:
        return

    # Determine candle body width proportional to the sampling interval.
    if len(dates) > 1:
        width = (dates[1] - dates[0]) * 0.6
    else:
        width = 0.6

    for x, o, h, l, c in zip(dates, opens, highs, lows, closes):
        if not np.isfinite([o, h, l, c]).all():
            continue
        color = "#089981" if c >= o else "#f23645"
        ax.plot([x, x], [l, h], color=color, linewidth=1)

        body_height = max(abs(c - o), 1e-6)
        lower = min(o, c)
        rect = Rectangle((x - width / 2, lower), width, body_height, facecolor=color, edgecolor=color, linewidth=0.5)
        ax.add_patch(rect)

    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))


def _plot_volume(ax: plt.Axes, dates: np.ndarray, volumes: Iterable[float], closes: Iterable[float], opens: Iterable[float]) -> None:
    if len(dates) == 0:
        return

    if len(dates) > 1:
        width = (dates[1] - dates[0]) * 0.6
    else:
        width = 0.6

    colors = ["#089981" if c >= o else "#f23645" for o, c in zip(opens, closes)]
    ax.bar(dates, volumes, width=width, color=colors, align="center", alpha=0.4)
    ax.set_ylabel("Volume")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)


def plot_candlestick(
    csv_path: Path,
    *,
    output: Path | None = None,
    start: str = "2000-01-01",
    freq: str = "15T",
    limit: int | None = 200,
) -> None:
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"CSV file must contain columns: {', '.join(sorted(REQUIRED_COLUMNS))}")

    # Normalize column names in case they differ by case.
    df = df.rename(columns=str.lower)

    timestamps = _infer_timestamps(df, start=start, freq=freq)
    df = df.assign(timestamp=timestamps)

    if limit is not None and limit > 0:
        df = df.tail(limit)

    timestamp_series = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamp_series.isna().any():
        raise ValueError("Timestamp column contains invalid values that cannot be parsed")
    dates = mdates.date2num(timestamp_series.dt.to_pydatetime())

    fig, (ax_price, ax_volume) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]})

    _candlestick(ax_price, dates, df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy())
    _plot_volume(ax_volume, dates, df["volume"].to_numpy(), df["close"].to_numpy(), df["open"].to_numpy())

    ax_price.set_title("Candlestick Chart")
    fig.autofmt_xdate()
    plt.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
    else:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, nargs="?", default=Path("data/mock_ohlcv.csv"), help="Path to the OHLCV CSV file")
    parser.add_argument("--output", type=Path, help="Optional path to save the chart instead of displaying it")
    parser.add_argument("--start", type=str, default="2000-01-01", help="Fallback start timestamp when CSV lacks datetime column")
    parser.add_argument("--freq", type=str, default="15T", help="Sampling frequency used to rebuild timestamps when absent")
    parser.add_argument("--limit", type=int, default=200, help="Number of most recent candles to plot (use <=0 for all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = args.limit if args.limit and args.limit > 0 else None
    plot_candlestick(
        args.csv_path,
        output=args.output,
        start=args.start,
        freq=args.freq,
        limit=limit,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


