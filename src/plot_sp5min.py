"""Utility script to visualize the SP_5min.csv dataset."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "SP_5min.csv"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def _timestamp_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _annotate_plot(ax: Axes, *, timestamp: str | None = None) -> None:
    label = timestamp or _timestamp_label()
    ax.text(
        0.995,
        0.02,
        f"Generated {label}",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
        ha="right",
        va="bottom",
        alpha=0.85,
    )


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and sort the SP 5-minute OHLCV dataset."""

    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],
        dtype={
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "volume": "float32",
        },
    )
    expected = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df.sort_values("timestamp").reset_index(drop=True)


def slice_fraction(df: pd.DataFrame, fraction_range: Tuple[float, float]) -> pd.DataFrame:
    """Return a slice corresponding to the given fraction range [start, end)."""

    start_frac, end_frac = fraction_range
    if not (0.0 <= start_frac < end_frac <= 1.0):
        raise ValueError("Fraction range must satisfy 0.0 <= start < end <= 1.0")
    start_idx = int(len(df) * start_frac)
    end_idx = int(len(df) * end_frac)
    return df.iloc[start_idx:end_idx].reset_index(drop=True)


def plot_close_price(df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Create a simple line plot showing the close price trajectory."""

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["timestamp"], df["close"], linewidth=1.0, color="#1f77b4", label="Close")
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Close Price")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    _annotate_plot(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    dataset = load_dataset(DATA_PATH)
    full_plot_path = PLOTS_DIR / "sp_5min_full.png"
    zoom_plot_path = PLOTS_DIR / "sp_5min_frac_0.70_0.85.png"
    zoom_plot_path_1 = PLOTS_DIR / "sp_5min_frac_0.00_0.70.png"
    plot_close_price(dataset, "SP 5-Min Close Price (Full Dataset)", full_plot_path)

    subset = slice_fraction(dataset, (0.70, 0.85))
    subset_1 = slice_fraction(dataset, (0.85, 1))
    plot_close_price(
        subset,
        "SP 5-Min Close Price (70% - 85% of Timeline)",
        zoom_plot_path,
    )
    plot_close_price(
        subset_1,
        "SP 5-Min Close Price (85% - 100% of Timeline)",
        zoom_plot_path_1,
    )
    print(f"Saved full-dataset plot to {full_plot_path}")
    print(f"Saved fractional slice plot to {zoom_plot_path}")
    print(f"Saved fractional slice plot to {zoom_plot_path_1}")


if __name__ == "__main__":
    main()

