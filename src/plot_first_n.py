import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

CSV_PATH = "data/btc_hourly_with_sentiment.csv"
N1 = 43924   # start rows
N2 = 53335   # end rows


def plot_candlestick(ax, dates, open_, high_, low_, close_):
    """Draw candlesticks (no mpl-finance dependency)."""
    if len(dates) > 1:
        width = (dates[1] - dates[0]) * 0.65
    else:
        width = 0.65

    for x, o, h, l, c in zip(dates, open_, high_, low_, close_):
        if not np.isfinite([o, h, l, c]).all():
            continue

        color = "#089981" if c >= o else "#f23645"
        ax.plot([x, x], [l, h], color=color, linewidth=1)

        body = Rectangle(
            (x - width / 2, min(o, c)),
            width,
            max(abs(c - o), 1e-6),
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
        )
        ax.add_patch(body)

    ax.set_ylabel("Price")
    ax.grid(alpha=0.30)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))


def main():
    df = pd.read_csv(CSV_PATH)
    N = N2 - N1
    df = df.iloc[N1:N2].reset_index(drop=True)

    # Parse datetime column
    df["date_time"] = pd.to_datetime(df["date_time"], utc=False)

    open_ = pd.to_numeric(df["open"], errors="coerce")
    high_ = pd.to_numeric(df["high"], errors="coerce")
    low_ = pd.to_numeric(df["low"], errors="coerce")
    close_ = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    dates = mdates.date2num(df["date_time"].dt.to_pydatetime())

    fig, (ax_price, ax_volume) = plt.subplots(
        2, 1, sharex=True, figsize=(20, 10),
        gridspec_kw={"height_ratios": [3, 1]}
    )

    plot_candlestick(ax_price, dates, open_, high_, low_, close_)

    # Volume subplot
    colors = ["#089981" if c >= o else "#f23645" for o, c in zip(open_, close_)]
    ax_volume.bar(dates, volume, width=0.04, color=colors, alpha=0.4)
    ax_volume.set_ylabel("Volume")
    ax_volume.grid(alpha=0.30)

    ax_price.set_title(f"BTC Hourly OHLC — First {N} Candles")
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig("btc_hourly_first_n_candles.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
