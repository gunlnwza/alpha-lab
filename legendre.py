import numpy as np
import pandas as pd
from numpy.polynomial import legendre as L

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from alpha_lab.utils import ForexData, split_timeseries


def get_windows(
    df: pd.DataFrame,
    cols: list[str],
    size: int = 50,
) -> pd.DataFrame:
    """
    Get windows for df
    Sort the columns
    shift_{size-1} ... shift_{0}
    """
    windows = []
    for col in cols:
        series = []
        for i in range(size-1, -1, -1):
            series.append(
                df[col]
                .shift(i)
                .rename(f"{col}_shift_{i}")
            )
        windows.append(pd.concat(series, axis=1))

    windows = pd.concat(windows, axis=1)
    return windows


def get_normalized_windows(
    df: pd.DataFrame,
    cols: list[str],
    size: int = 50,
) -> pd.DataFrame:
    windows = get_windows(df, cols, size)

    mean = df["close"].rolling(size).mean()
    std = df["close"].rolling(size).std()

    normalized_windows = (
        windows
        .sub(mean, axis=0)
        .divide(std, axis=0)
    )
    return normalized_windows


def get_legendre_coefs(
    windows: pd.DataFrame,
    size: int = 50,
    deg: int = 3
):
    x = np.linspace(-1, 1, size)
    X = windows.to_numpy()
    coefs = L.legfit(x, X.T, deg).T

    prefix = windows.columns[0].split("_")[0]
    cols = [f"{prefix}_coef_{i}" for i in range(deg + 1)]
    coefs = pd.DataFrame(
        coefs,
        columns=cols
    )
    return coefs


# -------------------------------
# Configuration
# -------------------------------
SIZE = 50          # window length for time‑series segments
DEG = 5            # degree of Legendre polynomial

# starting index (must be >= SIZE because rolling windows need history)
i = SIZE


# -------------------------------
# Load and prepare data
# -------------------------------
data = ForexData("twelve_data", "xauusd", "5min")
chunks = split_timeseries(data.ohlcv)
df = chunks[0]

# rolling statistics used for normalization and denormalization
mean = df["close"].rolling(SIZE).mean()
std = df["close"].rolling(SIZE).std()

# construct rolling windows and normalized versions
windows = get_windows(df, ["close"], SIZE)
normalized_windows = get_normalized_windows(df, ["close"], SIZE)

# compute Legendre coefficients for each normalized window
coefs = get_legendre_coefs(normalized_windows, SIZE, DEG)

# x coordinates for polynomial evaluation
x = np.linspace(-1, 1, SIZE)


# -------------------------------
# Initial polynomial reconstruction
# -------------------------------
y_fit = L.legval(x, coefs.iloc[i])


# -------------------------------
# Create figure layout
# -------------------------------
# 3 panels:
#   1. raw price window + reconstructed fit
#   2. normalized window + polynomial fit
#   3. Legendre coefficients
fig, axes = plt.subplots(3, 1, figsize=(10, 8), height_ratios=[2, 1, 1])
fig.suptitle(f"index = {i}")


# -------------------------------
# Panel 1 — raw price window
# -------------------------------
line_raw, = axes[0].plot(windows.iloc[i])
line_raw_fit, = axes[0].plot(y_fit * std.iloc[i] + mean.iloc[i])
axes[0].set_xticks([])


# -------------------------------
# Panel 2 — normalized window
# -------------------------------
line_normal, = axes[1].plot(normalized_windows.iloc[i])
line_fit, = axes[1].plot(y_fit)

axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_xticks([])
axes[1].set_ylim(-3, 3)


# -------------------------------
# Panel 3 — Legendre coefficients
# -------------------------------
bars = axes[2].bar(x=list(range(DEG + 1)), height=coefs.iloc[i])

axes[2].axhline(0, color="black", linewidth=0.5)
axes[2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
axes[2].set_ylim(-3, 3)


# -------------------------------
# Plot update logic
# -------------------------------
def update():
    global i

    # recompute polynomial from stored coefficients
    y_fit = L.legval(x, coefs.iloc[i])

    # update reconstructed signals
    line_fit.set_ydata(y_fit)
    line_raw_fit.set_ydata(y_fit * std.iloc[i] + mean.iloc[i])

    # update raw window
    line_raw.set_ydata(windows.iloc[i].values)
    axes[0].relim()
    axes[0].autoscale_view()

    # update normalized window
    line_normal.set_ydata(normalized_windows.iloc[i].values)
    axes[1].relim()
    axes[1].autoscale_view()

    # update coefficient bars
    for bar, h in zip(bars, coefs.iloc[i]):
        bar.set_height(h)

    fig.suptitle(f"index = {i}")
    fig.canvas.draw_idle()


# -------------------------------
# Keyboard navigation
# -------------------------------
def on_key(event):
    global i

    if event.key == "left":
        i = max(SIZE, i - 1)
        update()

    elif event.key == "right":
        i = min(len(normalized_windows) - 1, i + 1)
        update()


# attach keyboard listener
fig.canvas.mpl_connect("key_press_event", on_key)

# start interactive viewer
plt.show()
