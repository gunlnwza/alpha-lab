import joblib
from pathlib import Path

import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt

from utils import load_parquet, drop_weekend
from train import get_features_target

path = Path("models", "logreg_v1.pkl")
clf = joblib.load(path)

# --- Load your dataset (adjust path as needed)
def backtest_one(symbol):
    # data = load_parquet("twelve_data", symbol, "5min")
    # data = load_parquet("twelve_data", symbol, "1hour")
    try:
        data = load_parquet("twelve_data", symbol, "1day")
    except FileNotFoundError:
        return
    # data = 1 / data
    df = drop_weekend(data)

    # --- Feature preparation (must match training pipeline)
    X, _ = get_features_target(df)

    # --- Predict
    pred = clf.predict(X)

    close = df["close"].to_numpy()

    # atr = ta.atr(df["high"], df["low"], df["close"], length=14).to_numpy()
    std = ta.stdev(df["close"], 14).to_numpy()
    vol = std

    ema_short = ta.ema(df["close"], 100).to_numpy()
    ema_long = ta.ema(df["close"], 200).to_numpy()

    entry_idx = []
    exit_idx = []

    position = 0
    entry_price = 0.0
    pnl = 0.0
    trailing_stop = 0.0

    SL_VOL_MUL = 10
    for i in range(len(pred)):

        # Enter long
        indicators = not (np.isnan(vol[i]) or np.isnan(ema_short[i]) or np.isnan(ema_long[i]))
        if position == 0 and pred[i] == 1 and indicators and ema_short[i] > ema_long[i]:
            position = 1
            entry_price = close[i]
            trailing_stop = close[i] - SL_VOL_MUL * vol[i]  # 1 ATR initial stop
            entry_idx.append(i)

        # Manage open position with trailing stop
        elif position == 1:

            # Update trailing stop (only move upward)
            if not np.isnan(vol[i]):
                new_stop = close[i] - SL_VOL_MUL * vol[i]
                trailing_stop = max(trailing_stop, new_stop)

            # Exit if price hits trailing stop
            if close[i] <= trailing_stop:
                pnl += close[i] - entry_price
                position = 0
                exit_idx.append(i)

    # Close any open position at final price
    if position == 1:
        pnl += close[-1] - entry_price

    print(f"{s.upper()} | Final PnL: {pnl:.4f}")

    # --- Visualization
    plt.figure(figsize=(14, 6))

    plt.plot(close, label="Close Price", linewidth=1)
    plt.plot(ema_short, linewidth=1)
    plt.plot(ema_long, linewidth=1)
    # plt.plot(close - SL_VOL_MUL * vol, label="SL", linewidth=1)

    # Plot entries
    if entry_idx:
        plt.scatter(entry_idx, close[np.array(entry_idx)],
                    marker="^", color="green", s=80, label="Entry")

    # Plot exits
    if exit_idx:
        plt.scatter(exit_idx, close[np.array(exit_idx)],
                    marker="v", color="red", s=80, label="Exit")

    plt.title(f"Backtest Visualization {s.upper()}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


for s in ["xauusd", "usdcad", "audusd", "eurusd", "usdjpy"]:
    backtest_one(s)