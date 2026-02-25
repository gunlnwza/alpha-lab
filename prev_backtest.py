import joblib
from pathlib import Path

import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt

from utils import load_parquet, drop_weekend
from features import get_features


def load_model(name: str):
    path = Path("models", f"{name}.pkl")
    clf = joblib.load(path)
    return clf


def load_data(symbol: str):
    df = load_parquet("twelve_data", symbol, "5min")
    df = drop_weekend(df)
    return df


class Record:
    SL_VOL_MUL = 10

    def __init__(self, symbol: str, df: pd.DataFrame):
        self.symbol = symbol.upper()
        self.close = df["close"].to_numpy()

        # account / orders
        self.entry_idx = []
        self.exit_idx = []
        self.position = 0
        self.entry_price = 0.0
        self.pnl = 0.0
        self.trailing_stop = 0.0

        # indicators
        self.vol = ta.atr(df["high"], df["low"], df["close"], length=14).to_numpy()
        self.ma_short = ta.linreg(df["close"], 20).to_numpy()
        self.ma_long = ta.kama(df["close"], 50).to_numpy()

    def all_notna(self, i: int):
        return not (np.isnan(self.vol[i]) or np.isnan(self.ma_short[i]) or np.isnan(self.ma_long[i]))

    def update_trailing_stop(self, i: int):
        if not np.isnan(self.vol[i]):
            new_stop = self.close[i] - self.SL_VOL_MUL * self.vol[i]
            self.trailing_stop = max(self.trailing_stop, new_stop)

    def exit_if_hit_stop(self, i: int):
        if self.close[i] <= self.trailing_stop:
            self.pnl += self.trailing_stop - self.entry_price
            self.position = 0
            self.exit_idx.append(i)

    def enter_long(self, i: int):
        self.position = 1
        self.entry_price = self.close[i]
        self.trailing_stop = self.close[i] - self.SL_VOL_MUL * self.vol[i]
        self.entry_idx.append(i)


def report(rec: Record):
    print(f"{rec.symbol} | Final PnL: {rec.pnl:.4f}")


def visualize(rec: Record):
    plt.figure(figsize=(14, 6))
    plt.plot(rec.close, label="Close Price", linewidth=1)

    if rec.entry_idx:  # Plot entries
        plt.scatter(rec.entry_idx, rec.close[np.array(rec.entry_idx)],
                    marker="^", color="green", s=80, label="Entry")
    if rec.exit_idx:  # Plot exits
        plt.scatter(rec.exit_idx, rec.close[np.array(rec.exit_idx)],
                    marker="v", color="red", s=80, label="Exit")

    plt.title(f"Backtest Visualization {rec.symbol.upper()}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def backtest_one(symbol: str):
    clf = load_model("logreg_v1")
    df = load_data(symbol)

    rec = Record(symbol, df)

    X = get_features(df)
    pred = clf.predict(X)
    for i in range(len(pred)):
        if rec.position == 0:
            if pred[i] == 1 and rec.all_notna(i) and rec.ma_short[i] > rec.ma_long[i]:  # Enter long
                rec.enter_long(i)
        elif rec.position == 1:  # Manage open position with trailing stop
            rec.update_trailing_stop(i)
            rec.exit_if_hit_stop(i)

    if rec.position == 1:  # Close any open position at final price
        pnl += rec.close[-1] - rec.entry_price

    report(rec)
    visualize(rec)


def main():
    symbols = ["xauusd"]
    # symbols = ["xauusd", "usdcad", "audusd", "eurusd", "usdjpy"]

    for s in symbols:
        backtest_one(s)


if __name__ == "__main__":
    main()
