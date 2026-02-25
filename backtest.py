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


class ForexData:
    def __init__(self, symbol: str, ohlc: pd.DataFrame):
        self.symbol = symbol.upper()
        self.ohlc = ohlc
    
    @classmethod
    def from_file(cls, source: str, symbol: str, tf: str):
        ohlc = load_parquet("twelve_data", symbol, "5min")
        ohlc = drop_weekend(ohlc)
        return ForexData(symbol, ohlc)
    

class BacktestResult:
    SL_VOL_MUL = 10

    def __init__(self, data: ForexData):
        self.symbol = data.symbol.upper()

        df = data.ohlc
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

    def report(self):
        print(f"{self.symbol} | Final PnL: {self.pnl:.4f}")

    def visualize(self):
        plt.figure(figsize=(14, 6))
        plt.plot(self.close, label="Close Price", linewidth=1)

        if self.entry_idx:  # Plot entries
            plt.scatter(self.entry_idx, self.close[np.array(self.entry_idx)],
                        marker="^", color="green", s=80, label="Entry")
        if self.exit_idx:  # Plot exits
            plt.scatter(self.exit_idx, self.close[np.array(self.exit_idx)],
                        marker="v", color="red", s=80, label="Exit")

        plt.title(f"Backtest Visualization {self.symbol.upper()}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def backtest(data: ForexData) -> BacktestResult:
    res = BacktestResult(data)

    clf = load_model("logreg_v1")
    df = data.ohlc

    X = get_features(df)
    pred = clf.predict(X)
    for i in range(len(pred)):
        if res.position == 0:
            if pred[i] == 1 and res.all_notna(i) and res.ma_short[i] > res.ma_long[i]:  # Enter long
                res.enter_long(i)
        elif res.position == 1:  # Manage open position with trailing stop
            res.update_trailing_stop(i)
            res.exit_if_hit_stop(i)

    if res.position == 1:  # Close any open position at final price
        pnl += res.close[-1] - res.entry_price

    return res


def main():
    for s in ["XAUUSD"]:
        data = ForexData.from_file("twelve_data", s, "5min")
        res = backtest(data)
        res.report()
        res.visualize()


if __name__ == "__main__":
    main()
