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
        ohlc = load_parquet(source, symbol, tf)
        ohlc = drop_weekend(ohlc)
        return ForexData(symbol, ohlc)


class BuyOrder:
    def __init__(self, idx: int, close: float, sl: float):
        self.entry_idx = idx
        self.entry_price = close

        self.exit_idx = None
        self.exit_price = None

        self.sl = sl

        self.pnl = None

    def close(self, idx: int, close: float):
        self.pnl = close - self.entry_price
        self.exit_idx = idx
        self.exit_price = close

    def sl_hit(self, idx: int):
        self.close(idx, self.sl)

class Account:
    SL_VOL_MUL = 10

    def __init__(self, data: ForexData):
        self.closed_orders = []
        self.order: BuyOrder | None = None
        self.pnl = 0.0

        df = data.ohlc
        self.close = df["close"].to_numpy()
        self.vol = ta.atr(df["high"], df["low"], df["close"], length=14).to_numpy()
        self.ma_short = ta.linreg(df["close"], 20).to_numpy()
        self.ma_long = ta.kama(df["close"], 50).to_numpy()

    def enter_long(self, i: int):
        sl = self.close[i] - self.SL_VOL_MUL * self.vol[i]
        self.order = BuyOrder(i, self.close[i], sl)

    def close_order(self, idx: int, close: float):
        self.order.close(idx, close)
        self.pnl += self.order.pnl
        self.closed_orders.append(self.order)
        self.order = None

    def update_trailing_stop(self, i: int):
        if not np.isnan(self.vol[i]):
            new_stop = self.close[i] - self.SL_VOL_MUL * self.vol[i]
            self.order.sl = max(self.order.sl, new_stop)

    def exit_if_hit_stop(self, idx: int):
        if self.close[idx] <= self.order.sl:
            self.close_order(idx, self.order.sl)


class BacktestResult:
    def __init__(self, data: ForexData, acc: Account):
        self.symbol = data.symbol.upper()
        self.acc = acc

    def report(self):
        print(f"{self.symbol} | Final PnL: {self.acc.pnl:.4f}")

    def visualize(self):
        close = self.acc.close
        entry = [o.entry_idx for o in self.acc.closed_orders]
        exit = [o.exit_idx for o in self.acc.closed_orders]

        plt.figure(figsize=(14, 6))
        plt.plot(close, label="Close Price", linewidth=1)
        if entry:
            plt.scatter(entry, close[np.array(entry)],
                        marker="^", color="green", s=80, label="Entry")
        if exit:
            plt.scatter(exit, close[np.array(exit)],
                        marker="v", color="red", s=80, label="Exit")

        plt.title(f"Backtest Visualization {self.symbol.upper()}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def backtest(data: ForexData) -> BacktestResult:
    acc = Account(data)

    clf = load_model("logreg_v1")
    X = get_features(data.ohlc)
    pred = clf.predict(X)

    for i in range(len(pred)):
        if not acc.order:
            if (pred[i] == 1
                and not np.isnan(acc.vol[i])
                and not np.isnan(acc.ma_short[i])
                and not np.isnan(acc.ma_long[i])
                and acc.ma_short[i] > acc.ma_long[i]
            ):  # Enter long
                acc.enter_long(i)
        else:  # Manage open position with trailing stop
            acc.update_trailing_stop(i)
            acc.exit_if_hit_stop(i)

    if acc.order:  # Close any open position at final price
        acc.close_order(len(acc.close) - 1, acc.close[-1])

    res = BacktestResult(data, acc)
    return res


def main():
    for s in ["XAUUSD"]:
        data = ForexData.from_file("twelve_data", s, "5min")
        res = backtest(data)
        res.report()
        res.visualize()


if __name__ == "__main__":
    main()
