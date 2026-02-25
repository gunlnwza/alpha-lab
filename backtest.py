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
    def __init__(self, data: ForexData):
        self.closed_orders = []
        self.order: BuyOrder | None = None
        self.pnl = 0.0

        df = data.ohlc
        self.close = df["close"].to_numpy()
        self.low = df["low"].to_numpy()

        self.vol = ta.atr(df["high"], df["low"], df["close"], length=14).to_numpy()
        self.ma_short = ta.linreg(df["close"], 20).to_numpy()
        self.ma_long = ta.kama(df["close"], 50).to_numpy()

    @classmethod
    def calculate_sl(cls, close, vol, sl_vol_mul=10):
        return close - sl_vol_mul * vol

    def open_order(self, i: int):
        sl = self.calculate_sl(self.close[i], self.vol[i])
        self.order = BuyOrder(i, self.close[i], sl)

    def close_order(self, idx: int, close: float):
        self.order.close(idx, close)
        self.pnl += self.order.pnl
        self.closed_orders.append(self.order)
        self.order = None


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
            plt.scatter(entry, close[np.array(entry)], marker="^", color="green", s=80, label="Entry")
        if exit:
            plt.scatter(exit, close[np.array(exit)], marker="v", color="red", s=80, label="Exit")

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
            ):
                acc.open_order(i)
        else:
            if acc.low[i] <= acc.order.sl: # check hit stop loss (use lowest price, not close, heavy wick, liquidity sweep can delete order)
                acc.close_order(i, acc.order.sl)
            if not acc.order:  # if order is closed, continue
                continue

            if not np.isnan(acc.vol[i]):  # adjust stop loss if still have order
                new_sl = acc.calculate_sl(acc.close[i], acc.vol[i])
                acc.order.sl = max(acc.order.sl, new_sl)

    if acc.order:
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
