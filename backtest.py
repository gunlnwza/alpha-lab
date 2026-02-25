import joblib
from pathlib import Path

import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt

from utils import load_parquet, drop_weekend
from features import get_features

GATE_MA_SHORT_PERIOD = 20
GATE_MA_LONG_PERIOD = 50

ATR_PERIOD = 20
SL_VOL_MUL = 12


class ForexData:
    def __init__(self, symbol: str, ohlc: pd.DataFrame):
        self.symbol = symbol.upper()
        self.ohlc = ohlc
    
    @classmethod
    def from_file(cls, source: str, symbol: str, tf: str):
        ohlc = load_parquet(source, symbol, tf)
        ohlc = drop_weekend(ohlc)
        return ForexData(symbol, ohlc)


class SimulationData:
    def __init__(self, forex_data: ForexData):
        self.forex_data = forex_data
        self._ohlc = forex_data.ohlc

        self.open = self._ohlc["open"].to_numpy()
        self.high = self._ohlc["high"].to_numpy()
        self.low = self._ohlc["low"].to_numpy()
        self.close = self._ohlc["close"].to_numpy()

        self.vol = self._ohlc.ta.atr(ATR_PERIOD).to_numpy()
        self.ma_short = ta.linreg(self._ohlc["close"], GATE_MA_SHORT_PERIOD).to_numpy()
        self.ma_long = ta.kama(self._ohlc["close"], GATE_MA_LONG_PERIOD).to_numpy()


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
    def __init__(self):
        self.closed_orders = []
        self.order: BuyOrder | None = None
        self.pnl = 0.0

    @classmethod
    def calculate_sl(cls, close, vol):
        return close - SL_VOL_MUL * vol

    def open_order(self, idx: int, close: float, vol: float):
        sl = self.calculate_sl(close, vol)
        self.order = BuyOrder(idx, close, sl)

    def close_order(self, idx: int, close: float):
        self.order.close(idx, close)
        self.pnl += self.order.pnl
        self.closed_orders.append(self.order)
        self.order = None


class BacktestResult:
    def __init__(self, data: SimulationData, acc: Account):
        self.data = data
        self.symbol = self.data.forex_data.symbol.upper()

        self.acc = acc

    def report(self):
        print(f"{self.symbol} | Final PnL: {self.acc.pnl:.4f}")

    def visualize(self):
        close = self.data.close
        entry = [o.entry_idx for o in self.acc.closed_orders]
        exit = [o.exit_idx for o in self.acc.closed_orders]

        plt.figure(figsize=(14, 6))

        plt.plot(close, label="Close Price", linewidth=1)
        if entry:
            plt.scatter(entry, close[np.array(entry)], marker="^", color="green", s=80, label="Entry")
        if exit:
            plt.scatter(exit, close[np.array(exit)], marker="v", color="red", s=80, label="Exit")

        plt.title(f"Backtest Visualization {self.symbol}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def backtest(forex_data: ForexData) -> BacktestResult:
    acc = Account()
    data = SimulationData(forex_data)
    clf = joblib.load("models/logreg_v1.pkl")

    X = get_features(data.forex_data.ohlc)
    pred = clf.predict(X)
    pred = pd.Series(pred, index=X.index)
    data._ohlc["pred"] = pred

    for i in range(len(forex_data.ohlc)):
        pred = data._ohlc["pred"].iloc[i]
        if np.isnan(pred):
            continue
        # if no order, wait for signal
        if not acc.order:
            if (pred == 1
                and not np.isnan(data.vol[i])
                and not np.isnan(data.ma_short[i])
                and not np.isnan(data.ma_long[i])
                and data.ma_short[i] > data.ma_long[i]  
            ):
                acc.open_order(i, data.close[i], data.vol[i])
        else:
            # check if stop loss
            if data.low[i] <= acc.order.sl:
                acc.close_order(i, acc.order.sl)
            if not acc.order:
                continue

            # adjust stop loss if still have order
            if not np.isnan(data.vol[i]):
                new_sl = acc.calculate_sl(data.high[i], data.vol[i])
                acc.order.sl = max(acc.order.sl, new_sl)

    if acc.order:
        acc.close_order(len(data.close) - 1, data.close[-1])

    res = BacktestResult(data, acc)
    return res


def main():
    symbols = ["XAUUSD"]
    # symbols = ["XAUUSD", "USDJPY", "EURUSD"]
    for s in symbols:
        data = ForexData.from_file("twelve_data", s, "5min")
        res = backtest(data)
        res.report()
        res.visualize()


if __name__ == "__main__":
    main()
