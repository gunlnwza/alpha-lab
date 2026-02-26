import joblib
from pathlib import Path

import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt

from utils import load_parquet, drop_weekend, inverse_ohlcv
from models.features import get_features

GATE_MA_SHORT_PERIOD = 50
GATE_MA_LONG_PERIOD = 200

ATR_PERIOD = 10
SL_VOL_MUL = 10

# TODO: different configs for different assets

class ForexData:
    def __init__(self, source: str, symbol: str, tf: str):
        self.source = source
        self.symbol = symbol.upper()
        self.tf = tf

        self.ohlcv = load_parquet(source, symbol, tf)
        self.ohlcv = drop_weekend(self.ohlcv)

        self.inv_ohlcv = inverse_ohlcv(self.ohlcv)

    def __str__(self):
        return f"{self.source}'s {self.symbol} ({self.tf})"

    def __repr__(self):
        return f"ForexData({self.source}, {self.symbol}, {self.tf})"


class SimulationData:
    def __init__(self, forex_data: ForexData):
        self.forex_data = forex_data
        self._ohlcv = forex_data.ohlcv

        self.open = self._ohlcv["open"].to_numpy()
        self.high = self._ohlcv["high"].to_numpy()
        self.low = self._ohlcv["low"].to_numpy()
        self.close = self._ohlcv["close"].to_numpy()

        self.vol = self._ohlcv.ta.atr(ATR_PERIOD).to_numpy()
        self.ma_short = ta.ema(self._ohlcv["close"], GATE_MA_SHORT_PERIOD).to_numpy()
        self.ma_long = ta.ema(self._ohlcv["close"], GATE_MA_LONG_PERIOD).to_numpy()


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


def calculate_sl(close, vol):
    return close - SL_VOL_MUL * vol


class Account:
    def __init__(self):
        self.closed_orders = []
        self.order: BuyOrder | None = None
        self.pnl = 0.0

    def open_order(self, idx: int, close: float, vol: float):
        sl = calculate_sl(close, vol)
        self.order = BuyOrder(idx, close, sl)

    def close_order(self, idx: int, close: float):
        self.order.close(idx, close)
        self.pnl += self.order.pnl
        self.closed_orders.append(self.order)
        self.order = None


class BacktestResult:
    def __init__(self, data: SimulationData, acc: Account):
        self.data = data
        self.acc = acc

    def report(self):
        win = sum(1 for o in self.acc.closed_orders if o.pnl > 0)
        loss = sum(1 for o in self.acc.closed_orders if o.pnl < 0)
        trades = win + loss
        pos_pnl = sum(o.pnl for o in self.acc.closed_orders if o.pnl > 0)
        neg_pnl = sum(-o.pnl for o in self.acc.closed_orders if o.pnl < 0)

        win_rate = win / trades if trades > 0 else "inf"
        profit_per_loss = pos_pnl / neg_pnl if neg_pnl != 0 else 'inf'

        print("-" * 40)
        print(f"{self.data.forex_data}")
        print(f"    Win | Loss | Trades : {win:.0f} | {loss:.0f} | {trades:.0f}")
        print(f"               Win Rate : {win_rate:.2f}")
        print(f"+PnL | -PnL | Total PnL : {pos_pnl:.2f} | {-neg_pnl:.2f} | {pos_pnl - neg_pnl:.2f}")
        print(f"        Profit per Loss : {profit_per_loss:.2f}")
        print("-" * 40)

    def visualize(self):
        plt.figure(figsize=(14, 6))

        plt.plot(self.data.close, label="Close Price", linewidth=1)
        plt.plot(self.data.ma_short, label="Gate MA Short", linewidth=1)
        plt.plot(self.data.ma_long, label="Gate MA Long", linewidth=1)

        for order in self.acc.closed_orders:
            c = "green" if order.pnl >= 0 else "red"
            plt.plot([order.entry_idx, order.exit_idx], [order.entry_price, order.exit_price], color=c, lw=5)

        plt.title(f"Backtest Visualization | {self.data.forex_data}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def get_signals(data: SimulationData):
    # Two separate models
    # 1. Tactical (low level, low TF)
    X = get_features(data._ohlcv)
    low_tf_clf = joblib.load("models/logreg_v1.pkl")
    low_tf_raw = pd.Series(low_tf_clf.predict(X), index=X.index)

    # Align tactical signal to full OHLC index
    low_tf = low_tf_raw.reindex(data._ohlcv.index)

    # 2. Strategic (high level, high TF)
    ma_short = pd.Series(data.ma_short, index=data._ohlcv.index)
    ma_long = pd.Series(data.ma_long, index=data._ohlcv.index)
    high_tf = (ma_short > ma_long)

    # Final combined signal
    # Only valid where low_tf exists AND gate is True
    pred_full = (low_tf == 1) & (high_tf == True)

    return pred_full


def backtest(forex_data: ForexData) -> BacktestResult:
    acc = Account()
    data = SimulationData(forex_data)

    pred_full = get_signals(data)
    assert len(pred_full) == len(data._ohlcv)
    assert pred_full.index.to_list() == data._ohlcv.index.to_list()

    for i in range(len(data._ohlcv)):
        pred = pred_full.iloc[i]
        # if no order, wait for signal
        if not acc.order:
            if not np.isnan(pred) and pred == 1 and not np.isnan(data.vol[i]):
                acc.open_order(i, data.close[i], data.vol[i])
                # print(i, "open at", data.close[i], data._ohlcv.iloc[i].to_list())
            else:
                # print(i, "warm up", data._ohlcv.iloc[i].to_list())
                pass
        else:
            # check if stop loss
            if acc.order and data.low[i] < acc.order.sl:  # if low hooked sl, close
                # print(i, "close at", acc.order.sl, data._ohlcv.iloc[i].to_list())
                acc.close_order(i, acc.order.sl)

            # adjust stop loss if still have order
            if acc.order and not np.isnan(data.vol[i]):
                new_sl = calculate_sl(data.high[i], data.vol[i])  # update with high
                acc.order.sl = max(acc.order.sl, new_sl)
                # print(i, "adjust sl to", acc.order.sl, data._ohlcv.iloc[i].to_list())

    if acc.order:
        acc.close_order(len(data.close) - 1, data.close[-1])

    res = BacktestResult(data, acc)
    return res


def main():
    # symbols = ["XAUUSD", "USDJPY", "EURUSD", "AUDUSD"]
    symbols = ["XAUUSD"]
    # symbols = ["USDJPY"]
    # symbols = ["USDCAD"]
    # symbols = ["AUDUSD"]
    for s in symbols:
        for tf in ["1min", "5min", "15min", "1hour", "1day"]:
            data = ForexData("twelve_data", s, tf)
            res = backtest(data)
            res.report()
            res.visualize()


if __name__ == "__main__":
    main()
