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


class Order:
    def __init__(self, close, sl):
        self.position = 1
        self.entry_price = close
        self.sl = sl

class Account:
    def __init__(self):
        self.entry_idx = []
        self.exit_idx = []
        self.pnl = 0.0
        self.order = None

    def enter_long(self, close: float, sl: float, enter_idx: int):
        if close <= sl:
            raise ValueError("Invalid stop loss")
        if self.order:
            raise ValueError("Already have order")

        self.order = Order(close, sl)
        self.entry_idx.append(enter_idx)

    def _v_have_order(self):
        if self.order is None:
            raise ValueError("No order at the moment")

    def close_order(self, close: float, exit_idx: int):
        self._v_have_order()
        self.pnl += close - self.order.entry_price
        self.order = None
        self.exit_idx.append(exit_idx)

    def sl_hit(self, exit_idx: int):
        self._v_have_order()
        self.close_order(self.order.sl, exit_idx)  # price will hit sl first before close is concluded


class BacktestResult:
    def __init__(self, symbol: str, ohlc: pd.DataFrame, acc: Account):
        self.symbol = symbol.upper()
        self.ohlc = ohlc
        self.acc = acc


def report(res: BacktestResult):
    print(f"{res.symbol} | Final PnL: {res.acc.pnl:.4f}")


def visualize(res: BacktestResult):
    plt.figure(figsize=(14, 6))
    close = res.ohlc.close.to_numpy()

    plt.plot(close, label="Close Price", linewidth=1)

    if res.acc.entry_idx:  # Plot entries
        plt.scatter(res.acc.entry_idx, close[np.array(res.acc.entry_idx)],
                    marker="^", color="green", s=80, label="Entry")
    if res.acc.exit_idx:  # Plot exits
        plt.scatter(res.acc.exit_idx, close[np.array(res.acc.exit_idx)],
                    marker="v", color="red", s=80, label="Exit")

    plt.title(f"Backtest Visualization {res.symbol.upper()}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


class SignalExpert:
    """Tactical logistic regression + Strategic short/long MAs gate"""
    
    def __init__(self, clf):
        self.clf = clf

    def predict(self, ohlc: pd.DataFrame) -> pd.Series:
        X = get_features(ohlc)

        # Raw model prediction aligned to feature index
        y_pred_raw = self.clf.predict(X)
        y_pred = pd.Series(y_pred_raw, index=X.index)

        # Strategic MA gate (full-length series)
        ma_short = ta.linreg(ohlc["close"], 20)
        ma_long = ta.kama(ohlc["close"], 50)
        ma_filter = ma_short > ma_long

        # Reindex prediction to full OHLC index, fill warmup zone with False
        y_pred = y_pred.reindex(ohlc.index, fill_value=False)

        # Combine tactical + strategic
        y_pred = y_pred & ma_filter.fillna(False)

        return y_pred


class RiskExpert:
    def __init__(self):
        self.sl_vol_mul = 10

    def compute_sl(self, close: float, vol: float):
        return close - self.sl_vol_mul * vol


def backtest_one(symbol: str):
    ohlc = load_data(symbol)
    close = ohlc.close.to_numpy()
    vol = ohlc.ta.atr(14).to_numpy()

    acc = Account()  # only do accounting stuff, no trade related activity
    sig_expert = SignalExpert(load_model("logreg_v1"))  # handle signal
    risk_expert = RiskExpert()  # handle SL, later lot sizing

    pred = sig_expert.predict(ohlc).to_numpy()  # finalized signals

    for i in range(len(pred)):
        if not acc.order:
            if pred[i] == 1:
                sl = risk_expert.compute_sl(close[i], vol[i])
                acc.enter_long(close[i], sl, i)
        else:
            if close[i] < acc.order.sl:
                acc.sl_hit(i)
                continue
            new_sl = risk_expert.compute_sl(close[i], vol[i])
            acc.order.sl = max(acc.order.sl, new_sl)

    if acc.order:
        acc.close_order(close[-1], len(close))

    res = BacktestResult(symbol, ohlc, acc)
    report(res)
    visualize(res)


def main():
    symbols = ["xauusd"]
    # symbols = ["xauusd", "usdcad", "audusd", "eurusd", "usdjpy"]

    for s in symbols:
        backtest_one(s)


if __name__ == "__main__":
    main()
