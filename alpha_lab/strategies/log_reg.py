from alpha_lab.backtest.account import Account, Side, OrderType
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData, load_model

import numpy as np
import pandas as pd
import pandas_ta as ta

GATE_MA_SHORT_PERIOD = 50
GATE_MA_LONG_PERIOD = 200

ATR_PERIOD = 10
SL_VOL_MUL = 10


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return raw `X`
    - `df` is a copy of OHLCV dataframe
    """

    # indicator
    df["smooth(price)"] = ta.linreg(df["close"], 20)
    df["short"] = ta.ema(df["close"], 20)
    df["long"] = ta.ema(df["close"], 50)
    df["rsi"] = ta.rsi(df["close"], 20)

    # numeric features
    df["smooth(price)/short"] = df["smooth(price)"] / df["short"]  # too much noise, need smoothing
    df["smooth(smooth(price)/short)"] = ta.linreg(df["smooth(price)/short"], 5)

    df["smooth(price)/long"] = df["smooth(price)"] / df["long"]  # too much noise, need smoothing
    df["smooth(smooth(price)/long)"] = ta.linreg(df["smooth(price)/long"], 5)

    df["short/long"] = df["short"] / df["long"]

    # boolean features
    df["smooth(price)_above_short"] = (df["smooth(price)/short"] > 1).astype("int8")
    df["smooth(price)_above_long"]  = (df["smooth(price)/long"] > 1).astype("int8")
    df["short_above_long"]  = (df["short"] > df["long"]).astype("int8")
    df["rsi_overbought"] = (df["rsi"] > 70).astype("int8")
    df["rsi_oversold"] = (df["rsi"] < 30).astype("int8")

    return df[[
        # numeric
        "smooth(smooth(price)/short)",
        "smooth(smooth(price)/long)",
        "short/long",
        "rsi",

        # boolean
        "smooth(price)_above_short",
        "smooth(price)_above_long",
        "short_above_long",
        "rsi_overbought",
        "rsi_oversold"
    ]]


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = _compute_features(df).dropna()
    return X


class LogRegBot(BacktestBot):
    def __init__(self):
        super().__init__("logistic_regression")

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        data = PrecomputedData(forex_data)
        
        data.vol = forex_data.ohlcv.ta.atr(ATR_PERIOD).to_numpy()

        X = get_features(forex_data.ohlcv)
        clf = load_model("logreg_v1")
        low_tf_raw = pd.Series(clf.predict(X), index=X.index)
        low_tf = low_tf_raw.reindex(forex_data.ohlcv.index)

        ma_short = ta.ema(forex_data.ohlcv["close"], GATE_MA_SHORT_PERIOD)
        ma_long = ta.ema(forex_data.ohlcv["close"], GATE_MA_LONG_PERIOD)
        high_tf = (ma_short > ma_long)

        signals = (low_tf == 1) & (high_tf == True)
        assert signals.index.to_list() == forex_data.ohlcv.index.to_list()
        data.signals = signals.to_numpy()

        return data

    def act(self, data: PrecomputedData, acc: Account):
        bar = data.bar
        now = data.now

        order = acc.get_order()
        signal = data.signals[now]
        vol = data.vol[now]

        if order:
            new_sl = bar.close - SL_VOL_MUL * vol
            if new_sl > order.sl:
                order.set_sl(new_sl, bar)
        else:
            if signal and not np.isnan(vol):
                sl = bar.close - SL_VOL_MUL * vol
                acc.open_position(Side.BUY, bar, sl)
