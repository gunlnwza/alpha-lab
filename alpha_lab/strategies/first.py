from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta

from alpha_lab.backtest.account import Account
from alpha_lab.backtest.bot import BacktestBotTemplate, PrecomputedData
from alpha_lab.utils import ForexData

from alpha_lab.models.features import get_features  # TODO: smell

GATE_MA_SHORT_PERIOD = 50
GATE_MA_LONG_PERIOD = 200

ATR_PERIOD = 10
SL_VOL_MUL = 10


class LogisticRegressionBot(BacktestBotTemplate):
    def __init__(self):
        super().__init__(name="logistic_regression")

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        data = PrecomputedData(forex_data)
        
        # Signals
        X = get_features(forex_data.ohlcv)
        low_tf_clf = joblib.load(Path("alpha_lab", "models", "artifacts", "logreg_v1.pkl"))
        low_tf_raw = pd.Series(low_tf_clf.predict(X), index=X.index)
        low_tf = low_tf_raw.reindex(forex_data.ohlcv.index)

        ma_short = ta.ema(forex_data.ohlcv["close"], GATE_MA_SHORT_PERIOD)
        ma_long = ta.ema(forex_data.ohlcv["close"], GATE_MA_LONG_PERIOD)
        high_tf = (ma_short > ma_long)

        signals = (low_tf == 1) & (high_tf == True)
        assert signals.index.to_list() == forex_data.ohlcv.index.to_list()
        data.signals = signals.to_numpy()

        # Misc
        data.misc["vol"] = forex_data.ohlcv.ta.atr(ATR_PERIOD).to_numpy()  # Additional data, for SL

        return data
    
    def calculate_sl(self, close: float, vol: float):
        return close - SL_VOL_MUL * vol

    def act(self, idx: int, data: PrecomputedData, acc: Account):
        position = acc.get_position()
        close = data.prices.close[idx]
        vol = data.misc["vol"][idx]

        # ---
        if acc.have_position():
            new_sl = self.calculate_sl(close, vol)
            if new_sl > position.sl:
                position.set_sl(close, new_sl)
        else:
            if data.signals[idx] and not np.isnan(vol):
                sl = self.calculate_sl(close, vol)
                acc.open_position(idx, close, sl)
