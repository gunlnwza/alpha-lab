from pathlib import Path
import logging

import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib

from alpha_lab.models.features import get_features
from alpha_lab.backtest.account import Account
from alpha_lab.utils import ForexData

logger = logging.getLogger(__name__)

# TODO: different configs for different assets
GATE_MA_SHORT_PERIOD = 50
GATE_MA_LONG_PERIOD = 200
ATR_PERIOD = 10
SL_VOL_MUL = 10


class BacktestBot:
    def __init__(self):
        self.pred_full = None

    def precompute_data(self, forex_data: ForexData):
        """
        Compute time-index aligned signals
        - Vectorized for most things, hopefully with no lookahead bias
        """

        # self.n = len(self._ohlcv)
        # self._ohlcv = self.data.ohlcv
        # self.open = self._ohlcv["open"].to_numpy()
        # self.high = self._ohlcv["high"].to_numpy()
        # self.low = self._ohlcv["low"].to_numpy()
        # self.close = self._ohlcv["close"].to_numpy()

        data = forex_data
        ohlcv = data.ohlcv
    
        # Two separate models
        # 1. Tactical (low level, low TF)
        X = get_features(ohlcv)
        low_tf_clf = joblib.load(Path("alpha_lab", "models", "artifacts", "logreg_v1.pkl"))
        low_tf_raw = pd.Series(low_tf_clf.predict(X), index=X.index)
        low_tf = low_tf_raw.reindex(ohlcv.index)

        # 2. Strategic (high level, high TF)
        ma_short = ta.ema(ohlcv["close"], GATE_MA_SHORT_PERIOD)
        ma_long = ta.ema(ohlcv["close"], GATE_MA_LONG_PERIOD)
        high_tf = (ma_short > ma_long)

        # Final combined signal, Only valid where low_tf exists AND gate is True
        signals = (low_tf == 1) & (high_tf == True)
        assert signals.index.to_list() == ohlcv.index.to_list()

        self.vol = ohlcv.ta.atr(ATR_PERIOD).to_numpy()  # For SL
        self.signals = signals.to_numpy()

    @classmethod
    def calculate_sl(cls, close, vol):
        return close - SL_VOL_MUL * vol

    def act(self, idx: int, forex_data: ForexData, acc: Account):
        if acc.have_order():
            return
        
        sl = self.calculate_sl(forex_data.close[idx], self.vol[idx])
        acc.open_order(idx, forex_data.close[idx], sl)
