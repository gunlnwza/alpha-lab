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
SL_VOL_MUL = 20


class PrecomputedData:
    def __init__(self, forex_data: ForexData):
        self.prices = forex_data
        self.signals = None
        self.misc = {}


class BacktestBot:
    def __init__(self):
        self.pred_full = None

    def _precompute_signals(self, forex_data: ForexData):
        # Two separate models
        # 1. Tactical (low level, low TF)
        X = get_features(forex_data.ohlcv)
        low_tf_clf = joblib.load(Path("alpha_lab", "models", "artifacts", "logreg_v1.pkl"))
        low_tf_raw = pd.Series(low_tf_clf.predict(X), index=X.index)
        low_tf = low_tf_raw.reindex(forex_data.ohlcv.index)

        # 2. Strategic (high level, high TF)
        ma_short = ta.ema(forex_data.ohlcv["close"], GATE_MA_SHORT_PERIOD)
        ma_long = ta.ema(forex_data.ohlcv["close"], GATE_MA_LONG_PERIOD)
        high_tf = (ma_short > ma_long)

        # Final combined signal, Only valid where low_tf exists AND gate is True
        signals = (low_tf == 1) & (high_tf == True)
        # signals = low_tf
        assert signals.index.to_list() == forex_data.ohlcv.index.to_list()
        return signals.to_numpy()

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        """
        Compute time-index aligned signals
        - Vectorized for most things, hopefully with no lookahead bias
        """
        data = PrecomputedData(forex_data)
        data.signals = self._precompute_signals(forex_data)  # Signals
        data.misc["vol"] = forex_data.ohlcv.ta.atr(ATR_PERIOD).to_numpy()  # Additional data, for SL
        return data

    def calculate_sl(self, close, vol):
        sl = close - SL_VOL_MUL * vol
        return sl

    def act(self, idx: int, data: PrecomputedData, acc: Account):
        limit = acc.get_limit()
        position = acc.get_position()
        close = data.prices.close[idx]
        vol = data.misc["vol"][idx]

        # if acc.have_position():
        #     new_sl = self.calculate_sl(close, vol)
        #     if new_sl > position.sl:
        #         position.set_sl(close, new_sl)
        # else:
        #     if data.signals[idx] and not np.isnan(vol):
        #         sl = self.calculate_sl(close, vol)
        #         acc.open_position(idx, close, sl)

        if not limit and not position:
            if not np.isnan(vol):
                acc.open_limit(idx, close - 0.1 * vol, close - 0.2 * vol)
                limit = acc.get_limit()
                print(limit)
