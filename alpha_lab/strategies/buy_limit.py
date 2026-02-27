from alpha_lab.backtest.account import Account
from alpha_lab.backtest.bot import BacktestBotTemplate, PrecomputedData
from alpha_lab.utils import ForexData

import numpy as np
import pandas_ta as ta

# ---
# Config
LIMIT_TTL = 30

GATE_MA_SHORT_PERIOD = 50
GATE_MA_LONG_PERIOD = 200

ATR_PERIOD = 10
SL_VOL_MUL = 20
# ---


class BuyLimitBot(BacktestBotTemplate):
    def __init__(self):
        super().__init__("buy_limit")
        self.ttl = 0

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        data = PrecomputedData(forex_data)
        data.misc["vol"] = forex_data.ohlcv.ta.atr(ATR_PERIOD).to_numpy()  # Additional data, for SL
        data.misc["ma_short"] = ta.ema(forex_data.ohlcv["close"], GATE_MA_SHORT_PERIOD).to_numpy()
        data.misc["ma_long"] = ta.ema(forex_data.ohlcv["close"], GATE_MA_LONG_PERIOD).to_numpy()
        return data

    def calculate_sl(self, close: float, vol: float):
        return close - SL_VOL_MUL * vol

    def act(self, idx: int, data: PrecomputedData, acc: Account):
        close = data.prices.close[idx]
        ma_short = data.misc["ma_short"][idx]
        ma_long = data.misc["ma_long"][idx]
        vol = data.misc["vol"][idx]

        limit = acc.get_limit()
        position = acc.get_position()

        # ---
        uptrend = (not np.isnan(ma_long)) and ma_short > ma_long

        if limit:
            if self.ttl > 0:
                self.ttl -= 1
            if self.ttl == 0:
                acc.close_limit(idx)
        elif position:
            new_sl = self.calculate_sl(close, vol)
            if new_sl > position.sl:
                position.set_sl(close, new_sl)
        else:
            if uptrend and not np.isnan(vol):
                acc.open_limit(idx, close - 3 * vol, close - 6 * vol)
                self.ttl = LIMIT_TTL
