from alpha_lab.backtest.account import Account, Side, OrderType
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData

import numpy as np
import pandas_ta as ta

LIMIT_TTL = 30

GATE_MA_SHORT_PERIOD = 50
GATE_MA_LONG_PERIOD = 200

ATR_PERIOD = 10
SL_VOL_MUL = 10


class BuyLimitBot(BacktestBot):
    def __init__(self):
        super().__init__("buy_limit")
        self.limit_ttl = 0

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        data = PrecomputedData(forex_data)

        data.vol = forex_data.ohlcv.ta.atr(ATR_PERIOD).to_numpy()
        data.ma_short = ta.ema(forex_data.ohlcv["close"], GATE_MA_SHORT_PERIOD).to_numpy()
        data.ma_long = ta.ema(forex_data.ohlcv["close"], GATE_MA_LONG_PERIOD).to_numpy()

        return data

    def act(self, data: PrecomputedData, acc: Account):
        now = data.now
        ma_short = data.ma_short[now]
        ma_long = data.ma_long[now]
        vol = data.vol[now]
        close = data.forex_data.close[now]

        order = acc.get_order()

        uptrend = (not np.isnan(ma_long)) and ma_short > ma_long

        if order:
            if order.type == OrderType.LIMIT:
                self.limit_ttl -= 1
                if self.limit_ttl == 0:
                    acc.close_order()
            else:
                new_sl = close - SL_VOL_MUL * vol
                if new_sl > order.sl:
                    acc.set_sl(new_sl)
        else:
            if uptrend and not np.isnan(vol):
                acc.open_limit(Side.BUY, close - 3 * vol, close - 6 * vol)
                self.limit_ttl = LIMIT_TTL
