from alpha_lab.backtest.account import Account, Side, OrderType
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData

import numpy as np
import pandas_ta as ta


class StBot(BacktestBot):
    def __init__(self):
        super().__init__("songsak_techachan")

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        data = PrecomputedData(forex_data)

        # ===== 15m indicators =====
        ema9 = ta.ema(forex_data.ohlcv.close, 9)
        ema3_of_ema9 = ta.ema(ema9, 3)

        data.ma_short_15m = ema9.to_numpy()
        data.ma_long_15m = ema3_of_ema9.to_numpy()

        # ===== resample to 1H =====
        ohlcv_1h = forex_data.ohlcv.resample("h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        # ===== 1H indicators =====
        ema9_1h = ta.ema(ohlcv_1h.close, 9)
        ema3_of_ema9_1h = ta.ema(ema9_1h, 3)

        # align back to 15m timeline
        ema9_1h = ema9_1h.reindex(forex_data.ohlcv.index, method="ffill")
        ema3_of_ema9_1h = ema3_of_ema9_1h.reindex(forex_data.ohlcv.index, method="ffill")

        data.ma_short_1h = ema9_1h.to_numpy()
        data.ma_long_1h = ema3_of_ema9_1h.to_numpy()

        print(ema9)
        print()
        print(ema9_1h)

        return data

    def act(self, data: PrecomputedData, acc: Account):
        now = data.now

        # ===== 15m signals =====
        ma_short_15m = data.ma_short_15m[now]
        ma_long_15m = data.ma_long_15m[now]

        # ===== 1h signals =====
        ma_short_1h = data.ma_short_1h[now]
        ma_long_1h = data.ma_long_1h[now]

        buy_signal = (ma_short_15m > ma_long_15m) and (ma_short_1h > ma_long_1h)
        sell_signal = (ma_short_15m < ma_long_15m) and (ma_short_1h < ma_long_1h)

        order = acc.get_order()

        if buy_signal:
            if order and order.side == Side.SELL:
                acc.close_order()
            if not acc.have_order():
                acc.open_position(Side.BUY)

        elif sell_signal:
            if order and order.side == Side.BUY:
                acc.close_order()
            if not acc.have_order():
                acc.open_position(Side.SELL)
