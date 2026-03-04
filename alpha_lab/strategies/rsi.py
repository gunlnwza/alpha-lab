from alpha_lab.backtest.account import Account, Side, OrderType
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData

import numpy as np

RSI_PERIOD = 20
RSI_OFFSET = 20  # must be between [0, 50]

ATR_PERIOD = 20
SL_ATR_MUL = 1
TP_ATR_MUL = 3


class RsiBot(BacktestBot):
    def __init__(self):
        super().__init__("rsi")

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        data = PrecomputedData(forex_data)
        ohlcv = data.forex_data.ohlcv

        data.atr = ohlcv.ta.atr(ATR_PERIOD).to_numpy()
        data.rsi = ohlcv.ta.rsi(RSI_PERIOD).to_numpy()

        return data

    def act(self, data: PrecomputedData, acc: Account):
        bar = data.bar
        now = data.now

        rsi = data.rsi[now]
        atr = data.atr[now]
        if np.isnan(rsi) or np.isnan(atr):
            return
        
        if acc.have_order():
        #     sl = acc.get_order().sl
        #     if acc.get_order().side == Side.BUY:
        #         new_sl = bar.close - SL_ATR_MUL * atr
        #         acc.set_sl(max(sl, new_sl))
        #     else:
        #         new_sl = bar.close + SL_ATR_MUL * atr
        #         acc.set_sl(min(sl, new_sl))
            return

        if rsi < 50 - RSI_OFFSET:
            sl = bar.close - SL_ATR_MUL * atr
            tp = bar.close + TP_ATR_MUL * atr
            acc.open_position(Side.BUY, sl, tp)
        if rsi > 50 + RSI_OFFSET:
            sl = bar.close + SL_ATR_MUL * atr
            tp = bar.close - TP_ATR_MUL * atr
            acc.open_position(Side.SELL, sl, tp)
