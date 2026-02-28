from alpha_lab.backtest.account import Account, Side, OrderType
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData

import pandas_ta as ta

SHORT_PERIOD = 100
LONG_PERIOD = 200

ADX_THRESHOLD = 25

ATR_PERIOD = 10
ATR_MEAN_PERIOD = 10
ATR_MUL = 3

class TrendGateBot(BacktestBot):
    def __init__(self):
        super().__init__("trend_gate")

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        data = PrecomputedData(forex_data)
        ohlcv = data.forex_data.ohlcv

        adx_df = ohlcv.ta.adx()
        data.adx = adx_df["ADX_14"].to_numpy()

        atr = ohlcv.ta.atr(ATR_PERIOD)
        data.atr = atr.to_numpy()
        data.atr_mean = ta.sma(atr, ATR_MEAN_PERIOD).to_numpy()

        data.ma_short = ohlcv.ta.ema(SHORT_PERIOD).to_numpy()
        data.ma_long = ohlcv.ta.ema(LONG_PERIOD).to_numpy()

        return data

    def act(self, data: PrecomputedData, acc: Account):
        bar = data.bar
        now = data.now

        short = data.ma_short[now - 1:now + 1]
        long = data.ma_long[now - 1:now + 1]
        if len(short) < 2:
            return

        adx = data.adx[now]
        atr = data.atr[now]
        atr_mean = data.atr_mean[now]
        close = bar.close

        if short[-1] > long[-1] and short[-2] < long[-2]:
            if acc.have_order():
                order = acc.get_order()
                if order.side == Side.BUY:
                    acc.set_sl(max(order.sl, close - ATR_MUL * atr_mean))
                else:
                    acc.close_order()
            if not acc.have_order() and adx >= ADX_THRESHOLD:
                acc.open_position(
                    Side.BUY,
                    sl=close - ATR_MUL * atr_mean
                )

        elif short[-1] < long[-1] and short[-2] > long[-2]:
            if acc.have_order():
                order = acc.get_order()
                if order.side == Side.BUY:
                    acc.close_order()
                else:
                    acc.set_sl(min(order.sl, close + ATR_MUL * atr_mean))
            if not acc.have_order() and adx >= ADX_THRESHOLD:
                acc.open_position(
                    Side.SELL,
                    sl=close + ATR_MUL * atr_mean
                )
