from alpha_lab.backtest.account import Account, Side, OrderType
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData

import pandas_ta as ta

SHORT_PERIOD = 50
LONG_PERIOD = 200


class MaCrossBot(BacktestBot):
    def __init__(self, name="ma_cross"):
        self.name = name

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        data = PrecomputedData(forex_data)

        data.ma_short = data.prices.ohlcv.ta.sma(SHORT_PERIOD).to_numpy()
        data.ma_long = data.prices.ohlcv.ta.sma(LONG_PERIOD).to_numpy()

        return data

    def act(self, idx: int, data: PrecomputedData, acc: Account):
        order = acc.get_order()
        close = data.prices.close[idx]

        if data.ma_short[idx] > data.ma_long[idx]:
            if acc.have_order():
                if order.side == Side.SELL:
                    acc.close_order(idx, close)
            else:
                acc.open_order(Side.BUY, OrderType.POSITION, idx, close)
        else:
            if acc.have_order():
                if order.side == Side.BUY:
                    acc.close_order(idx, close)
            else:
                acc.open_order(Side.SELL, OrderType.POSITION, idx, close)
