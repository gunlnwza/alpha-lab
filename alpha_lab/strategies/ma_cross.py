from alpha_lab.backtest.account import Account, Side
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData

SHORT_PERIOD = 50
LONG_PERIOD = 200


class MaCrossBot(BacktestBot):
    def __init__(self):
        super().__init__("ma_cross")

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        data = PrecomputedData(forex_data)

        data.ma_short = data._forex_data.ohlcv.ta.sma(SHORT_PERIOD).to_numpy()
        data.ma_long = data._forex_data.ohlcv.ta.sma(LONG_PERIOD).to_numpy()

        return data

    def act(self, data: PrecomputedData, acc: Account):
        now = data.now
        ma_short = data.ma_short[now]
        ma_long = data.ma_long[now]

        order = acc.get_order()

        if ma_short > ma_long:
            if order and order.side == Side.SELL:
                acc.close_order()
            if not acc.have_order():
                acc.open_position(Side.BUY)
        else:
            if order and order.side == Side.BUY:
                acc.close_order()
            if not acc.have_order():
                acc.open_position(Side.SELL)
