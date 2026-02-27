from alpha_lab.backtest.account import Account, Side, OrderType
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData

SHORT_PERIOD = 50
LONG_PERIOD = 200


class MaCrossBot(BacktestBot):
    def __init__(self, name="ma_cross"):
        self.name = name

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        data = PrecomputedData(forex_data)

        data.ma_short = data._forex_data.ohlcv.ta.sma(SHORT_PERIOD).to_numpy()
        data.ma_long = data._forex_data.ohlcv.ta.sma(LONG_PERIOD).to_numpy()

        return data

    def act(self, data: PrecomputedData, acc: Account):
        now = data.now
        bar = data.bar
        order = acc.get_order()

        if data.ma_short[now] > data.ma_long[now]:
            if acc.have_order():
                if order.side == Side.SELL:
                    acc.close_order(bar)
            else:
                acc.open_order(Side.BUY, OrderType.POSITION, now, bar.close)
        else:
            if acc.have_order():
                if order.side == Side.BUY:
                    acc.close_order(bar)
            else:
                acc.open_order(Side.SELL, OrderType.POSITION, now, bar.close)
