from alpha_lab.backtest.account import Account, Side, OrderType
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData


class HoldBot(BacktestBot):
    def __init__(self, name="buy_and_hold"):
        self.name = name

    def act(self, data: PrecomputedData, acc: Account):
        if not acc.have_order():
            acc.open_order(Side.BUY, OrderType.POSITION, data.now, data._forex_data.close[data.now])
