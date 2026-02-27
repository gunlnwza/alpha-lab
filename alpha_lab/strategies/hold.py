from alpha_lab.backtest.account import Account, Side, OrderType
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData


class HoldBot(BacktestBot):
    def __init__(self):
        super().__init__("buy_and_hold")

    def act(self, data: PrecomputedData, acc: Account):
        if not acc.have_order():
            acc.open_position(Side.BUY, data.bar)
