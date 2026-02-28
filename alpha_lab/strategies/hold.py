from alpha_lab.backtest.account import Account, Side
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData


class HoldBot(BacktestBot):
    def __init__(self):
        super().__init__("hold")

    def act(self, data: PrecomputedData, acc: Account):
        if not acc.have_order():
            acc.open_position(Side.BUY)
