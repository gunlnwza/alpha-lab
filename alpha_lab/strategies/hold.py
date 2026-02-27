from alpha_lab.backtest.account import Account, Side, OrderType
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData


class HoldBot(BacktestBot):
    def __init__(self, name="buy_and_hold"):
        self.name = name

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        data = PrecomputedData(forex_data)
        return data

    def act(self, idx: int, data: PrecomputedData, acc: Account):
        if not acc.have_order():
            acc.open_order(Side.BUY, OrderType.POSITION, idx, data.prices.close[idx])
