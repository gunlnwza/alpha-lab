from alpha_lab.backtest.account import Account, Side, OrderType
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData


class TemplateBot(BacktestBot):
    def __init__(self):
        super().__init__("template")

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        return super().precompute_data(forex_data)

    def act(self, data: PrecomputedData, acc: Account):
        pass
