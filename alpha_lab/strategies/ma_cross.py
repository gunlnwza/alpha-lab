from alpha_lab.backtest.account import Account, Side, OrderType
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData

# ---
# Config

# ---


class TemplateBot(BacktestBot):
    def __init__(self, name="template"):
        self.name = name

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        data = PrecomputedData(forex_data)
        return data

    def act(self, idx: int, data: PrecomputedData, acc: Account):
        pass
