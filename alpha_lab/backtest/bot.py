from abc import ABC, abstractmethod

from alpha_lab.backtest.account import Account
from alpha_lab.utils import ForexData


class PrecomputedData:
    def __init__(self, forex_data: ForexData):
        self.prices = forex_data
        self.signals = None
        self.misc = {}


class BacktestBot(ABC):
    def __init__(self, name="abstract_base_class"):
        self.name = name

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        """
        Compute time-index aligned data for the whole chart.
        - Vectorized for most things, hopefully with no lookahead bias.
        - Will return `PrecomputedData`, wrapping at least `ForexData`.
        """
        data = PrecomputedData(forex_data)
        return data

    @abstractmethod
    def act(self, idx: int, data: PrecomputedData, acc: Account):
        """
        Action after candle[idx] has ended, must not peak beyond i > idx.
        """
        pass
