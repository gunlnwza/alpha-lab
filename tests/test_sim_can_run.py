import pytest

from alpha_lab.utils import ForexData
from alpha_lab.backtest.simulation import Simulation

from alpha_lab.strategies.hold import HoldBot
from alpha_lab.strategies.ma_cross import MaCrossBot
from alpha_lab.strategies.buy_limit import BuyLimitBot


@pytest.fixture
def forex_data():
    return ForexData("twelve_data", "xauusd", "1day")


def test_hold_bot(forex_data):
    bot = HoldBot()
    sim = Simulation(forex_data, bot)
    sim.run()


def test_buy_limit_bot(forex_data):
    bot = BuyLimitBot()
    sim = Simulation(forex_data, bot)
    sim.run()


def test_ma_cross_bot(forex_data):
    bot = MaCrossBot()
    sim = Simulation(forex_data, bot)
    sim.run()
