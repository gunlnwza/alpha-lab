import time

from alpha_lab.utils import ForexData
from alpha_lab.backtest.account import Account
from alpha_lab.backtest.bot import BacktestBot
from alpha_lab.backtest.result import SimulationResult


class Simulation:
    def __init__(self, forex_data: ForexData, bot: BacktestBot):
        self.forex_data = forex_data
        self.bot = bot

        self.result = None

    def run(self):
        time_start = time.perf_counter()

        data = self.bot.precompute_data(self.forex_data)
        acc = Account(data)
        bot = self.bot

        while not data.is_last_bar():
            acc._process_bar()
            bot.act(data, acc)
            acc._update_money()
            data.step()

        if acc.have_order():
            acc.close_order()
        acc._update_money()

        time_end = time.perf_counter()
        sim_time = time_end - time_start

        self.result = SimulationResult(data, acc, bot, sim_time)
