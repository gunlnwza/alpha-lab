from alpha_lab.utils import ForexData
from alpha_lab.backtest.account import Account
from alpha_lab.backtest.bot import BacktestBot
from alpha_lab.backtest.result import SimulationResult


class Simulation:
    def __init__(self, forex_data: ForexData, acc: Account, bot: BacktestBot):
        self.forex_data = forex_data
        self.acc = acc
        self.bot = bot

        self.result = None

    def run(self):
        data = self.bot.precompute_data(self.forex_data)

        while not data.is_last_bar():
            self.acc.process_bar(data.bar)
            self.bot.act(data, self.acc)
            self.acc.update_money(data.bar)
            data.step()

        if self.acc.have_order():
            self.acc.close_order(data.bar)
        self.acc.update_money(data.bar)

        self.result = SimulationResult(data, self.acc, self.bot)
