import logging
from alpha_lab.backtest.simulation import Simulation
from alpha_lab.backtest.account import Account
from alpha_lab.backtest.bot import BacktestBot
from alpha_lab.utils import ForexData

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    filemode="w"
)


def main():
    forex_data = ForexData("twelve_data", "XAUUSD", "5min")
    acc = Account()
    bot = BacktestBot()

    sim = Simulation(forex_data, acc, bot)
    sim.run()
    sim.result.report()
    sim.result.visualize()


if __name__ == "__main__":
    main()
