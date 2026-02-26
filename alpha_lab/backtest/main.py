import sys
import logging
import argparse

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
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    args = parser.parse_args()

    try:
        forex_data = ForexData("twelve_data", args.symbol, "5min")
    except RuntimeError as e:
        sys.exit(e)

    acc = Account()
    bot = BacktestBot()

    sim = Simulation(forex_data, acc, bot)
    sim.run()
    sim.result.report()
    sim.result.visualize()


if __name__ == "__main__":
    main()
