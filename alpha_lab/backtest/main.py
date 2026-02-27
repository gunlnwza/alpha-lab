import sys
import logging
import argparse
import importlib

from alpha_lab.backtest.simulation import Simulation
from alpha_lab.backtest.account import Account
from alpha_lab.utils import ForexData

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    filemode="w"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("symbol")
    parser.add_argument("tf")
    parser.add_argument("strategy")
    args = parser.parse_args()

    try:
        forex_data = ForexData(args.source, args.symbol, args.tf)
    except RuntimeError as e:
        sys.exit(e)

    try:
        strategy_name = args.strategy  # e.g. "log_reg", "buy_limit"
        module = importlib.import_module(f"alpha_lab.strategies.{strategy_name}")
        class_name = "".join(part.capitalize() for part in strategy_name.split("_")) + "Bot"  # Convention: class name is CamelCase + "Bot"
        BotClass = getattr(module, class_name)
        bot = BotClass()
    except (ModuleNotFoundError, AttributeError) as e:
        sys.exit(f"Cannot load strategy '{args.strategy}'")

    acc = Account()
    sim = Simulation(forex_data, acc, bot)
    sim.run()
    sim.result.report()
    sim.result.visualize()


if __name__ == "__main__":
    main()
