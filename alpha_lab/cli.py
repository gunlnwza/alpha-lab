import sys
import argparse
import importlib

from alpha_lab.utils import ForexData
from alpha_lab.backtest.simulation import Simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("symbol")
    parser.add_argument("tf")
    parser.add_argument("strategy")
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("--result_name", default="result")
    args = parser.parse_args()

    try:
        forex_data = ForexData(args.source, args.symbol, args.tf)
    except FileNotFoundError as e:
        sys.exit(f"Cannot load forex data '{args.source}'s {args.symbol.upper()} ({args.tf})'")

    try:
        strategy_name = args.strategy  # e.g. "log_reg", "buy_limit"
        module = importlib.import_module(f"alpha_lab.strategies.{strategy_name}")
        class_name = "".join(part.capitalize() for part in strategy_name.split("_")) + "Bot"  # Convention: class name is CamelCase + "Bot"
        BotClass = getattr(module, class_name)
        bot = BotClass()
    except (ModuleNotFoundError, AttributeError) as e:
        sys.exit(f"Cannot load strategy '{args.strategy}'")

    sim = Simulation(forex_data, bot)
    sim.run()

    sim.result.report()
    sim.result.render(f"result.png")
    if args.interactive:
        sim.result.show()


if __name__ == "__main__":
    main()
