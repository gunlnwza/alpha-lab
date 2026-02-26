import logging
from alpha_lab.utils import ForexData
from alpha_lab.backtest.simulation import Simulation

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    filemode="w"
)


def main():
    forex_data = ForexData("twelve_data", "XAUUSD", "5min")

    sim = Simulation(forex_data)
    sim.run()
    sim.result.report()
    sim.result.visualize()


if __name__ == "__main__":
    main()
