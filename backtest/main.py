import logging
from utils import ForexData
from simulation import Simulation

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    filemode="w"
)


def main():
    data = ForexData("twelve_data", "XAUUSD", "5min")
    # data.ohlcv = data.ohlcv[:500]

    sim = Simulation(data)
    sim.run()
    sim.result.report()
    sim.result.visualize()


if __name__ == "__main__":
    main()
