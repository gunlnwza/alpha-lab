from utils import ForexData
from backtest.simulation.simulate import simulate


def main():
    data = ForexData("twelve_data", "XAUUSD", "5min")
    res = simulate(data)
    res.report()
    res.visualize()


if __name__ == "__main__":
    main()
