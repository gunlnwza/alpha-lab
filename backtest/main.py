import logging
from utils import ForexData
from simulation import simulate

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    filemode="w"   # <-- this overwrites the file
)


def main():
    data = ForexData("twelve_data", "XAUUSD", "5min")
    # data.ohlcv = data.ohlcv[:500]

    res = simulate(data)
    res.report()
    res.visualize()


if __name__ == "__main__":
    main()
