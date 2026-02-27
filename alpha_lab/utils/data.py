from pathlib import Path

import pandas as pd

from alpha_lab.utils.preprocessing import drop_weekend

_DATA = Path(__file__).parents[2] / ".finloader_data"


def _get_data_path(source: str, symbol: str, tf: str, extension: str):
    symbol = symbol.upper()
    return Path(_DATA, source, symbol, f"{source}_{symbol}_{tf}.{extension}")


def load_csv(source: str, symbol: str, tf: str) -> pd.DataFrame:
    path = _get_data_path(source, symbol, tf, "csv")
    return pd.read_csv(path, index_col="time", parse_dates=True)


def load_parquet(source: str, symbol: str, tf: str) -> pd.DataFrame:
    path = _get_data_path(source, symbol, tf, "parquet")
    return pd.read_parquet(path)


class ForexData:
    def __init__(self, source: str, symbol: str, tf: str):
        self.source = source
        self.symbol = symbol.upper()
        self.tf = tf

        if self.symbol in ["XAUUSD", "USDJPY"]:
            self.decimal_places = 2
            self.tick_size = 0.01
        else:
            self.decimal_places = 4
            self.tick_size = 0.0001

        try:
            ohlcv_raw = load_parquet(source, symbol, tf)
        except FileNotFoundError as e:
            raise e
        self.ohlcv = drop_weekend(ohlcv_raw)  # remove weekends, like most charting software

        self.open = self.ohlcv.open.to_numpy()
        self.high = self.ohlcv.high.to_numpy()
        self.low = self.ohlcv.low.to_numpy()
        self.close = self.ohlcv.close.to_numpy()
        self.volume = self.ohlcv.volume.to_numpy()

    def __len__(self):
        return len(self.ohlcv)

    def __str__(self):
        return f"{self.source}'s {self.symbol} ({self.tf})"

    def __repr__(self):
        return f"ForexData({self.source}, {self.symbol}, {self.tf})"
