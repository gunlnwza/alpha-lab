from pathlib import Path
import pandas as pd

from .preprocess import drop_weekend, inverse_ohlcv

_DATA_DIR = Path(__file__).parents[1] / ".finloader_data"


def _get_path(source: str, symbol: str, tf: str, extension: str):
    symbol = symbol.upper()
    return Path(_DATA_DIR, source, symbol, f"{source}_{symbol}_{tf}.{extension}")


def load_csv(source: str, symbol: str, tf: str) -> pd.DataFrame:
    path = _get_path(source, symbol, tf, "csv")
    return pd.read_csv(path, index_col="time", parse_dates=True)


def load_parquet(source: str, symbol: str, tf: str) -> pd.DataFrame:
    path = _get_path(source, symbol, tf, "parquet")
    return pd.read_parquet(path)


class ForexData:
    def __init__(self, source: str, symbol: str, tf: str):
        self.source = source
        self.symbol = symbol.upper()
        self.tf = tf

        ohlcv_raw = load_parquet(source, symbol, tf)
        self.ohlcv = drop_weekend(ohlcv_raw)  # remove weekend, like most charting software

        self.inv_ohlcv = inverse_ohlcv(self.ohlcv)

    def __str__(self):
        return f"{self.source}'s {self.symbol} ({self.tf})"

    def __repr__(self):
        return f"ForexData({self.source}, {self.symbol}, {self.tf})"
