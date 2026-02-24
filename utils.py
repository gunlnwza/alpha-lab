"""
source: alpha_vantage, massive, twelve_data, tradingview
symbol: EURUSD, AUDUSD, ...
tf: 1min, 1hour, 1day, ...
"""

from pathlib import Path
import pandas as pd


###############################################################################
# Loading
###############################################################################


def _get_path(data_dir: str, source: str, symbol: str, tf: str, file_extension: str):
    symbol = symbol.upper()
    return Path(data_dir, source, symbol, f"{source}_{symbol}_{tf}.{file_extension}")


def load_csv(source: str, symbol: str, tf: str, *, data_dir="finloader_data") -> pd.DataFrame:
    path = _get_path(data_dir, source, symbol, tf, "csv")
    return pd.read_csv(path, index_col="time", parse_dates=True)


def load_parquet(source: str, symbol: str, tf: str, *, data_dir="finloader_data") -> pd.DataFrame:
    path = _get_path(data_dir, source, symbol, tf, "parquet")
    return pd.read_parquet(path)


###############################################################################
# Preprocessing
###############################################################################


def drop_weekend(df: pd.DataFrame) -> pd.DataFrame:
    weekday = df.index.weekday
    return df[((weekday >= 0) & (weekday <= 4))]


def get_chunks(df: pd.DataFrame, chunk_size: str = "W"):
    """
    chunk_size (like pandas time acronym):
    - 'h': hour
    - 'D': day
    - 'W': week
    - 'M': month
    """
    if chunk_size not in ('h', 'D', 'W', 'M'):
        raise ValueError("Invalid chunk size")
    return [g for _, g in df.groupby(pd.Grouper(freq=chunk_size))]
