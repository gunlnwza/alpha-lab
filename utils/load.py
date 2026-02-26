from pathlib import Path
import pandas as pd

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
