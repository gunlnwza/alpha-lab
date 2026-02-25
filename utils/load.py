from pathlib import Path
import pandas as pd


def _get_path(data_dir: str, source: str, symbol: str, tf: str, file_extension: str):
    symbol = symbol.upper()
    return Path(data_dir, source, symbol, f"{source}_{symbol}_{tf}.{file_extension}")


def load_csv(source: str, symbol: str, tf: str, *, data_dir="finloader_data") -> pd.DataFrame:
    path = _get_path(data_dir, source, symbol, tf, "csv")
    return pd.read_csv(path, index_col="time", parse_dates=True)


def load_parquet(source: str, symbol: str, tf: str, *, data_dir="finloader_data") -> pd.DataFrame:
    path = _get_path(data_dir, source, symbol, tf, "parquet")
    return pd.read_parquet(path)
