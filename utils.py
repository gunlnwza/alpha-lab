"""
source: alpha_vantage, massive, twelve_data, tradingview
symbol: EURUSD, AUDUSD, ...
tf: 1min, 1hour, 1day, ...
"""

from pathlib import Path
import pandas as pd
import pandas_ta as ta
import numpy as np


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


def label_triple_barrier(
    df: pd.DataFrame,
    *,
    bars=20,
    atr_period=14,
    tp_atr_mul=1,
    sl_atr_mul=1
) -> pd.DataFrame:
    """Return one-hot encoding of what horizontal barrier was hit first"""

    if bars <= 0 or atr_period <= 0 or tp_atr_mul <= 0 or sl_atr_mul <= 0:
        raise ValueError("parameters must be positive")

    df = df.copy()
    df.ta.atr(length=atr_period, append=True)

    close = df["close"].to_numpy()
    atr = df[f"ATRr_{atr_period}"].to_numpy()

    tp = close + tp_atr_mul * atr
    sl = close - sl_atr_mul * atr
    df["tp"] = tp
    df["sl"] = sl

    n = len(df)
    tp_hit = np.zeros(n, dtype=bool)
    sl_hit = np.zeros(n, dtype=bool)

    valid_start = np.where(~np.isnan(atr))[0][0]
    for i in range(valid_start, n - bars):
        future_prices = close[i+1:i+bars+1]

        upper = future_prices > tp[i]
        lower = future_prices < sl[i]
        tp_cross = upper.any()
        sl_cross = lower.any()

        if tp_cross and sl_cross:
            if np.argmax(upper) < np.argmax(lower):
                tp_hit[i] = True
            else:
                sl_hit[i] = True
        elif tp_cross:
            tp_hit[i] = True
        elif sl_cross:
            sl_hit[i] = True

    df["tp_hit"] = tp_hit
    df["sl_hit"] = sl_hit
    return df
