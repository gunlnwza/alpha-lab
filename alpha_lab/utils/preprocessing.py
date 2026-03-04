import pandas as pd


def drop_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Saturday and Sunday (UTC time)."""
    weekday = df.index.weekday
    return df[((weekday >= 0) & (weekday <= 4))]


def split_timeseries(df: pd.DataFrame, freq: str = "W") -> list[pd.DataFrame]:
    """
    Split a time series dataframe into chunks of the given frequency.

    - `freq` (like what Pandas use):
        - 'h': hour
        - 'D': day
        - 'W': week
        - 'M': month
    """
    if freq not in ('h', 'D', 'W', 'M'):
        raise ValueError("Invalid chunk size")
    return [g for _, g in df.groupby(pd.Grouper(freq=freq))]


def inverse_ohlcv(ohlcv: pd.DataFrame):
    """Get the reciprocal prices"""
    return pd.DataFrame({
        "open": 1 / ohlcv["open"],
        "high": 1 / ohlcv["low"],
        "low": 1 / ohlcv["high"],
        "close": 1 / ohlcv["close"],
        "volume": ohlcv["volume"]
    })
