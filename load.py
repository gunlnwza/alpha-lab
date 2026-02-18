from pathlib import Path
import pandas as pd


def load_data(source: str, symbol: str, tf: str, *, data_dir="data/"):
    """
    source: alpha_vantage, massive, twelve_data, tradingview
    symbol: EURUSD, AUDUSD, ...
    tf: 1min, 1hour, 1day, ...
    """
    symbol = symbol.upper()
    data_dir = Path(data_dir)

    filename = f"{source}_{symbol}_{tf}.csv"
    filepath = data_dir / source / symbol / filename

    df = pd.read_csv(filepath, index_col="time", parse_dates=True)
    return df
