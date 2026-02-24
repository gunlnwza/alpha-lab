import pandas as pd


def load_investing_dot_com_data(filename: str, dir="data/investing.com") -> pd.DataFrame:
    df = pd.read_csv(f"{dir}/{filename}.csv", parse_dates=True, index_col="Date")

    df.drop("Change %", axis=1, inplace=True)

    mapper = {
        "Price": "close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Vol.": "volume"
    }
    df.rename(mapper, axis=1, inplace=True)
    df.index.name = "datetime"

    return df


def load_tradingview_data(filename: str, dir="backup_data_18_Feb/tradingview") -> pd.DataFrame:
    df = pd.read_csv(f"{dir}/{filename}.csv")
    df["time"] = pd.to_datetime(df['time'], unit='s')
    df.index = df["time"]
    df.drop("time", axis=1, inplace=True)
    return df


def join_close_prices(df1, df2) -> pd.DataFrame:
    lst = [df1['close'].rename('a'), df2['close'].rename('b')]
    df = pd.concat(lst, axis=1, sort=True).dropna()
    return df
