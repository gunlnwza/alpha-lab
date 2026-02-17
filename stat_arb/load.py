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


def load_tradingview_data(filename: str, dir="data/tradingview", unix=True) -> pd.DataFrame:
    df = pd.read_csv(f"{dir}/{filename}.csv")
    df["time"] = pd.to_datetime(df['time'], unit='s')
    df.index = df["time"]
    df.drop("time", axis=1, inplace=True)
    return df


###############################################################################

def join_close_prices(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    lst = [df['close'].rename(name) for name, df in dfs.items()]
    df = pd.concat(lst, axis=1).dropna()
    return df


###############################################################################

if __name__ == "__main__":
    df = load_tradingview_data('kbankf')
    print(df.dtypes)
    print(df)
