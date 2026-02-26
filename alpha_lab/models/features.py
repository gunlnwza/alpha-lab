import pandas as pd
import pandas_ta as ta


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return raw `X`
    - `df` is a copy of OHLCV dataframe
    """

    # indicator
    df["smooth(price)"] = ta.linreg(df["close"], 20)
    df["short"] = ta.ema(df["close"], 20)
    df["long"] = ta.ema(df["close"], 50)
    df["rsi"] = ta.rsi(df["close"], 20)

    # numeric features
    df["smooth(price)/short"] = df["smooth(price)"] / df["short"]  # too much noise, need smoothing
    df["smooth(smooth(price)/short)"] = ta.linreg(df["smooth(price)/short"], 5)

    df["smooth(price)/long"] = df["smooth(price)"] / df["long"]  # too much noise, need smoothing
    df["smooth(smooth(price)/long)"] = ta.linreg(df["smooth(price)/long"], 5)

    df["short/long"] = df["short"] / df["long"]

    # boolean features
    df["smooth(price)_above_short"] = (df["smooth(price)/short"] > 1).astype("int8")
    df["smooth(price)_above_long"]  = (df["smooth(price)/long"] > 1).astype("int8")
    df["short_above_long"]  = (df["short"] > df["long"]).astype("int8")
    df["rsi_overbought"] = (df["rsi"] > 70).astype("int8")
    df["rsi_oversold"] = (df["rsi"] < 30).astype("int8")

    return df[[
        # numeric
        "smooth(smooth(price)/short)",
        "smooth(smooth(price)/long)",
        "short/long",
        "rsi",

        # boolean
        "smooth(price)_above_short",
        "smooth(price)_above_long",
        "short_above_long",
        "rsi_overbought",
        "rsi_oversold"
    ]]


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = _compute_features(df).dropna()
    return X


def get_features_labels(df: pd.DataFrame, signals: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Trim both `X`'s and `y`'s NaN, removing the warmup period in engineered `df`"""
    X = get_features(df)
    y = signals.reindex(X.index).astype("int8")  # safe alignment
    return X, y
