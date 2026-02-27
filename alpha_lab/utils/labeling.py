import pandas as pd
import pandas_ta as ta
import numpy as np


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    bars: int = 20,
    atr_period: int = 14,
    tp_atr_mul: float = 1.0,
    sl_atr_mul: float = 1.0,
) -> pd.Series:
    """
    Triple Barrier labeling.

    Returns:
        1  -> Take Profit hit first
        0  -> Time limit reached
        -1 -> Stop Loss hit first
    """

    if min(bars, atr_period, tp_atr_mul, sl_atr_mul) <= 0:
        raise ValueError("All parameters must be positive")

    n = len(df)
    labels = np.zeros(n, dtype=np.int8)

    # --- Smoothed price (2-pass EMA)
    price = ta.ema(ta.ema(df["close"], 10), 10).to_numpy()
    atr = ta.atr(df["high"], df["low"], df["close"], atr_period).to_numpy()

    # First valid index where both price and ATR exist
    valid_mask = ~np.isnan(price) & ~np.isnan(atr)
    if not valid_mask.any():
        raise ValueError("No valid window for triple barrier labeling")

    start = np.argmax(valid_mask)

    # Precompute barriers
    tp = price + tp_atr_mul * atr
    sl = price - sl_atr_mul * atr

    for i in range(start, n - bars):

        if not valid_mask[i]:
            continue

        future = price[i + 1 : i + bars + 1]

        upper_hits = np.flatnonzero(future > tp[i])
        lower_hits = np.flatnonzero(future < sl[i])

        if upper_hits.size and lower_hits.size:
            labels[i] = 1 if upper_hits[0] < lower_hits[0] else -1
        elif upper_hits.size:
            labels[i] = 1
        elif lower_hits.size:
            labels[i] = -1
        # else remains 0 (time barrier)

    return pd.Series(labels, index=df.index, name="barrier_hit")
