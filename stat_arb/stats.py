from itertools import combinations
import sys

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm

from utils import load_investing_dot_com_data, load_tradingview_data, join_close_prices

P_VALUE = 0.05


def adf_test(series: pd.Series) -> float:
    res = adfuller(series)
    p_value = res[1]
    return p_value


def hedge_ratio(y, x):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    return model.params.iloc[1]


def spread_half_life(y, x):
    beta = hedge_ratio(y, x)
    spread = y - beta * x

    spread_lag = spread.shift(1)
    spread_ret = spread - spread_lag
    df = pd.concat([spread_ret, spread_lag], axis=1).dropna()

    model = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1])).fit()
    lambda_ = model.params.iloc[1]

    # No mean reversion
    if lambda_ >= 0:
        return np.inf

    # More precise half-life calculation
    phi = 1 + lambda_
    half_life = -np.log(2) / np.log(abs(phi))

    return half_life


if __name__ == "__main__":
    # if len(sys.argv) != 3:
        # sys.exit("Usage: python3 filter_spread.py name1 name2")

    # df1 = load_tradingview_data(sys.argv[1])
    # df2 = load_tradingview_data(sys.argv[2])
    a = "TFEX_GO1!"
    b = "TFEX_GO2!"
    df1 = load_tradingview_data(a)
    df2 = load_tradingview_data(b)
    prices = join_close_prices(df1, df2)

    print(f'x p_value={adf_test(prices.a):.4f}')
    print(f'x diff p_value={adf_test(prices.a.diff().dropna()):.4f}')
    print()
    print(f'y p_value={adf_test(prices.b):.4f}')
    print(f'y diff p_value={adf_test(prices.b.diff().dropna()):.4f}')
    print()
    print(f'coint(x, y) p_value={coint(prices.a, prices.b)[1]:.4f}')
    print(f'coint(y, x) p_value={coint(prices.b, prices.a)[1]:.4f}')
    print()
    
    beta = hedge_ratio(prices.a, prices.b)
    spread = prices.a - beta * prices.b
    print(f"(x, y), spread_p_val={adf_test(spread):.4f}")
    beta = hedge_ratio(prices.b, prices.a)
    spread = prices.b - beta * prices.a
    print(f"(y, x), spread_p_val={adf_test(spread):.4f}")
    print()

    print(f"half-life: {spread_half_life(prices.a, prices.b)}")
