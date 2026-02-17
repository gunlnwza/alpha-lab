from itertools import combinations
import sys

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm

from load import load_investing_dot_com_data, load_tradingview_data, join_close_prices

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

    spread_lag = spread.shift(1).dropna()
    spread_ret = spread.diff().dropna()

    model = sm.OLS(spread_ret, sm.add_constant(spread_lag)).fit()
    lambda_ = model.params.iloc[1]

    half_life = -np.log(2) / lambda_
    return half_life

###############################################################################


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        sys.exit("Usage: python3 filter_spread.py name1 name2 ...")
    names = sys.argv[1:]

    # load_func = load_investing_dot_com_data
    load_func = load_tradingview_data
    
    dfs = {name: load_func(name) for name in names}
    prices = join_close_prices(dfs)

    # enforce node consistency
    print("Price stationarity checks:")
    for name in prices.columns:
        orig_not_stationary = adf_test(prices[name]) >= P_VALUE
        diff_stationary = adf_test(prices[name].diff().dropna()) < P_VALUE
        if not (orig_not_stationary and diff_stationary):
            if not orig_not_stationary:
                print(f"{name} is stationary")
            if not diff_stationary:
                print(f"{name}'s diff is not stationary")
            prices.drop(name, axis=1, inplace=True)

    # enforce arc consistency
    print("\nCoint and spread's stationarity check:")
    res = []
    for a, b in combinations(prices.columns, 2):
        y = prices[a]
        x = prices[b]
        score, p_value, _ = coint(y, x)
        if p_value >= P_VALUE:
            print((a, b), "coint is bad")
            continue

        beta = hedge_ratio(y, x)
        spread = y - beta * x  # p-value must < 0.05, spread must be stationary
        spread_stationary = adf_test(spread) < 0.05
        if not spread_stationary:
            print((a, b), "spread is not stationary")
            continue
        res.append((a, b))

    print("\nValid stat arb pairs:")
    print(*res, sep="\n")
