from itertools import combinations
import warnings

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm

import matplotlib.pyplot as plt


def load_investing_dot_com_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=True, index_col="Date")

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


def join_close_prices(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    lst = [df['close'].rename(name) for name, df in dfs.items()]
    df = pd.concat(lst, axis=1).dropna()
    return df


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


def rolling_stability(y, x, window=100):
    rolling_pvals = []
    for i in range(window, len(y)):
        try:
            _, pval, _ = coint(
                y.iloc[i-window:i],
                x.iloc[i-window:i]
            )
            rolling_pvals.append(pval)
        except ZeroDivisionError:
            warnings.warn("ZeroDivisionError occured in rolling_stability()")
            continue
    return rolling_pvals


if __name__ == "__main__":
    names = ["kbank", "scb", "bbl"]
    dfs = {name: load_investing_dot_com_data(f"data/{name}.csv") for name in names}
    prices = join_close_prices(dfs)
    
    # enforce node consistency
    for name in prices.columns:
        print(name, adf_test(prices[name]))
    print()

    for name in prices.columns:
        print(name, 'diff', adf_test(prices[name].diff().dropna()))
    print()

    # enforce arc consistency
    for a, b in combinations(prices.columns, 2):
        score, pvalue, _ = coint(prices[a], prices[b])
        print(a, b, score, pvalue)
    print()

    # run each valid pairs
    for a, b in combinations(('kbank', 'scb', 'bbl'), 2):
        y = prices[a]
        x = prices[b]
        beta = hedge_ratio(y, x)
        spread = y - beta * x  # p-value must < 0.05, spread must be stationary
        print(a, b, "beta", beta, "p-value spread", adf_test(spread))
    
        rolling_pvals = rolling_stability(y, x, window=50)
        plt.title(f"{a} vs {b}")
        plt.plot(rolling_pvals)
        plt.show()
