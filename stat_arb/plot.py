import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import coint
import sys
import seaborn as sns

from utils import load_investing_dot_com_data, load_tradingview_data, join_close_prices

WINDOW = 250

def rolling_stability(a, b):
    rolling_pvals = []
    for i in range(WINDOW, len(a)):
        try:
            _, pval, _ = coint(
                a.iloc[i-WINDOW:i], # y
                b.iloc[i-WINDOW:i] # x
            )
            rolling_pvals.append(pval)
        except ZeroDivisionError:
            warnings.warn("ZeroDivisionError occured in rolling_stability()")
            continue
    return rolling_pvals


def plot_rolling_statbility(prices):
    res = rolling_stability(prices.a, prices.b)
    plt.plot(res)
    plt.show()


def plot_prices_spread_z(prices):
    spread = prices.a - prices.b
    rolling_mean = spread.rolling(WINDOW).mean()
    rolling_std = spread.rolling(WINDOW).std()
    zscore = (spread - rolling_mean) / rolling_std

    fig, axes = plt.subplots(3, 1, sharex=True, height_ratios=[2, 2,1], figsize=(10,8))

    axes[0].set_title(f"{a} vs {b}")
    axes[0].plot(prices.a)
    axes[0].plot(prices.b)

    axes[1].set_title(f"Spread = {a} - {b}")
    axes[1].plot(spread)
    axes[1].plot(rolling_mean, label=f"Rolling Mean ({WINDOW}d)")
    axes[1].legend()

    axes[2].set_title(f"Z-Score ({WINDOW}d)")
    axes[2].plot(zscore)
    axes[2].axhline(0, linestyle="--")
    axes[2].axhline(2, linestyle="--")
    axes[2].axhline(-2, linestyle="--")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    a = "TFEX_GO1!"
    b = "TFEX_GO2!"
    df1 = load_tradingview_data(a)
    df2 = load_tradingview_data(b)
    prices = join_close_prices(df1, df2)

    # plot_rolling_statbility(prices)
    plot_prices_spread_z(prices)
