import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import coint
import sys

from load import load_investing_dot_com_data, load_tradingview_data


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


###############################################################################

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 plot_rolling.py y x")

    load_func = load_tradingview_data
    a = sys.argv[1]
    b = sys.argv[2]
    y = load_func(a)['close']
    x = load_func(b)['close']

    rolling_pvals = rolling_stability(y, x, window=100)
    plt.title(f"Rolling Stability: {a} vs {b}")
    plt.plot(rolling_pvals)
    plt.show()
