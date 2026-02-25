from pathlib import Path
import joblib

import pandas as pd
import pandas_ta as ta
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

from utils import load_parquet, drop_weekend, get_chunks, triple_barrier_labels


def plot_labels(df: pd.DataFrame, signals: pd.Series):
    s = signals

    # Show stats
    n = len(s)
    tp_hit = s[s == 1].count()
    time_limit = s[s == 0].count()
    sl_hit = s[s == -1].count()
    print(f"{'':10} {'count':>5} {'percent':>7}")
    print(f"{'tp_hit':10} {tp_hit:5} {tp_hit / n:7.2f}")
    print(f"{'sl_hit':10} {sl_hit:5} {sl_hit / n:7.2f}")
    print(f"{'time_limit':10} {time_limit:5} {time_limit / n:7.2f}")
    print(f"{'total':10} {n:5} {1:7.2f}")

    # Ploting
    plt.figure(figsize=(15, 5))
    plt.title("Triple Barrier Labels")

    plt.plot(df.close,
        color="black",
        marker='o', markersize=1,
        linestyle=':', linewidth=0.5)

    # Buy label
    plt.scatter(df.index[s == 1], df.close[s == 1],
        color="green", s=15)

    # Sell label
    plt.scatter(df.index[s == -1], df.close[s == -1],
        color="red", s=15)

    plt.show()


def get_features_target(df: pd.DataFrame, signals: pd.Series = None):
    """
    Feature engineer `df`, and trim `signals` so there's no NaN.

    - Return only `X` if only df is provided.
    - Return both `X` and `y` if `signals` is also provided.
    """
    
    # TODO: use Kalman filter to smooth
    df = df.copy()

    df["smoothed_price"] = df.ta.sma(5)
    df["short"] = df.ta.ema(20)
    df["long"] = df.ta.ema(50)

    df["rsi"] = df.ta.rsi(20)

    df["smoothed_price/short"] = df["smoothed_price"] / df["short"]  # too much noise
    df["smoothed_smoothed_price/short"] = df["smoothed_price/short"].rolling(5).mean()

    df["smoothed_price/long"] = df["smoothed_price"] / df["long"]  # too much noise
    df["smoothed_smoothed_price/long"] = df["smoothed_price/long"].rolling(5).mean()
    
    df["short/long"] = df["short"] / df["long"]

    # boolean
    df["smoothed_price_above_short"] = (df["smoothed_price/short"] > 1).astype("int8")
    df["smoothed_price_above_long"]  = (df["smoothed_price/long"] > 1).astype("int8")
    df["short_above_long"]  = (df["short"] > df["long"]).astype("int8")
    df["rsi_overbought"] = (df["rsi"] > 70).astype("int8")
    df["rsi_oversold"] = (df["rsi"] < 30).astype("int8")

    cols = [
        "smoothed_smoothed_price/short",
        "smoothed_smoothed_price/long",
        "short/long",
        "rsi",

        "smoothed_price_above_short",
        "smoothed_price_above_long",
        "short_above_long",
        "rsi_overbought",
        "rsi_oversold"
    ]
    X = df[cols].dropna()
    if signals is None:
        return X, None

    y = signals.reindex(X.index).astype("int8")  # safe alignment
    return X, y


def validate_train_result(X_test, y_test, clf, name):
    y_pred = clf.predict(X_test)
    print(name)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def visual_backtest(df, clf, name):
    X, _ = get_features_target(df)

    proba = clf.predict_proba(X)[:, 1]   # probability of class 1
    y_pred = proba > 0.5                 # decision threshold

    X = X.copy()
    X["close"] = df["close"]

    fig, axes = plt.subplots(
        2, 1,
        height_ratios=[2, 1],
        figsize=(15, 7),
        sharex=True
    )

    # --- Price chart ---
    axes[0].set_title(name)
    axes[0].plot(X.index, X.close, label="Price", linewidth=1, color="black", alpha=0.5)
    axes[0].plot(df.ta.sma(5)[X.index])
    axes[0].plot(df.ta.ema(100)[X.index])
    axes[0].scatter(
        X.index[y_pred],
        X.close[y_pred],
        color="green",
        s=50,
        label="Buy"
    )
    axes[0].legend()

    # --- Probability panel ---
    axes[1].plot(X.index, proba, color="blue", label="P(buy)")
    axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5)
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    xauusd_5min = load_parquet("twelve_data", "xauusd", "5min")
    week_chunks = get_chunks(drop_weekend(xauusd_5min))

    df = week_chunks[0]
    signals = triple_barrier_labels(df, bars=36, tp_atr_mul=4, sl_atr_mul=1)
    signals = (signals == 1).astype("int8")
    plot_labels(df, signals)

    X, y = get_features_target(df, signals)  # Extract features and target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False, random_state=42)

    lr_clf = LogisticRegression(random_state=42)
    lr_clf.fit(X_train, y_train)

    validate_train_result(X_test, y_test, lr_clf, "Logistic Regression")
    visual_backtest(week_chunks[0], lr_clf, "Logistic Regression")
    visual_backtest(week_chunks[1], lr_clf, "Logistic Regression (Test | Week 1)")
    visual_backtest(week_chunks[2], lr_clf, "Logistic Regression (Test | Week 2)")

    path = Path("models", "logreg_v1.pkl")
    path.parent.mkdir(exist_ok=True)
    joblib.dump(lr_clf, path)
    print(f"Model saved to {path}!")


if __name__ == "__main__":
    main()
