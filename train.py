from pathlib import Path
import joblib
import sys

import pandas as pd
import pandas_ta as ta

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

from utils import load_parquet, drop_weekend, divide_timeseries, triple_barrier_labels
from features import get_features, get_features_labels


def plot_buy_labels(df: pd.DataFrame, signals: pd.Series):
    plt.figure(figsize=(15, 5))
    plt.title("Buy Labels")

    plt.plot(
        df.close,
        color="black",
        marker='o', markersize=1,
        linestyle=':', linewidth=0.5
    )

    plt.scatter(
        df.index[signals == 1],
        df.close[signals == 1],
        color="green",
        s=15
    )

    plt.show()


def validate_train_result(X_test, y_test, clf, name):
    y_pred = clf.predict(X_test)
    print(name)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def visual_backtest(df, clf, name):
    X = get_features(df)

    THRESHOLD = 0.5
    proba = clf.predict_proba(X)[:, 1]   # probability of class 1
    y_pred = proba > THRESHOLD

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
    axes[0].plot(ta.kama(X["close"], 50))
    axes[0].plot(ta.kama(X["close"], 100))
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
    axes[1].axhline(THRESHOLD, color="red", linestyle="--", alpha=0.5)
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    xauusd_5min = load_parquet("twelve_data", "xauusd", "5min")
    week_chunks = divide_timeseries(drop_weekend(xauusd_5min))
    df = week_chunks[0]

    signals = triple_barrier_labels(df, bars=36, tp_atr_mul=4, sl_atr_mul=1)
    signals = (signals == 1).astype("int8")
    plot_buy_labels(df, signals)
    if len(sys.argv) == 2:
        sys.exit()

    X, y = get_features_labels(df, signals)
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
