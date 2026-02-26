import joblib
import pandas as pd

from simulation import SimulationData

from models.features import get_features


def get_signals(data: SimulationData):
    # Two separate models
    # 1. Tactical (low level, low TF)
    X = get_features(data._ohlcv)
    low_tf_clf = joblib.load("models/artifacts/logreg_v1.pkl")
    low_tf_raw = pd.Series(low_tf_clf.predict(X), index=X.index)

    # Align tactical signal to full OHLC index
    low_tf = low_tf_raw.reindex(data._ohlcv.index)

    # 2. Strategic (high level, high TF)
    ma_short = pd.Series(data.ma_short, index=data._ohlcv.index)
    ma_long = pd.Series(data.ma_long, index=data._ohlcv.index)
    high_tf = (ma_short > ma_long)

    # Final combined signal
    # Only valid where low_tf exists AND gate is True
    pred_full = (low_tf == 1) & (high_tf == True)

    return pred_full