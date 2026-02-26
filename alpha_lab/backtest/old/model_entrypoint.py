from pathlib import Path

import pandas as pd
import joblib

from alpha_lab.models.features import get_features

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from alpha_lab.backtest.simulation import SimulationData

# TODO: different configs for different assets

SL_VOL_MUL = 10

GATE_MA_SHORT_PERIOD = 50
GATE_MA_LONG_PERIOD = 200

ATR_PERIOD = 10


def get_signals(data: "SimulationData"):
    """
    Return time-index aligned signals
    - Vectorized for most things, hopefully with no lookahead bias
    """

    # Two separate models
    # 1. Tactical (low level, low TF)
    X = get_features(data._ohlcv)
    low_tf_clf = joblib.load(Path("alpha_lab", "models", "artifacts", "logreg_v1.pkl"))
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

    assert pred_full.index.to_list() == data._ohlcv.index.to_list()
    return pred_full
