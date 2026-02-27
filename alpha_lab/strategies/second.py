from alpha_lab.backtest.bot import BacktestBotTemplate, PrecomputedData


class BacktestBotTemplate:
    def __init__(self):
        self.pred_full = None

        self.ttl = 0

    def _precompute_signals(self, forex_data: ForexData):
        # Two separate models
        # 1. Tactical (low level, low TF)
        X = get_features(forex_data.ohlcv)
        low_tf_clf = joblib.load(Path("alpha_lab", "models", "artifacts", "logreg_v1.pkl"))
        low_tf_raw = pd.Series(low_tf_clf.predict(X), index=X.index)
        low_tf = low_tf_raw.reindex(forex_data.ohlcv.index)

        # 2. Strategic (high level, high TF)
        ma_short = ta.ema(forex_data.ohlcv["close"], GATE_MA_SHORT_PERIOD)
        ma_long = ta.ema(forex_data.ohlcv["close"], GATE_MA_LONG_PERIOD)
        high_tf = (ma_short > ma_long)

        # Final combined signal, Only valid where low_tf exists AND gate is True
        signals = (low_tf == 1) & (high_tf == True)
        # signals = low_tf
        assert signals.index.to_list() == forex_data.ohlcv.index.to_list()
        return signals.to_numpy()

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        """
        Compute time-index aligned signals
        - Vectorized for most things, hopefully with no lookahead bias
        """
        data = PrecomputedData(forex_data)
        data.signals = self._precompute_signals(forex_data)  # Signals
        data.misc["vol"] = forex_data.ohlcv.ta.atr(ATR_PERIOD).to_numpy()  # Additional data, for SL
        return data
    
    def update_trailing_stop(self, position, close: float, vol: float):
        new_sl = close - SL_VOL_MUL * vol
        if new_sl > position.sl:
            position.set_sl(close, new_sl)

    def act(self, idx: int, data: PrecomputedData, acc: Account):
        limit = acc.get_limit()
        position = acc.get_position()
        close = data.prices.close[idx]
        vol = data.misc["vol"][idx]

        # if acc.have_position():
            # self.update_trailing_stop(position, close, vol)
        # else:
        #     if data.signals[idx] and not np.isnan(vol):
        #         sl = self.calculate_sl(close, vol)
        #         acc.open_position(idx, close, sl)

        if limit:
            if self.ttl > 0:
                self.ttl -= 1
            if self.ttl == 0:
                acc.close_limit(idx)
        elif position:
            self.update_trailing_stop(position, close, vol)
            pass
        else:
            if not np.isnan(vol):
                acc.open_limit(idx, close - 2 * vol, close - 4 * vol)
                self.ttl = 10  # limit is valid for only 10 bars

