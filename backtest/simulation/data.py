import pandas_ta as ta

from config import ATR_PERIOD, GATE_MA_SHORT_PERIOD, GATE_MA_LONG_PERIOD
from utils import ForexData


class SimulationData:
    def __init__(self, forex_data: ForexData):
        self.forex_data = forex_data
        self._ohlcv = forex_data.ohlcv

        self.open = self._ohlcv["open"].to_numpy()
        self.high = self._ohlcv["high"].to_numpy()
        self.low = self._ohlcv["low"].to_numpy()
        self.close = self._ohlcv["close"].to_numpy()

        self.vol = self._ohlcv.ta.atr(ATR_PERIOD).to_numpy()
        self.ma_short = ta.ema(self._ohlcv["close"], GATE_MA_SHORT_PERIOD).to_numpy()
        self.ma_long = ta.ema(self._ohlcv["close"], GATE_MA_LONG_PERIOD).to_numpy()
