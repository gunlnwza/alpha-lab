import numpy as np
import logging

from utils import ForexData

from .account import Account
from .data import SimulationData
from .result import SimulationResult
from model_entrypoint import get_signals

logger = logging.getLogger(__file__)


class Simulation:
    def __init__(self, forex_data: ForexData):
        self.data = SimulationData(forex_data)
        self.acc = Account()

        self.pred_full = None
        self.result = None

    def _handle_no_order(self, i: int):
        data = self.data
        acc = self.acc
        if self.pred_full[i] == 1 and not np.isnan(data.vol[i]):
            logger.info(f"{i} | {data._ohlcv.iloc[i].to_list()} | Open order at {data.close[i]}")
            acc.open_order(i, data.close[i], data.vol[i])

    def _handle_have_order(self, i: int):
        data = self.data
        acc = self.acc
        if data.low[i] < acc.order_manager.order.sl:  # if low hooked sl, close
            logger.info(f"{i} | {data._ohlcv.iloc[i].to_list()} | Close order at {acc.order_manager.order.sl}")
            acc.close_order(i, acc.order_manager.order.sl)
            return

        if not np.isnan(data.vol[i]):
            new_sl = acc.order_manager.calculate_sl(data.high[i], data.vol[i])  # update with high
            if new_sl > acc.order_manager.order.sl:
                logger.debug(f"{i} | {data._ohlcv.iloc[i].to_list()} | Adjust SL to {new_sl}")
                acc.order_manager.order.sl = new_sl

    def run(self):
        self.pred_full = get_signals(self.data).to_numpy()

        data = self.data
        acc = self.acc
        for i in range(len(data._ohlcv)):
            if not self.acc.have_order():
                self._handle_no_order(i)
            else:
                self._handle_have_order(i)
        if acc.have_order():
            acc.close_order(len(data.close) - 1, data.close[-1])

        self.result = SimulationResult(data, acc)
