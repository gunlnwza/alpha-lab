import numpy as np
import logging

from utils import ForexData

from .account import Account
from .data import SimulationData
from .result import SimulationResult

logger = logging.getLogger(__file__)


def simulate(forex_data: ForexData) -> SimulationResult:
    acc = Account()
    data = SimulationData(forex_data)

    from trade_signals import get_signals
    pred_full = get_signals(data)
    assert len(pred_full) == len(data._ohlcv)
    assert pred_full.index.to_list() == data._ohlcv.index.to_list()

    def handle_no_order():
        if pred == 1 and not np.isnan(data.vol[i]):
            logger.info(f"{i} | {data._ohlcv.iloc[i].to_list()} | Open order at {data.close[i]}")
            acc.open_order(i, data.close[i], data.vol[i])

    def handle_have_order():
        if data.low[i] < acc.order_manager.order.sl:  # if low hooked sl, close
            logger.info(f"{i} | {data._ohlcv.iloc[i].to_list()} | Close order at {acc.order_manager.order.sl}")
            acc.close_order(i, acc.order_manager.order.sl)
            return

        if not np.isnan(data.vol[i]):
            new_sl = acc.order_manager.calculate_sl(data.high[i], data.vol[i])  # update with high
            if new_sl > acc.order_manager.order.sl:
                logger.debug(f"{i} | {data._ohlcv.iloc[i].to_list()} | Adjust SL to {new_sl}")
                acc.order_manager.order.sl = new_sl

    for i in range(len(data._ohlcv)):
        pred = pred_full.iloc[i]
        if not acc.have_order():
            handle_no_order()
        else:
            handle_have_order()

    if acc.have_order():
        acc.close_order(len(data.close) - 1, data.close[-1])

    res = SimulationResult(data, acc)
    return res
