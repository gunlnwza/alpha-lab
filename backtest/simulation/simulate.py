import numpy as np

from account import Account, calculate_sl
from trade_signals import get_signals
from backtest.simulation.data import SimulationData
from backtest.simulation.result import SimulationResult

from utils import ForexData


def simulate(forex_data: ForexData) -> SimulationResult:
    acc = Account()
    data = SimulationData(forex_data)

    pred_full = get_signals(data)
    assert len(pred_full) == len(data._ohlcv)
    assert pred_full.index.to_list() == data._ohlcv.index.to_list()

    for i in range(len(data._ohlcv)):
        pred = pred_full.iloc[i]
        # if no order, wait for signal
        if not acc.order:
            if not np.isnan(pred) and pred == 1 and not np.isnan(data.vol[i]):
                acc.open_order(i, data.close[i], data.vol[i])
                # print(i, "open at", data.close[i], data._ohlcv.iloc[i].to_list())
            else:
                # print(i, "warm up", data._ohlcv.iloc[i].to_list())
                pass
        else:
            # check if stop loss
            if acc.order and data.low[i] < acc.order.sl:  # if low hooked sl, close
                # print(i, "close at", acc.order.sl, data._ohlcv.iloc[i].to_list())
                acc.close_order(i, acc.order.sl)

            # adjust stop loss if still have order
            if acc.order and not np.isnan(data.vol[i]):
                new_sl = calculate_sl(data.high[i], data.vol[i])  # update with high
                acc.order.sl = max(acc.order.sl, new_sl)
                # print(i, "adjust sl to", acc.order.sl, data._ohlcv.iloc[i].to_list())

    if acc.order:
        acc.close_order(len(data.close) - 1, data.close[-1])

    res = SimulationResult(data, acc)
    return res
