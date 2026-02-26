from orders import BuyOrder
from config import SL_VOL_MUL

# TODO: smelly
def calculate_sl(close, vol):
    return close - SL_VOL_MUL * vol


class Account:
    def __init__(self):
        self.closed_orders = []
        self.order: BuyOrder | None = None
        self.pnl = 0.0

    def open_order(self, idx: int, close: float, vol: float):
        sl = calculate_sl(close, vol)
        self.order = BuyOrder(idx, close, sl)

    def close_order(self, idx: int, close: float):
        self.order.close(idx, close)
        self.pnl += self.order.pnl
        self.closed_orders.append(self.order)
        self.order = None
