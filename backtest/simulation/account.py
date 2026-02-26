from config import SL_VOL_MUL


class BuyOrder:
    def __init__(self, idx: int, close: float, sl: float):
        self.entry_idx = idx
        self.entry_price = close

        self.exit_idx = None
        self.exit_price = None

        self.sl = sl

        self.pnl = None

    def close(self, idx: int, close: float):
        self.pnl = close - self.entry_price
        self.exit_idx = idx
        self.exit_price = close

    def sl_hit(self, idx: int):
        self.close(idx, self.sl)


class OrderManager:
    def __init__(self):
        self.closed_orders = []
        self.order: BuyOrder | None = None

    @classmethod
    def calculate_sl(cls, close, vol):
        return close - SL_VOL_MUL * vol

    def open_order(self, idx: int, close: float, vol: float):
        sl = self.calculate_sl(close, vol)
        self.order = BuyOrder(idx, close, sl)

    def close_order(self, idx: int, close: float) -> float:
        self.order.close(idx, close)
        pnl = self.order.pnl

        self.closed_orders.append(self.order)
        self.order = None

        return pnl


class Account:
    def __init__(self):
        self.order_manager = OrderManager()
        self.pnl = 0.0

    def have_order(self):
        return self.order_manager.order is not None

    def open_order(self, idx: int, close: float, vol: float):
        self.order_manager.open_order(idx, close, vol)

    def close_order(self, idx: int, close: float):
        pnl = self.order_manager.close_order(idx, close)
        self.pnl += pnl
