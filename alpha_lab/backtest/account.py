class BuyOrder:
    def __init__(self, idx: int, close: float, sl: float):
        self.entry_idx = idx
        self.entry_price = close

        self.exit_idx = None
        self.exit_price = None

        self.sl = sl

        self.pnl = None

    def close(self, idx: int, close: float):
        """Close the order"""
        self.pnl = close - self.entry_price
        self.exit_idx = idx
        self.exit_price = close

    def sl_hit(self, idx: int):
        """Close the order at stop loss"""
        self.close(idx, self.sl)


class OrderManager:
    def __init__(self):
        self.closed_orders = []
        self.order: BuyOrder | None = None

    def open_order(self, idx: int, close: float, sl: float):
        self.order = BuyOrder(idx, close, sl)

    def close_order(self, idx: int, close: float) -> float:
        self.order.close(idx, close)
        self.closed_orders.append(self.order)

        pnl = self.order.pnl
        self.order = None
        return pnl

    def unrealized_pnl(self, close: float) -> float:
        if not self.order:
            return 0.0
        return close - self.order.entry_price


class Account:
    def __init__(self):
        self.order_manager = OrderManager()
        self.equity = []
        self.balance = 0.0

    def have_order(self):
        return self.order_manager.order is not None

    def open_order(self, idx: int, close: float, sl: float):
        self.order_manager.open_order(idx, close, sl)

    def close_order(self, idx: int, close: float):
        pnl = self.order_manager.close_order(idx, close)
        self.balance += pnl

    def update_equity(self, close: float):
        unrealized = self.order_manager.unrealized_pnl(close)
        self.equity.append(float(self.balance + unrealized))
