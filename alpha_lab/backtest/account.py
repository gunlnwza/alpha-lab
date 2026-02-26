class BuyOrder:
    def __init__(self, idx: int, close: float, sl: float):
        self.entry_idx = idx
        self.entry_price = close

        self.exit_idx = None
        self.exit_price = None

        self.sl = sl

        self.pnl = None

    def __repr__(self):
        return f"BuyOrder({self.entry_idx}, {self.entry_price}, {self.sl})"

    def _close(self, idx: int, close: float):
        """Close the order"""
        self.pnl = close - self.entry_price
        self.exit_idx = idx
        self.exit_price = close


class OrderManager:
    def __init__(self):
        self.closed_orders = []
        self.order: BuyOrder | None = None

    def _open_order(self, idx: int, close: float, sl: float):
        self.order = BuyOrder(idx, close, sl)

    def _close_order(self, idx: int, close: float) -> float:
        self.order._close(idx, close)
        self.closed_orders.append(self.order)

        pnl = self.order.pnl
        self.order = None
        return pnl

    def _unrealized_pnl(self, close: float) -> float:
        if not self.order:
            return 0.0
        return close - self.order.entry_price


class Account:
    def __init__(self):
        self.order_manager = OrderManager()
        self.equity = []
        self.balance = []

        self.cumu_balance = 0

    def have_order(self):
        return self.order_manager.order is not None
    
    def get_order(self):
        return self.order_manager.order

    def open_order(self, idx: int, close: float, sl: float):
        self.order_manager._open_order(idx, close, sl)

    def close_order(self, idx: int, close: float):
        pnl = self.order_manager._close_order(idx, close)
        self.cumu_balance += pnl

    def update_money(self, close: float):        
        self.balance.append(self.cumu_balance)

        pnl = float(self.order_manager._unrealized_pnl(close))
        self.equity.append(self.cumu_balance + pnl)
