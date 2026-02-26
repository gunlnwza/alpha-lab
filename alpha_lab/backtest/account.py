from abc import ABC, abstractmethod


class Position(ABC):
    def __init__(self, idx: int, close: float, sl: float):
        self.entry_idx = idx
        self.entry_price = close
        self.sl = sl

        self.exit_idx = None
        self.exit_price = None

        self.pnl = None  # realized pnl

    def __repr__(self):
        return f"{self.__class__.__name__}({self.entry_idx}, {self.entry_price}, {self.sl})"

    @abstractmethod
    def _pnl(self, close: float) -> float:
        """Calculate unrealized PnL"""

    def _close(self, idx: int, close: float) -> float:
        """Close the order"""
        if self.pnl is not None:
            raise ValueError("Position is already closed")

        self.pnl = self._pnl(close)

        self.exit_idx = idx
        self.exit_price = close


class BuyPosition(Position):
    def __init__(self, idx: int, close: float, sl: float):
        super().__init__(idx, close, sl)

    def _pnl(self, close: float) -> float:
        return close - self.entry_price


class Limit(ABC):
    def __init__(self, idx: int, entry_price: float, entry_sl: float):
        self.placed_idx = idx
        self.entry_price = entry_price
        self.entry_sl = entry_sl

    @abstractmethod
    def try_execute(self, idx: int, high: float, low: float):
        pass


class BuyLimit(Limit):
    def __init__(self, idx, entry_price, entry_sl):
        super().__init__(idx, entry_price, entry_sl)

    def try_execute(self, idx: int, high: float, low: float) -> BuyPosition | None:
        if low <= self.entry_price:
            return BuyPosition(idx, self.entry_price, self.sl)
        return None


class OrderManager:
    def __init__(self):
        self.closed_orders = []
        self.limit: Limit = None
        self.order: Position | None = None

    def _open_order(self, idx: int, close: float, sl: float):
        self.order = BuyPosition(idx, close, sl)

    def _close_order(self, idx: int, close: float) -> float:
        self.order._close(idx, close)
        self.closed_orders.append(self.order)

        pnl = self.order.pnl
        self.order = None
        return pnl

    def _unrealized_pnl(self, close: float) -> float:
        if not isinstance(self.order, Position):
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

    def update_order(self, idx: int, high: float, low: float, close: float):
        if self.have_order():
            order = self.order_manager.order
            if low < order.sl:
                self.close_order(idx, order.sl)

    def update_money(self, idx: int, close: float):        
        self.balance.append(self.cumu_balance)

        pnl = float(self.order_manager._unrealized_pnl(close))
        self.equity.append(self.cumu_balance + pnl)
