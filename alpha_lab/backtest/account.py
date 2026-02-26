from abc import ABC, abstractmethod


class Position(ABC):
    def __init__(self, idx: int, close: float, sl: float):
        self.entry_idx = idx
        self.entry_price = close
        self.sl = sl

        self.exit_idx = None
        self.exit_price = None

        self.pnl = None  # realized pnl

        self._assert_sl_tp_consistent(close)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.entry_idx}, {self.entry_price}, {self.sl})"

    @abstractmethod
    def _pnl(self, close: float) -> float:
        """Calculate unrealized PnL"""

    @abstractmethod
    def _assert_sl_tp_consistent(self, close: float):
        """Check that sl, tp, and close are consistent"""

    def _assert_is_open(self):
        if self.pnl is not None:
            raise ValueError("Position is already closed")

    def set_sl(self, close: float, sl: float):
        self._assert_is_open()
        self.sl = sl
        self._assert_sl_tp_consistent(close)

    def _close(self, idx: int, close: float) -> float:
        """Close the order"""
        self._assert_is_open()
        self.pnl = self._pnl(close)

        self.exit_idx = idx
        self.exit_price = close


class BuyPosition(Position):
    def __init__(self, idx: int, close: float, sl: float):
        super().__init__(idx, close, sl)

    def _assert_sl_tp_consistent(self, close: float):
        if self.sl > close:
            raise ValueError("Invalid SL")

    def _pnl(self, close: float) -> float:
        return close - self.entry_price


class Limit(ABC):
    def __init__(self, idx: int, entry_price: float, entry_sl: float):
        self.idx = idx
        self.entry_price = entry_price
        self.entry_sl = entry_sl

    def __repr__(self):
        return f"{self.__class__.__name__}({self.idx}, {self.entry_price}, {self.entry_sl})"


class BuyLimit(Limit):
    def __init__(self, idx, entry_price, entry_sl):
        super().__init__(idx, entry_price, entry_sl)


class OrderManager:
    def __init__(self):
        self.limit: Limit | None = None

        self.closed_positions = []
        self.position: Position | None = None

    def _open_limit(self, idx: int, entry_price: float, entry_sl: float):
        if self.limit is not None:
            raise RuntimeError("Cannot open limit: limit already exists")
        if self.position is not None:
            raise RuntimeError("Cannot open limit: position already exists")
        self.limit = BuyLimit(idx, entry_price, entry_sl)

    def _close_limit(self, idx: int):
        if self.limit is None:
            raise RuntimeError("No open limit to close")
        self.limit = None

    def _open_position(self, idx: int, close: float, sl: float):
        if self.position is not None:
            raise RuntimeError("Cannot open position: position already exists")
        self.position = BuyPosition(idx, close, sl)

    def _close_position(self, idx: int, close: float) -> float:
        if self.position is None:
            raise RuntimeError("No open position to close")
        self.position._close(idx, close)
        self.closed_positions.append(self.position)

        pnl = self.position.pnl
        self.position = None
        return pnl

    def _unrealized_pnl(self, close: float) -> float:
        if self.position is None:
            return 0.0
        return self.position._pnl(close)


class Account:
    def __init__(self):
        self.order_manager = OrderManager()

        self.equity = []
        self.balance = []
        self.cumu_balance = 0

    # ---
    # Public interfaces

    # Position
    def have_position(self):
        return self.order_manager.position is not None

    def get_position(self):
        return self.order_manager.position
    
    def open_position(self, idx: int, close: float, sl: float):
        self.order_manager._open_position(idx, close, sl)

    def close_position(self, idx: int, close: float):
        pnl = self.order_manager._close_position(idx, close)
        self.cumu_balance += pnl

    # Limit
    def have_limit(self):
        return self.order_manager.limit is not None

    def get_limit(self):
        return self.order_manager.limit

    def open_limit(self, idx: int, entry_price: float, entry_sl: float):
        self.order_manager._open_limit(idx, entry_price, entry_sl)

    def close_limit(self, idx: int):
        self.order_manager._close_limit(idx)

    # ---
    # Simulation interfaces
    def _check_stop_loss(self, idx: int, high: float, low: float, close: float):
        """Assume `position` exist. Close position if low hook sl."""
        position = self.get_position()
        if low < position.sl:
            self.close_position(idx, position.sl)

    def _check_limit(self, idx: int, high: float, low: float, close: float):
        """Assume `limit` exist. Send market order if the low hook limit."""
        limit = self.get_limit()
        if low < limit.entry_price:
            self.open_position(idx, limit.entry_price, limit.entry_sl)
            self.close_limit(idx)

    def _update_money(self, idx: int, close: float):        
        self.balance.append(self.cumu_balance)

        pnl = float(self.order_manager._unrealized_pnl(close))
        self.equity.append(self.cumu_balance + pnl)
