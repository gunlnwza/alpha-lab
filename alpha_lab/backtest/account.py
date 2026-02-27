from enum import Enum


class Side(Enum):
    BUY = 1
    SELL = -1


class Position:
    def __init__(
        self,
        side: Side,
        idx: int,
        entry_price: float,
        sl: float,
        tp: float | None = None,
    ):
        self.side = side
        self.entry_idx = idx
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp

        self.exit_idx: int | None = None
        self.exit_price: float | None = None
        self.pnl: float | None = None

        self._assert_sl_tp_consistent(entry_price)

    def __repr__(self):
        return (
            f"Position(side={self.side.name}, idx={self.entry_idx}, "
            f"entry={self.entry_price}, sl={self.sl}, tp={self.tp})"
        )

    # ---
    # Invariants

    def _assert_sl_tp_consistent(self, price: float):
        if self.side == Side.BUY:
            if self.sl >= price:
                raise ValueError("Invalid SL for BUY")
            if self.tp is not None and self.tp <= price:
                raise ValueError("Invalid TP for BUY")
        else:
            if self.sl <= price:
                raise ValueError("Invalid SL for SELL")
            if self.tp is not None and self.tp >= price:
                raise ValueError("Invalid TP for SELL")

    def _assert_is_open(self):
        if self.pnl is not None:
            raise ValueError("Position already closed")

    # ---
    # Money

    def unrealized_pnl(self, price: float) -> float:
        direction = self.side.value
        return direction * (price - self.entry_price)

    def close(self, idx: int, price: float) -> float:
        self._assert_is_open()
        pnl = self.unrealized_pnl(price)
        self.pnl = pnl
        self.exit_idx = idx
        self.exit_price = price
        return pnl

    # ---
    # Modify

    def set_sl(self, price: float, sl: float):
        self._assert_is_open()
        self.sl = sl
        self._assert_sl_tp_consistent(price)

    def set_tp(self, price: float, tp: float):
        self._assert_is_open()
        self.tp = tp
        self._assert_sl_tp_consistent(price)


class Limit:
    def __init__(
        self,
        side: Side,
        idx: int,
        entry_price: float,
        sl: float,
        tp: float | None = None,
    ):
        self.side = side
        self.idx = idx
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp

    def __repr__(self):
        return (
            f"Limit(side={self.side.name}, idx={self.idx}, "
            f"entry={self.entry_price}, sl={self.sl}, tp={self.tp})"
        )


class OrderManager:
    def __init__(self):
        self.limit: Limit | None = None

        self.closed_positions = []
        self.position: Position | None = None

    def _open_limit(self, idx: int, entry_price: float, entry_sl: float, entry_tp: float | None = None):
        if self.limit is not None:
            raise RuntimeError("Cannot open limit: limit already exists")
        if self.position is not None:
            raise RuntimeError("Cannot open limit: position already exists")
        self.limit = BuyLimit(idx, entry_price, entry_sl, entry_tp)

    def _close_limit(self, idx: int):
        if self.limit is None:
            raise RuntimeError("No open limit to close")
        self.limit = None

    def _open_position(self, idx: int, close: float, sl: float, tp: float | None = None):
        if self.position is not None:
            raise RuntimeError("Cannot open position: position already exists")
        self.position = BuyPosition(idx, close, sl, tp)

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
    
    def open_position(self, idx: int, close: float, sl: float, tp: float | None = None):
        self.order_manager._open_position(idx, close, sl, tp)

    def close_position(self, idx: int, close: float):
        pnl = self.order_manager._close_position(idx, close)
        self.cumu_balance += pnl

    # Limit
    def have_limit(self):
        return self.order_manager.limit is not None

    def get_limit(self):
        return self.order_manager.limit

    def open_limit(self, idx: int, entry_price: float, entry_sl: float, entry_tp: float | None = None):
        self.order_manager._open_limit(idx, entry_price, entry_sl, entry_tp)

    def close_limit(self, idx: int):
        self.order_manager._close_limit(idx)

    # ---
    # Simulation interfaces
    def _check_sl(self, idx: int, high: float, low: float):
        """Assume `position` exist. Close position if low hook sl."""
        position = self.get_position()
        if low < position.sl:
            self.close_position(idx, position.sl)

    def _check_tp(self, idx: int, high: float, low: float):
        """Assume `position` exist. Close position if high hook tp."""
        position = self.get_position()
        if position.tp and high > position.tp:
            self.close_position(idx, position.tp)

    def _check_limit(self, idx: int, high: float, low: float):
        """Assume `limit` exist. Send market order if the low hook limit."""
        limit = self.get_limit()
        if low < limit.entry_price:
            self.open_position(idx, limit.entry_price, limit.entry_sl, limit.entry_tp)
            self.close_limit(idx)

    def _update_money(self, idx: int, close: float):        
        self.balance.append(self.cumu_balance)

        pnl = float(self.order_manager._unrealized_pnl(close))
        self.equity.append(self.cumu_balance + pnl)
