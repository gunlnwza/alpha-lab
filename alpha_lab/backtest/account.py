from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional


class Side(Enum):
    BUY = 1
    SELL = -1


class OrderType(Enum):
    LIMIT = "limit"
    POSITION = "position"


class Order(ABC):
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
        pnl = f"{self.pnl:.2f}" if self.pnl is not None else None
        return (
            f"{self.__class__.__name__}(side={self.side.name}, idx={self.entry_idx}, "
            f"entry={self.entry_price:.2f}, pnl={pnl})"
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
        if self.is_closed():
            raise ValueError(f"{self.__class__.__name__} already closed")

    # ---
    # Setter

    def set_sl(self, price: float, sl: float):
        self._assert_is_open()
        self.sl = sl
        self._assert_sl_tp_consistent(price)

    def set_tp(self, price: float, tp: float):
        self._assert_is_open()
        self.tp = tp
        self._assert_sl_tp_consistent(price)

    # ---
    # Getter

    def is_open(self):
        return self.pnl is None
    
    def is_closed(self):
        return self.pnl is not None

    # ---
    # Money

    @abstractmethod
    def unrealized_pnl(self, price: float) -> float:
        self._assert_is_open()  # if already closed, use self.pnl field instead
        return 0.0

    def close(self, idx: int, price: float) -> float:
        self._assert_is_open()
        pnl = self.unrealized_pnl(price)
        self.pnl = pnl
        self.exit_idx = idx
        self.exit_price = price  # It's valid to say "I closed a buy limit at this exit time and price, making 0 pnl"
        return pnl

    # ---
    # Update

    @abstractmethod
    def on_bar(self, idx: int, high: float, low: float, close: float) -> tuple[float, Optional["Order"]]:
        pnl = 0.0
        new_order = None
        return pnl, new_order


class Limit(Order):
    def __init__(
        self,
        side: Side,
        idx: int,
        entry_price: float,
        sl: float,
        tp: float | None = None,
    ):
        super().__init__(side, idx, entry_price, sl, tp)

    def unrealized_pnl(self, price: float) -> float:
        return super().unrealized_pnl(price)
    
    def on_bar(self, idx: int, high: float, low: float, close: float):
        """Might close, must check after"""
        new_order = None

        if self.side == Side.BUY:
            if low < self.entry_price:
                self.close(idx, close)
                new_order = Position(Side.BUY, idx, self.entry_price, self.sl, self.tp)
        else:
            if high > self.entry_price:
                self.close(idx, close)
                new_order = Position(Side.SELL, idx, self.entry_price, self.sl, self.tp)

        return 0.0, new_order


class Position(Order):
    def __init__(
        self,
        side: Side,
        idx: int,
        entry_price: float,
        sl: float,
        tp: float | None = None,
    ):
        super().__init__(side, idx, entry_price, sl, tp)

    def unrealized_pnl(self, price: float) -> float:
        self._assert_is_open()
        direction = self.side.value
        return direction * (price - self.entry_price)

    def on_bar(self, idx: int, high: float, low: float, close: float):
        """Might close, must check after"""
        pnl = 0.0

        if self.side == Side.BUY:
            if self.sl and low < self.sl:
                pnl = self.close(idx, self.sl)
            elif self.tp and high > self.tp:
                pnl = self.close(idx, self.tp)
        else:
            if self.sl and high > self.sl:
                pnl = self.close(idx, self.sl)
            elif self.tp and low < self.tp:
                pnl = self.close(idx, self.tp)

        return pnl, None


class PositionEngine:
    def __init__(self):
        self.closed_limits = []
        self.closed_positions = []
        self.order: Order | None = None
    
    def have_order(self):
        return self.order is not None

    def get_order(self):
        return self.order
    
    def _assert_no_order(self):
        if self.have_order():
            raise ValueError("Have open order")
        
    def _assert_have_order(self):
        if not self.have_order():
            raise ValueError("No open order")

    def open_limit(self, side: Side, idx: int, entry_price: float, sl: float, tp: float):
        self._assert_no_order()
        self.order = Limit(side, idx, entry_price, sl, tp)

    def open_position(self, side: Side, idx: int, entry_price: float, sl: float, tp: float):
        self._assert_no_order()
        self.order = Position(side, idx, entry_price, sl, tp)

    def _append_closed_order(self, order: Order):
        assert order.is_closed()

        if isinstance(order, Limit):
            self.closed_limits.append(order)
        elif isinstance(order, Position):
            self.closed_positions.append(order)
        else:
            raise ValueError("Unknown order type")

    def close_order(self, idx: int, close: float) -> float:
        """Return PnL"""
        self._assert_have_order()

        order = self.order
        self.order = None

        pnl = order.close(idx, close)
        self._append_closed_order(order)
        return pnl

    def unrealized_pnl(self, close: float) -> float:
        if self.order is None:
            return 0.0
        return self.order.unrealized_pnl(close)
    
    def process_bar(self, idx: int, high: float, low: float, close: float) -> float:
        """Return realized PnL for this bar"""
        if self.order is None:
            return 0.0

        current_order = self.order
        pnl, new_order = current_order.on_bar(idx, high, low, close)

        # If order closed during this bar, archive it exactly once
        if current_order.is_closed():
            self._append_closed_order(current_order)
            self.order = new_order  # may be None or a new Position

        return pnl


class Account:
    def __init__(self):
        self.engine = PositionEngine()

        self.equity = []
        self.balance = []
        self.cumu_balance = 0

    def have_order(self) -> bool:
        return self.engine.have_order()
    
    def get_order(self) -> Limit | Position | None:
        return self.engine.get_order()

    def open_order(self, side: Side, order_type: OrderType, idx: int, entry_price: float, sl: float, tp: float | None = None):  # TODO: group idx..tp as struct
        if order_type == OrderType.LIMIT:
            self.engine.open_limit(side, idx, entry_price, sl, tp)
        elif order_type == OrderType.POSITION:
            self.engine.open_position(side, idx, entry_price, sl, tp)

    def close_order(self, idx: int, close: int):
        pnl = self.engine.close_order(idx, close)
        self.cumu_balance += pnl

    def process_bar(self, idx: int, high: float, low: float, close: float):
        pnl = self.engine.process_bar(idx, high, low, close)
        self.cumu_balance += pnl

    def update_money(self, idx: int, close: float):  # TODO: might use `idx` with numpy array later
        self.balance.append(self.cumu_balance)

        pnl = float(self.engine.unrealized_pnl(close))
        self.equity.append(self.cumu_balance + pnl)
