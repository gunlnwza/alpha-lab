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
        sl: float | None = None,
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
        sl: float | None = None,
        tp: float | None = None,
    ):
        super().__init__(side, idx, entry_price, sl, tp)
        self.type = OrderType.LIMIT

    def unrealized_pnl(self, price: float) -> float:
        return super().unrealized_pnl(price)
    
    def on_bar(self, idx: int, high: float, low: float, close: float) -> tuple[float, Optional["Order"]]:
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
        sl: float | None = None,
        tp: float | None = None,
    ):
        super().__init__(side, idx, entry_price, sl, tp)
        self.type = OrderType.POSITION

    def unrealized_pnl(self, price: float) -> float:
        self._assert_is_open()
        direction = self.side.value
        return direction * (price - self.entry_price)

    def on_bar(self, idx: int, high: float, low: float, close: float) -> tuple[float, Optional["Order"]]:
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
