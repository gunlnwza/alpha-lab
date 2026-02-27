from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional

from alpha_lab.backtest.data import Bar


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
        sl: float | None,
        tp: float | None
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
            if self.sl and self.sl >= price:
                raise ValueError("Invalid SL for BUY")
            if self.tp is not None and self.tp <= price:
                raise ValueError("Invalid TP for BUY")
        else:
            if self.sl and self.sl <= price:
                raise ValueError("Invalid SL for SELL")
            if self.tp is not None and self.tp >= price:
                raise ValueError("Invalid TP for SELL")

    def _assert_is_open(self):
        if self.is_closed():
            raise ValueError(f"{self.__class__.__name__} already closed")

    # ---
    # Setter

    def set_sl(self, sl: float, bar: Bar):
        """Set SL safely, comparing with `bar.close`"""
        self._assert_is_open()
        self.sl = sl
        self._assert_sl_tp_consistent(bar.close)

    def set_tp(self, tp: float, bar: Bar):
        """Set TP safely, comparing with `bar.close`"""
        self._assert_is_open()
        self.tp = tp
        self._assert_sl_tp_consistent(bar.close)

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

    def close(self, idx: int, price: float) -> float:  # must use this interface to ensure flexible closing at any price
        self._assert_is_open()
        pnl = self.unrealized_pnl(price)
        self.pnl = pnl
        self.exit_idx = idx
        self.exit_price = price  # It's valid to say "I closed a buy limit at this exit time and price, making 0 pnl"
        return pnl

    # ---
    # Update

    @abstractmethod
    def on_bar(self, bar: Bar) -> tuple[float, Optional["Order"]]:  # more convenient and easier to read to use bar
        pnl = 0.0
        new_order = None
        return pnl, new_order


class Limit(Order):
    def __init__(
        self,
        side: Side,
        idx: int,
        entry_price: float,
        sl: float | None,
        tp: float | None
    ):
        super().__init__(side, idx, entry_price, sl, tp)
        self.type = OrderType.LIMIT

    def unrealized_pnl(self, price: float) -> float:
        return super().unrealized_pnl(price)
    
    def on_bar(self, bar: Bar) -> tuple[float, Optional["Order"]]:
        """Might close, must check after"""
        new_order = None

        if (
            (self.side == Side.BUY and bar.low < self.entry_price)
            or (self.side == Side.SELL and bar.high > self.entry_price)
        ):
            self.close(bar.idx, bar.close)
            new_order = Position(self.side, bar.idx, self.entry_price, self.sl, self.tp)

        return 0.0, new_order


class Position(Order):
    def __init__(
        self,
        side: Side,
        idx: int,
        entry_price: float,
        sl: float | None,
        tp: float | None
    ):
        super().__init__(side, idx, entry_price, sl, tp)
        self.type = OrderType.POSITION

    def unrealized_pnl(self, price: float) -> float:
        self._assert_is_open()
        direction = self.side.value
        return direction * (price - self.entry_price)

    def on_bar(self, bar: Bar) -> tuple[float, Optional["Order"]]:
        """Might close, must check after"""
        pnl = 0.0

        if self.side == Side.BUY:
            if self.sl and bar.low < self.sl:
                pnl = self.close(bar.idx, self.sl)
            elif self.tp and bar.high > self.tp:
                pnl = self.close(bar.idx, self.tp)
        else:
            if self.sl and bar.high > self.sl:
                pnl = self.close(bar.idx, self.sl)
            elif self.tp and bar.low < self.tp:
                pnl = self.close(bar.idx, self.tp)

        return pnl, None
