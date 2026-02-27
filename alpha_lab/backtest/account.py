from enum import Enum


class Side(Enum):
    BUY = 1
    SELL = -1


class OrderType(Enum):
    LIMIT = "limit",
    POSITION = "position"


class Order:  # TODO: do proper inheritance
    pass

class Position(Order):
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


class Limit(Order):
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
        self._assert_sl_tp_consistent(entry_price)

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

    def __repr__(self):
        return (
            f"Limit(side={self.side.name}, idx={self.idx}, "
            f"entry={self.entry_price}, sl={self.sl}, tp={self.tp})"
        )


class PositionEngine:
    def __init__(self):
        self.closed_limits = []
        self.limit: Limit | None = None

        self.closed_positions = []
        self.position: Position | None = None
    
    def have_order(self):
        return self.limit or self.position

    def get_order(self):
        if self.limit:
            assert self.position is None
            return self.limit
        elif self.position:
            assert self.limit is None
            return self.position
        return None

    def open_limit(self, side: Side, idx: int, entry_price: float, sl: float, tp: float):
        if self.have_order():
            raise ValueError("Already have order, cannot open new limit")
        self.limit = Limit(side, idx, entry_price, sl, tp)

    def open_position(self, side: Side, idx: int, entry_price: float, sl: float, tp: float):
        if self.have_order():
            raise ValueError("Already have order, cannot open new position")
        self.position = Position(side, idx, entry_price, sl, tp)

    def close_order(self, idx: int, close: float) -> float:
        """Return PnL"""
        order = self.get_order()
        if order is None:
            raise ValueError("No open order to close")

        if isinstance(order, Limit):
            pnl = 0.0
            self.closed_limits.append(self.limit)
            self.limit = None
        elif isinstance(order, Position):
            pnl = self.position.close(idx, close)
            self.closed_positions.append(self.position)
            self.position = None
        else:
            raise ValueError("Unknown order type")

        return pnl

    def unrealized_pnl(self, close: float) -> float:
        if self.position is None:
            return 0.0
        return self.position.unrealized_pnl(close)
    
    def process_bar(self, idx: int, high: float, low: float, close: float) -> float:
        """Return PnL"""
        order = self.get_order()
        if order is None:
            return 0.0

        if isinstance(order, Limit):
            if order.side == Side.BUY:  # TODO: make orders own process bar logic
                if low < order.entry_price:
                    self.close_order(idx, close)
                    self.open_position(Side.BUY, idx, order.entry_price, order.sl, order.tp)
            else:
                if high > order.entry_price:
                    self.close_order(idx, close)
                    self.open_position(Side.SELL, idx, order.entry_price, order.sl, order.tp)
            return 0.0
        elif isinstance(order, Position):
            pnl = 0.0
            if order.side == Side.BUY:
                if low < order.sl:
                    pnl = self.close_order(idx, order.sl)
                elif order.tp and high > order.tp:
                    pnl = self.close_order(idx, order.tp)
            else:
                if high > order.sl:
                    pnl = self.close_order(idx, order.sl)
                elif order.tp and low < order.tp:
                    pnl = self.close_order(idx, order.tp)
            return pnl
        else:
            raise ValueError("Unknown order type")


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
