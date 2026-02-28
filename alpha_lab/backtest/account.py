from alpha_lab.backtest.data import Bar, PrecomputedData
from alpha_lab.backtest.order import Side, OrderType, Order, Limit, Position


class OrderEngine:
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

    def open_limit(self, side: Side, idx: int, entry_price: float, sl: float | None, tp: float | None):
        self._assert_no_order()
        self.order = Limit(side, idx, entry_price, sl, tp)

    def open_position(self, side: Side, idx: int, entry_price: float, sl: float | None, tp: float | None):
        self._assert_no_order()
        self.order = Position(side, idx, entry_price, sl, tp)

    def _append_closed_order(self, order: Order):
        if not order.is_closed():
            raise ValueError("Order is still open")

        if isinstance(order, Limit):
            self.closed_limits.append(order)
        elif isinstance(order, Position):
            self.closed_positions.append(order)
        else:
            raise ValueError("Unknown order type")

    def close_order(self, bar: Bar) -> float:
        """Return PnL"""
        self._assert_have_order()

        order = self.order
        self.order = None

        pnl = order.close(bar.idx, bar.close)
        self._append_closed_order(order)
        return pnl

    def unrealized_pnl(self, close: float) -> float:
        if self.order is None:
            return 0.0
        return self.order.unrealized_pnl(close)
    
    def process_bar(self, bar: Bar) -> float:
        """Return realized PnL for this bar"""
        if self.order is None:
            return 0.0

        current_order = self.order
        pnl, new_order = current_order.on_bar(bar)

        # If order closed during this bar, archive it exactly once
        if current_order.is_closed():
            self._append_closed_order(current_order)
            self.order = new_order  # may be None or a new Position

            # in case of a new Position, process again: edge case where price go down so much, buy limit and sl both trigger
            if self.order:
                pnl = self.process_bar(bar)  # this pnl is fresh, and actually from a Position

        return pnl


class Account:
    def __init__(self, data: PrecomputedData):
        self._data = data

        self.engine = OrderEngine()

        self.equity = []
        self.balance = []
        self.cumu_balance = 0

    def have_order(self) -> bool:
        return self.engine.have_order()
    
    def get_order(self) -> Limit | Position | None:
        return self.engine.get_order()

    def open_limit(
            self,
            side: Side,
            entry_price: float,
            sl: float | None = None,
            tp: float | None = None
        ):
        bar = self._data.bar
        self.engine.open_limit(side, bar.idx, entry_price, sl, tp)

    def open_position(
            self,
            side: Side,
            sl: float | None = None,
            tp: float | None = None
        ):
        bar = self._data.bar
        self.engine.open_position(side, bar.idx, bar.close, sl, tp)

    def set_sl(self, sl: float | None):
        self.engine.order.set_sl(sl, self._data.bar)
    
    def set_tp(self, tp: float | None):
        self.engine.order.set_sl(tp, self._data.bar)

    def close_order(self):
        pnl = self.engine.close_order(self._data.bar)
        self.cumu_balance += pnl

    def _process_bar(self):  # for Simulation to call
        pnl = self.engine.process_bar(self._data.bar)
        self.cumu_balance += pnl

    def _update_money(self):  # for Simulation to call
        self.balance.append(self.cumu_balance)

        bar = self._data.bar
        pnl = float(self.engine.unrealized_pnl(bar.close))
        self.equity.append(self.cumu_balance + pnl)
