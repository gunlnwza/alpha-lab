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

    def open_limit(self, side: Side, idx: int, entry_price: float, sl: float, tp: float):
        self._assert_no_order()
        self.order = Limit(side, idx, entry_price, sl, tp)

    def open_position(self, side: Side, idx: int, entry_price: float, sl: float, tp: float):
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
        self.engine = OrderEngine()

        self.equity = []
        self.balance = []
        self.cumu_balance = 0

    def have_order(self) -> bool:
        return self.engine.have_order()
    
    def get_order(self) -> Limit | Position | None:
        return self.engine.get_order()

    def open_order(
        self,
        side: Side,
        order_type: OrderType,
        idx: int,
        entry_price: float,
        sl: float | None = None,
        tp: float | None = None
    ):  # TODO: group idx..tp as struct?
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
