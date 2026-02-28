from alpha_lab.backtest.data import Bar
from alpha_lab.backtest.account import Account, Side, OrderType
from alpha_lab.backtest.bot import BacktestBot, PrecomputedData
from alpha_lab.utils import ForexData


class VbcBot(BacktestBot):
    """
# Volume-Break-Confirm (many-one-one candles)

---
## Buy Side
Context: After strong selling pressure, look for absorption + reversal attempt.

### 1. Large high-volume red candle
- ≥ 600 points range
- Clearly dominant sell pressure
### 2. Green “break” candle forms
- Closes above prior candle body
- Size ≈ 25-50% of the red candle
- Not too large (avoid exhaustion spike)
### 3. Third candle = confirmation
- Dips to test the green break candle (ideally into its body)
- Holds above red candle low
- Shows rejection (wick or strong close)
### 4. Entry
- Buy when third candle confirms strength
- Do not chase if confirm already extended ≥ 200 points
### 5. Risk Management
- SL: Below structure low (below red candle low or confirm wick)
- TP: At least 1.2-1.5R minimum
- Avoid fixed 300/500 — structure-based SL is better
### 6. Filters
- Avoid during strong directional delivery (impulsive continuation)
- Works better ≥ 15m timeframe

---
## Sell Side
Context: After strong buying pressure, look for distribution + reversal attempt.

### 1. Large high-volume green candle
- ≥ 600 points range
- Clearly dominant buy pressure
### 2. Red “break” candle forms
- Closes below prior candle body
- Size ≈ 25-50% of the green candle
- Not oversized (avoid panic spike)
### 3. Third candle = confirmation
- Rallies to test the red break candle (into its body)
- Holds below green candle high
- Shows rejection (upper wick or strong bearish close)
### 4. Entry
- Sell when third candle confirms weakness
- Do not chase if confirm already extended ≥ 200 points
### 5. Risk Management
- SL: Above structure high
- TP: ≥ 1.2-1.5R
- Avoid fixed point system unless volatility normalized
### 6. Filters
- Avoid strong trend continuation phase
- ≥ 15m timeframe preferred
    """

    def __init__(self):
        super().__init__("vbc")

        self.bars: list[Bar] = []
        self.BARS = 3

    def precompute_data(self, forex_data: ForexData) -> PrecomputedData:
        self.tens = pow(10, forex_data.decimal_places)
        self.tick_size = forex_data.tick_size
        return super().precompute_data(forex_data)

    def tick_to_point(self, tick: float):
        return tick * self.tens

    def point_to_tick(self, point: int):
        return point * self.tick_size

    def act(self, data: PrecomputedData, acc: Account):
        bar = data.bar

        self.bars.insert(0, bar)  # MT4 indexing convention, 0 is the current bar, 1 is the previous bar
        if len(self.bars) < self.BARS:
            return
        if len(self.bars) > self.BARS:
            self.bars.pop()

        # Indexing:
        # bars[2] = impulse candle (volume candle)
        # bars[1] = break candle
        # bars[0] = confirm candle
        bar_v = self.bars[2]
        bar_break = self.bars[1]
        bar_confirm = self.bars[0]

        order = acc.get_order()
        if order:
            if order.type == OrderType.LIMIT:
                acc.close_order(bar_confirm)
            return

        body_v = self.tick_to_point(bar_v.close - bar_v.open)
        body_break = self.tick_to_point(bar_break.close - bar_break.open)

        # =========================
        # BUY SIDE
        # =========================
        if body_v <= -600:  # large red impulse

            # break candle must be green and 25-50% of impulse body
            if body_break > 0 and 0.25 <= abs(body_break / body_v) <= 0.5:

                # confirm must dip into break body and hold above impulse low
                if (
                    bar_confirm.low <= bar_break.close
                    and bar_confirm.low > bar_v.low
                ):
                    risk_points = self.tick_to_point(bar_confirm.close - bar_v.low)
                    if risk_points > 0:
                        sl_price = bar_v.low
                        tp_price = bar_confirm.close + self.point_to_tick(int(risk_points * 1.5))

                        acc.open_limit(
                            Side.BUY,
                            bar_confirm,
                            bar_confirm.close,
                            sl_price,
                            tp_price,
                        )
                        return

        # =========================
        # SELL SIDE (reversed logic)
        # =========================
        if body_v >= 600:  # large green impulse

            # break candle must be red and 25-50% of impulse body
            if body_break < 0 and 0.25 <= abs(body_break / body_v) <= 0.5:

                # confirm must rally into break body and hold below impulse high
                if (
                    bar_confirm.high >= bar_break.close
                    and bar_confirm.high < bar_v.high
                ):
                    risk_points = self.tick_to_point(bar_v.high - bar_confirm.close)
                    if risk_points > 0:
                        sl_price = bar_v.high
                        tp_price = bar_confirm.close - self.point_to_tick(int(risk_points * 1.5))

                        acc.open_limit(
                            Side.SELL,
                            bar_confirm,
                            bar_confirm.close,
                            sl_price,
                            tp_price,
                        )
                        return
