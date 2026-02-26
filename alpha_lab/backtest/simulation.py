import pandas as pd
import matplotlib.pyplot as plt

from alpha_lab.utils import ForexData
from alpha_lab.backtest.bot import BacktestBot
from alpha_lab.backtest.account import Account


class SimulationResult:
    def __init__(self, forex_data: ForexData, acc: Account, bot: BacktestBot):
        self.forex_data = forex_data
        self.acc = acc
        self.bot = bot

    def report(self):
        closed_orders = self.acc.order_manager.closed_orders

        # Trades
        win = sum(1 for o in closed_orders if o.pnl > 0)
        loss = sum(1 for o in closed_orders if o.pnl < 0)
        trades = win + loss
        pos_pnl = sum(o.pnl for o in closed_orders if o.pnl > 0)
        neg_pnl = sum(-o.pnl for o in closed_orders if o.pnl < 0)

        # Rate
        win_rate = f"{win / trades:.2f}" if trades > 0 else "inf"
        profit_per_loss = f"{pos_pnl / neg_pnl:.2f}" if neg_pnl != 0 else "inf"

        # Drawdown
        equity = pd.Series(self.acc.equity, index=self.forex_data.ohlcv.index)
        max_drawdown = (equity.cummax() - equity).max()

        print(f"{self.forex_data}")
        print(f"      Win | Loss | Trades : {win:.0f} | {loss:.0f} | {trades:.0f}")
        print(f"                 Win Rate : {win_rate}")
        print(f"  +PnL | -PnL | Total PnL : {pos_pnl:.2f} | {-neg_pnl:.2f} | {pos_pnl - neg_pnl:.2f}")
        print(f"          Profit per Loss : {profit_per_loss}")
        print(f"Max Drawdown (price diff) : {max_drawdown:.2f}")
        print()

    def visualize(self):
        fig, axes = plt.subplots(2, 1, height_ratios=[2, 1], sharex=True, figsize=(12, 6))

        # axes[0]
        axes[0].set_title(f"Simulation Result | {self.forex_data}")
        axes[0].set_ylabel("Price")
        axes[0].grid(alpha=0.3)

        axes[0].plot(self.forex_data.ohlcv.close, label="Close Price", linewidth=1)

        index = self.forex_data.ohlcv.index
        closed_orders = self.acc.order_manager.closed_orders
        for order in closed_orders:
            c = "green" if order.pnl >= 0 else "red"
            axes[0].plot([index[order.entry_idx], index[order.exit_idx]], [order.entry_price, order.exit_price], color=c, lw=5)

        axes[0].legend()

        # axes[1]
        balance = pd.Series(self.acc.balance, index=index)
        equity = pd.Series(self.acc.equity, index=index)

        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Price Diff")
        axes[1].grid(alpha=0.3)

        axes[1].plot(balance, label="Balance")
        axes[1].plot(equity, label="Equity")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


class Simulation:
    def __init__(self, forex_data: ForexData, acc: Account, bot: BacktestBot):
        self.forex_data = forex_data
        self.acc = acc
        self.bot = bot

        self.result = None

    def run(self):
        prices = self.forex_data
        data = self.bot.precompute_data(prices)

        for i in range(len(data.prices)):
            if i == len(data.prices) - 1:  # latest bar, no meaning asking bot what to do
                if self.acc.have_order():
                    self.acc.close_order(i, prices.close[i])
            else:
                self.acc.check_stop_loss(i, prices.high[i], prices.low[i], prices.close[i])
                self.bot.act(i, data, self.acc)

            self.acc.update_money(i, prices.close[i])

        self.result = SimulationResult(prices, self.acc, self.bot)
