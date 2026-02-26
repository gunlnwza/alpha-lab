import matplotlib.pyplot as plt

from backtest.simulation.data import SimulationData
from account import Account


class SimulationResult:
    def __init__(self, data: SimulationData, acc: Account):
        self.data = data
        self.acc = acc

    def report(self):
        win = sum(1 for o in self.acc.closed_orders if o.pnl > 0)
        loss = sum(1 for o in self.acc.closed_orders if o.pnl < 0)
        trades = win + loss
        pos_pnl = sum(o.pnl for o in self.acc.closed_orders if o.pnl > 0)
        neg_pnl = sum(-o.pnl for o in self.acc.closed_orders if o.pnl < 0)

        win_rate = win / trades if trades > 0 else "inf"
        profit_per_loss = pos_pnl / neg_pnl if neg_pnl != 0 else 'inf'

        print("-" * 40)
        print(f"{self.data.forex_data}")
        print(f"    Win | Loss | Trades : {win:.0f} | {loss:.0f} | {trades:.0f}")
        print(f"               Win Rate : {win_rate:.2f}")
        print(f"+PnL | -PnL | Total PnL : {pos_pnl:.2f} | {-neg_pnl:.2f} | {pos_pnl - neg_pnl:.2f}")
        print(f"        Profit per Loss : {profit_per_loss:.2f}")
        print("-" * 40)

    def visualize(self):
        plt.figure(figsize=(14, 6))

        plt.plot(self.data.close, label="Close Price", linewidth=1)
        plt.plot(self.data.ma_short, label="Gate MA Short", linewidth=1)
        plt.plot(self.data.ma_long, label="Gate MA Long", linewidth=1)

        for order in self.acc.closed_orders:
            c = "green" if order.pnl >= 0 else "red"
            plt.plot([order.entry_idx, order.exit_idx], [order.entry_price, order.exit_price], color=c, lw=5)

        plt.title(f"Backtest Visualization | {self.data.forex_data}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
