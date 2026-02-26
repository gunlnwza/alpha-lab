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
        win_rate = round(win / trades, 2) if trades > 0 else "inf"
        profit_per_loss = round(pos_pnl / neg_pnl, 2) if neg_pnl != 0 else "inf"

        # Drawdown
        equity = pd.Series(self.acc.equity, index=self.forex_data.ohlcv.index)
        cummax = equity.cummax()
        cummax = cummax.replace(0, pd.NA)

        drawdown = (cummax - equity) / cummax
        drawdown = drawdown.fillna(0)

        self.dd = drawdown
        max_drawdown = drawdown.max()


        print(f"{self.forex_data}")
        print(f"    Win | Loss | Trades : {win:.0f} | {loss:.0f} | {trades:.0f}")
        print(f"               Win Rate : {win_rate}")
        print(f"+PnL | -PnL | Total PnL : {pos_pnl:.2f} | {-neg_pnl:.2f} | {pos_pnl - neg_pnl:.2f}")
        print(f"        Profit per Loss : {profit_per_loss}")
        print(f"           Max Drawdown : {max_drawdown:.2f}")
        # print(self.acc.equity)
        print()

    def visualize(self):
        plt.figure(figsize=(14, 6))

        plt.plot(self.forex_data.ohlcv.close, label="Close Price", linewidth=1)
        # plt.plot(self.dd, label="Drawdown", linewidth=1)
        # plt.plot(self.data.ma_short, label="Gate MA Short", linewidth=1)
        # plt.plot(self.data.ma_long, label="Gate MA Long", linewidth=1)

        index = self.forex_data.ohlcv.index
        closed_orders = self.acc.order_manager.closed_orders
        for order in closed_orders:
            c = "green" if order.pnl >= 0 else "red"
            plt.plot([index[order.entry_idx], index[order.exit_idx]], [order.entry_price, order.exit_price], color=c, lw=5)

        plt.title(f"Backtest Visualization | {self.forex_data}")
        plt.legend()
        plt.grid(alpha=0.3)
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
            order = self.acc.order_manager.order
            if order and prices.low[i] < order.sl:
                self.acc.order_manager.close_order(i, order.sl)

            self.bot.act(i, data, self.acc)

            self.acc.update_equity(prices.close[i])

        if self.acc.have_order():
            self.acc.close_order(len(prices) - 1, prices.close[len(prices) - 1])

        self.result = SimulationResult(prices, self.acc, self.bot)
