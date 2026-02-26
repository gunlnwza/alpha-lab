import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from alpha_lab.utils import ForexData
from alpha_lab.backtest.bot import BacktestBot
from alpha_lab.backtest.account import Account


class SimulationResult:
    def __init__(self, forex_data: ForexData, acc: Account, bot: BacktestBot):
        self.forex_data = forex_data
        self.acc = acc
        self.bot = bot

        self.index = self.forex_data.ohlcv.index
        self.closed_positions = self.acc.order_manager.closed_positions

        self.tens = pow(10, self.forex_data.decimal_places)
        self.balance_points = pd.Series(self.acc.balance, index=self.index) * self.tens
        self.equity_points = pd.Series(self.acc.equity, index=self.index) * self.tens

    def report(self):
        # Trades
        win = sum(1 for o in self.closed_positions if o.pnl > 0)
        loss = sum(1 for o in self.closed_positions if o.pnl < 0)
        trades = win + loss
        win_rate = f"{win / trades:.2f}" if trades > 0 else "inf"

        # PnL
        pos_point = sum(p.pnl for p in self.closed_positions if p.pnl > 0) * self.tens
        neg_point = sum(-p.pnl for p in self.closed_positions if p.pnl < 0) * self.tens
        profit_factor = f"{pos_point / neg_point:.2f}" if neg_point != 0 else "inf"
        avg_pos_point = f"{pos_point / win:.2f}" if win > 0 else "inf"
        avg_neg_point = f"{-neg_point / loss:.2f}" if loss > 0 else "inf"

        # Drawdown
        max_equity_drawdown = (self.equity_points.cummax() - self.equity_points).max()
        max_balance_drawdown = (self.balance_points.cummax() - self.balance_points).max()

        print(f"{self.forex_data}")
        print(f"Win | Loss | Trades | Win Rate                : {win:.0f} | {loss:.0f} | {trades:.0f} | {win_rate}")
        print(f"+Point | -Point | Total Point | Profit Factor : {pos_point:.0f} | {-neg_point:.0f} | {pos_point - neg_point:.0f} | {profit_factor}")
        print(f"Average +Point | Average -Point               : {avg_pos_point} | {avg_neg_point}")
        print(f"Max Equity Drawdown | Max Balance Drawdown    : {-max_equity_drawdown:.0f} | {-max_balance_drawdown:.0f}")
        print()

    def visualize(self):
        fig, axes = plt.subplots(2, 1, height_ratios=[2, 1], sharex=True, figsize=(12, 6))

        # axes[0]
        axes[0].set_title(f"Simulation Result | {self.forex_data}")
        axes[0].set_ylabel("Price")
        axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter(f'%.{self.forex_data.decimal_places}f'))
        axes[0].grid(alpha=0.3)

        axes[0].plot(self.forex_data.ohlcv["close"], label=self.forex_data.symbol, linewidth=1)
        for order in self.closed_positions:
            c = "green" if order.pnl >= 0 else "red"
            axes[0].plot(
                [self.index[order.entry_idx], self.index[order.exit_idx]],
                [order.entry_price, order.exit_price],
                color=c, lw=5
            )
        axes[0].legend()

        # axes[1]
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Point Diff")
        axes[1].grid(alpha=0.3)

        axes[1].plot(self.balance_points, label="Balance (point)")
        axes[1].plot(self.equity_points, label="Equity (point)")
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
        acc = self.acc

        for i in range(len(data.prices)):
            if i == len(data.prices) - 1:  # latest bar, no meaning asking bot what to do
                if acc.have_position():
                    acc.close_position(i, prices.close[i])
            else:
                if acc.have_limit():
                    assert not acc.have_position()
                    acc._check_limit(i, prices.high[i], prices.low[i])
                elif acc.have_position():
                    assert not acc.have_limit()
                    acc._check_sl(i, prices.high[i], prices.low[i])
                    if acc.have_position():
                        acc._check_tp(i, prices.high[i], prices.low[i])

                self.bot.act(i, data, self.acc)

            self.acc._update_money(i, prices.close[i])

        self.result = SimulationResult(prices, self.acc, self.bot)
