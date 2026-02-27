import pandas as pd

from rich.console import Console
from rich.table import Table
from rich import box

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from alpha_lab.utils import ForexData
from alpha_lab.backtest.data import PrecomputedData
from alpha_lab.backtest.account import Account
from alpha_lab.backtest.bot import BacktestBot

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


class SimulationResult:
    def __init__(self, data: PrecomputedData, acc: Account, bot: BacktestBot):
        self.data = data
        self.acc = acc
        self.bot = bot

        self.index = self.data._forex_data.ohlcv.index
        self.closed_positions = self.acc.engine.closed_positions

        self.tens = pow(10, self.data._forex_data.decimal_places)
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
        total_point = pos_point - neg_point

        avg_pos_point = f"{pos_point / win:.0f}" if win > 0 else "inf"
        avg_neg_point = f"{-neg_point / loss:.0f}" if loss > 0 else "inf"
        profit_factor = f"{pos_point / neg_point:.2f}" if neg_point != 0 else "inf"

        # Drawdown
        max_equity_drawdown = (self.equity_points.cummax() - self.equity_points).max()
        max_balance_drawdown = (self.balance_points.cummax() - self.balance_points).max()

        # Print
        table = Table(title=f"{self.data._forex_data}\n{self.bot.name}", box=box.SIMPLE_HEAVY)

        table.add_column("Metric", justify="left"); table.add_column("Value", justify="right")
        table.add_row("Win", f"{win:.0f}")
        table.add_row("Loss", f"{loss:.0f}")
        table.add_row("Trades", f"{trades:.0f}")
        table.add_row("Win Rate", f"{win_rate}")
        table.add_row("", "")
        table.add_row("+Point", f"{pos_point:.0f}")
        table.add_row("-Point", f"{-neg_point:.0f}")
        table.add_row("Total Point", f"{total_point:.0f}")
        table.add_row("Profit Factor", f"{profit_factor}")
        table.add_row("", "")
        table.add_row("Average +Point", f"{avg_pos_point}")
        table.add_row("Average -Point", f"{avg_neg_point}")
        table.add_row("", "")
        table.add_row("Max Balance Drawdown", f"{-max_balance_drawdown:.0f}")
        table.add_row("Max Equity Drawdown", f"{-max_equity_drawdown:.0f}")

        console.print(table)

    def visualize(self):
        fig, axes = plt.subplots(2, 1, height_ratios=[2, 1], sharex=True, figsize=(12, 6))

        # axes[0]
        axes[0].set_title(f"Simulation Result | {self.data._forex_data} | {self.bot.name}")
        axes[0].set_ylabel("Price")
        axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter(f'%.{self.data._forex_data.decimal_places}f'))
        axes[0].grid(alpha=0.3)

        axes[0].plot(self.data._forex_data.ohlcv["close"], label=self.data._forex_data.symbol, linewidth=1)
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
        data = self.bot.precompute_data(self.forex_data)

        while not data.is_last_bar():
            self.acc.process_bar(data.bar)
            self.bot.act(data, self.acc)
            self.acc.update_money(data.bar)
            data.step()

        if self.acc.have_order():
            self.acc.close_order(data.bar)
        self.acc.update_money(data.bar)

        self.result = SimulationResult(data, self.acc, self.bot)
