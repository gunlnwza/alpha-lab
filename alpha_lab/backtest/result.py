import pandas as pd

from rich.console import Console
from rich.table import Table
from rich import box
from rich.columns import Columns

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from alpha_lab.backtest.data import PrecomputedData
from alpha_lab.backtest.account import Account
from alpha_lab.backtest.bot import BacktestBot

console = Console()


class SimulationResult:
    def __init__(self, data: PrecomputedData, acc: Account, bot: BacktestBot):
        self.data = data
        self.acc = acc
        self.bot = bot

        self.forex_data = data.forex_data
        self.index = self.forex_data.ohlcv.index
        self.closed_positions = self.acc.engine.closed_positions

        self.tens = pow(10, self.forex_data.decimal_places)
        self.balance_points = pd.Series(self.acc.balance, index=self.index) * self.tens
        self.equity_points = pd.Series(self.acc.equity, index=self.index) * self.tens

        self.metrics = self._compute_metrics()

    def _compute_metrics(self):
        # Trades
        win = sum(1 for o in self.closed_positions if o.pnl > 0)
        loss = sum(1 for o in self.closed_positions if o.pnl < 0)
        trades = win + loss
        win_rate = f"{win / trades:.2f}" if trades > 0 else "inf"

        # PnL
        pos_point = round(sum(p.pnl for p in self.closed_positions if p.pnl > 0) * self.tens)
        neg_point = round(sum(-p.pnl for p in self.closed_positions if p.pnl < 0) * self.tens)
        total_point = pos_point - neg_point

        avg_pos_point = f"{pos_point / win:.0f}" if win > 0 else "inf"
        avg_neg_point = f"{-neg_point / loss:.0f}" if loss > 0 else "inf"
        profit_factor = f"{pos_point / neg_point:.2f}" if neg_point != 0 else "inf"

        # Drawdown
        max_balance_dd = round(-(self.balance_points.cummax() - self.balance_points).max())
        max_equity_dd = round(-(self.equity_points.cummax() - self.equity_points).max())

        return {
            "Win": str(win),
            "Loss": str(loss),
            "Trades": str(trades),
            "Win Rate": str(win_rate),
            "+Point": str(pos_point),
            "-Point": str(-neg_point),
            "Total Point": str(total_point),
            "Profit Factor": str(profit_factor),
            "Average +Point": str(avg_pos_point),
            "Average -Point": str(avg_neg_point),
            "Max Balance DD": str(max_balance_dd),
            "Max Equity DD": str(max_equity_dd),
        }

    def _print_table(self):
        def init_table(title) -> Table:
            t = Table(title=title, box=box.SIMPLE_HEAVY)
            t.add_column("Metric", justify="left")
            t.add_column("Value", justify="right")
            return t

        def add_rows(table: Table, rows: list[str]):
            for row in rows:
                table.add_row(row, self.metrics[row])

        t1 = init_table("Trades")
        add_rows(t1, ["Win", "Loss", "Trades", "Win Rate"])

        t2 = init_table("Points")
        add_rows(t2, ["+Point", "-Point", "Total Point", "Max Balance DD", "Max Equity DD"])

        t3 = init_table("Averages")
        add_rows(t3, ["Average +Point", "Average -Point", "Profit Factor"])

        console.print(Columns([t1, t2, t3]))

    def report(self):
        self._print_table()

    def visualize(self):
        fig, axes = plt.subplots(2, 1, height_ratios=[2, 1], sharex=True, figsize=(12, 6))

        # axes[0]
        axes[0].set_title(f"Simulation Result | {self.forex_data} | {self.bot.name}")
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

        # Render
        plt.tight_layout()
        plt.show()
