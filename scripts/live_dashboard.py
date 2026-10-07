#!/usr/bin/env python3
"""
scripts/live_dashboard.py
=========================
Interactive Real-Time Terminal HUD for Forex Live Trading.

Displays:
  • Account KPIs: Equity, Balance, Floating P&L, Active Lots, Win Rate
  • Live Positions: Symbol, Direction, Lots, Entry, Live Price, P&L ($/pips), TP/SL barriers
  • Recent Execution Journal: Last closed trades, fill prices, and realized profits
  • Model Conviction & Signal Monitor: Per-pair status and 5-minute candle countdown

Usage:
  python scripts/live_dashboard.py            # Continuous live terminal HUD
  python scripts/live_dashboard.py --once     # Single snapshot print
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows console encoding for rich
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env", override=False)

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Error: 'rich' library is required. Install via: pip install rich")
    sys.exit(1)

from trading.live_engine import OANDABroker, price_to_pips


PAIRS = ["EURUSD", "USDJPY", "USDCAD"]
PIP_SIZES = {"EURUSD": 0.0001, "USDJPY": 0.01, "USDCAD": 0.0001}


def get_next_bar_countdown(bar_minutes: int = 5) -> str:
    now = datetime.now(timezone.utc)
    total_seconds = now.minute * 60 + now.second
    bar_period_s = bar_minutes * 60
    rem_s = bar_period_s - (total_seconds % bar_period_s)
    mins = rem_s // 60
    secs = rem_s % 60
    return f"{mins:02d}:{secs:02d}"


def read_trade_journals() -> tuple[list[dict], dict]:
    journal_dir = ROOT_DIR / "logs" / "live"
    recent_events = []
    closed_trades = []

    for path in glob.glob(str(journal_dir / "trade_journal_*.jsonl")):
        try:
            pair_name = Path(path).stem.replace("trade_journal_", "").upper()
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                for line in lines[-25:]:
                    try:
                        record = json.loads(line)
                        record["pair"] = pair_name
                        recent_events.append(record)
                        if record.get("event") == "trade_closed":
                            closed_trades.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    total_closed = len(closed_trades)
    winning = [t for t in closed_trades if float(t.get("pnl_usd", 0.0)) > 0]
    total_realized_pnl = sum(float(t.get("pnl_usd", 0.0)) for t in closed_trades)
    win_rate = (len(winning) / total_closed * 100.0) if total_closed > 0 else 0.0

    stats = {
        "total_closed": total_closed,
        "wins": len(winning),
        "losses": total_closed - len(winning),
        "win_rate": win_rate,
        "realized_pnl": total_realized_pnl,
        "recent_closed": closed_trades[-8:],
        "recent_events": recent_events[-10:],
    }
    return recent_events, stats


def build_dashboard(broker: OANDABroker) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="kpi", size=4),
        Layout(name="positions", size=8),
        Layout(name="lower"),
    )
    layout["lower"].split_row(
        Layout(name="closed_trades", ratio=3),
        Layout(name="signals", ratio=2),
    )

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    next_bar_cd = get_next_bar_countdown(5)

    # 1. Header
    acct_id = getattr(broker, "_account_id", "N/A")
    venue = getattr(broker, "venue", "OANDA").upper()
    header_text = Text()
    header_text.append(" [LIVE TRADING HUD] ", style="bold white on blue")
    header_text.append(f"  |  Broker: {venue} Practice ({acct_id})", style="bold cyan")
    header_text.append(f"  |  Time: {now_utc}", style="dim")
    header_text.append("  |  Next Bar: ", style="bold yellow")
    header_text.append(f"{next_bar_cd}", style="bold white on red")

    layout["header"].update(Panel(header_text, style="blue"))

    # 2. Fetch OANDA data
    try:
        acct = broker.get_account() or {}
        equity = float(acct.get("equity", 0.0))
        balance = float(acct.get("balance", 0.0))
    except Exception:
        equity, balance = 0.0, 0.0

    try:
        positions_map = broker.get_positions() or {}
        avg_prices = getattr(broker, "_avg_prices", {})
    except Exception:
        positions_map, avg_prices = {}, {}

    quotes = {}
    for p in PAIRS:
        try:
            bid, ask = broker.get_bid_ask(p)
            quotes[p] = (bid, ask, (bid + ask) * 0.5)
        except Exception:
            quotes[p] = (0.0, 0.0, 0.0)

    _, stats = read_trade_journals()

    # Calculate Floating P&L
    total_floating_pnl = 0.0
    active_lots = 0.0
    open_positions_rows = []

    for p in PAIRS:
        lots = float(positions_map.get(p, 0.0))
        if abs(lots) < 1e-5:
            continue
        active_lots += abs(lots)
        entry_px = float(avg_prices.get(p, 0.0))
        bid, ask, mid = quotes.get(p, (0.0, 0.0, 0.0))

        # Floating P&L calculation
        pip_size = PIP_SIZES.get(p, 0.0001)
        units = lots * 10_000.0
        if lots > 0:
            direction = "BUY"
            cur_price = bid
            pip_pnl = (cur_price - entry_px) / pip_size if entry_px > 0 else 0.0
            if "JPY" in p:
                usd_pnl = (cur_price - entry_px) * units / cur_price if cur_price > 0 else 0.0
            elif "CAD" in p:
                usd_pnl = (cur_price - entry_px) * units / cur_price if cur_price > 0 else 0.0
            else:
                usd_pnl = (cur_price - entry_px) * units
        else:
            direction = "SELL"
            cur_price = ask
            pip_pnl = (entry_px - cur_price) / pip_size if entry_px > 0 else 0.0
            if "JPY" in p:
                usd_pnl = (entry_px - cur_price) * abs(units) / cur_price if cur_price > 0 else 0.0
            elif "CAD" in p:
                usd_pnl = (entry_px - cur_price) * abs(units) / cur_price if cur_price > 0 else 0.0
            else:
                usd_pnl = (entry_px - cur_price) * abs(units)

        total_floating_pnl += usd_pnl
        open_positions_rows.append({
            "pair": p,
            "direction": direction,
            "lots": abs(lots),
            "entry": entry_px,
            "current": cur_price,
            "pips": pip_pnl,
            "usd_pnl": usd_pnl,
        })

    # 2. KPI Panel
    kpi_table = Table.grid(expand=True)
    kpi_table.add_column(ratio=1)
    kpi_table.add_column(ratio=1)
    kpi_table.add_column(ratio=1)
    kpi_table.add_column(ratio=1)
    kpi_table.add_column(ratio=1)

    float_color = "green" if total_floating_pnl >= 0 else "red"
    realized_color = "green" if stats["realized_pnl"] >= 0 else "red"

    kpi_table.add_row(
        Panel(f"[bold white]${equity:,.2f}[/]\n[dim]NAV / Equity[/]", title="Equity", border_style="cyan"),
        Panel(f"[bold white]${balance:,.2f}[/]\n[dim]Cash Balance[/]", title="Balance", border_style="cyan"),
        Panel(f"[bold {float_color}]${total_floating_pnl:+,.2f}[/]\n[dim]{active_lots:.3f} Lots Active[/]", title="Floating P&L", border_style=float_color),
        Panel(f"[bold {realized_color}]${stats['realized_pnl']:+,.2f}[/]\n[dim]{stats['wins']}W / {stats['losses']}L ({stats['win_rate']:.1f}%)[/]", title="Realized P&L", border_style=realized_color),
        Panel(f"[bold yellow]{next_bar_cd}[/]\n[dim]5m Candle Closes[/]", title="Next Bar", border_style="yellow"),
    )
    layout["kpi"].update(kpi_table)

    # 3. Positions Table
    pos_table = Table(expand=True, header_style="bold cyan", border_style="dim")
    pos_table.add_column("Symbol", justify="left")
    pos_table.add_column("Direction", justify="center")
    pos_table.add_column("Size (Lots)", justify="right")
    pos_table.add_column("Entry Price", justify="right")
    pos_table.add_column("Market Price", justify="right")
    pos_table.add_column("Spread", justify="right")
    pos_table.add_column("Floating Pips", justify="right")
    pos_table.add_column("Unrealized P&L ($)", justify="right")

    if not open_positions_rows:
        pos_table.add_row("[dim]No active open positions[/]", "-", "-", "-", "-", "-", "-", "-")
    else:
        for r in open_positions_rows:
            dir_style = "bold green" if r["direction"] == "BUY" else "bold red"
            pnl_style = "green" if r["usd_pnl"] >= 0 else "red"
            bid, ask, _ = quotes.get(r["pair"], (0, 0, 0))
            spread_pips = (ask - bid) / PIP_SIZES.get(r["pair"], 0.0001) if ask > bid else 0.0

            pos_table.add_row(
                f"[bold white]{r['pair']}[/]",
                f"[{dir_style}]{r['direction']}[/]",
                f"{r['lots']:.4f}",
                f"{r['entry']:.5f}" if r["entry"] > 0 else "Market",
                f"{r['current']:.5f}",
                f"{spread_pips:.1f} pips",
                f"[{pnl_style}]{r['pips']:+.1f}[/]",
                f"[{pnl_style}]${r['usd_pnl']:+,.2f}[/]",
            )

    layout["positions"].update(Panel(pos_table, title="Active Positions", border_style="blue"))

    # 4. Recent Closed Trades Table
    closed_table = Table(expand=True, header_style="bold magenta", border_style="dim")
    closed_table.add_column("Pair", justify="left")
    closed_table.add_column("Reason", justify="left")
    closed_table.add_column("Entry", justify="right")
    closed_table.add_column("Exit", justify="right")
    closed_table.add_column("P&L ($)", justify="right")

    recent_closed = list(reversed(stats["recent_closed"]))[:5]
    if not recent_closed:
        closed_table.add_row("[dim]No trades closed yet[/]", "-", "-", "-", "-")
    else:
        for t in recent_closed:
            pnl = float(t.get("pnl_usd", 0.0))
            p_style = "bold green" if pnl >= 0 else "bold red"
            reason = str(t.get("reason", "closed"))
            closed_table.add_row(
                f"[bold white]{t.get('pair', 'FX')}[/]",
                f"[dim]{reason}[/]",
                f"{float(t.get('entry', 0)):.5f}",
                f"{float(t.get('exit', 0)):.5f}",
                f"[{p_style}]${pnl:+,.2f}[/]",
            )
    layout["closed_trades"].update(Panel(closed_table, title="Recent Closed Executions", border_style="magenta"))

    # 5. Model Signals & Radar Table
    sig_table = Table(expand=True, header_style="bold green", border_style="dim")
    sig_table.add_column("Pair", justify="left")
    sig_table.add_column("Live Mid", justify="right")
    sig_table.add_column("Model Status", justify="center")

    for p in PAIRS:
        _, _, mid = quotes.get(p, (0, 0, 0))
        is_open = any(r["pair"] == p for r in open_positions_rows)
        if is_open:
            status_text = "[bold cyan]Active Trade[/]"
        else:
            status_text = "[dim yellow]Scanning Feed[/]"
        sig_table.add_row(
            f"[bold white]{p}[/]",
            f"{mid:.5f}",
            status_text,
        )
    layout["signals"].update(Panel(sig_table, title="Multi-Pair Signal Radar", border_style="green"))

    return layout


def main():
    parser = argparse.ArgumentParser(description="Live Forex Trading HUD")
    parser.add_argument("--once", action="store_true", help="Print single snapshot and exit")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    console = Console()
    broker = OANDABroker()
    if not broker.connect():
        console.print("[bold red]Failed to connect to OANDA Practice Broker. Check .env![/]")
        sys.exit(1)

    if args.once:
        layout = build_dashboard(broker)
        console.print(layout)
        return

    try:
        with Live(build_dashboard(broker), console=console, refresh_per_second=1, screen=True) as live:
            while True:
                time.sleep(args.interval)
                live.update(build_dashboard(broker))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]HUD exited cleanly.[/]")


if __name__ == "__main__":
    main()
