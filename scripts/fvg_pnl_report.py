#!/usr/bin/env python3
"""
fvg_pnl_report.py — Daily and cumulative P&L report for the FVG paper account.

Run standalone:
    python scripts/fvg_pnl_report.py

Or imported from a cron job that passes the report text somewhere.
"""
import sys
import csv
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

# US/Eastern — the reference timezone for all market-session logic.
_ET = ZoneInfo("America/New_York")

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# ── Alpaca account ────────────────────────────────────────────────────────────
try:
    from alpaca.trading.client import TradingClient
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import os as _os
    API_KEY    = _os.getenv("ALPACA_API_KEY", "")
    SECRET_KEY  = _os.getenv("ALPACA_SECRET_KEY", "")
    has_alpaca  = bool(API_KEY and SECRET_KEY)
except Exception:
    has_alpaca = False

# ── CSV trade log ─────────────────────────────────────────────────────────────
TRADE_LOG = ROOT / "paper_trades.csv"

def get_account() -> dict:
    if not has_alpaca:
        return {}
    try:
        client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        acc = client.get_account()
        return {
            "equity": float(acc.equity),
            "cash": float(acc.cash),
            "currency": "USD",
        }
    except Exception as e:
        return {"error": str(e)}

def get_positions() -> list[dict]:
    if not has_alpaca:
        return []
    try:
        client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealised_pnl": float(p.unrealized_pl),
            }
            for p in client.get_all_positions()
        ]
    except Exception:
        return []

def read_trades() -> list[dict]:
    if not TRADE_LOG.exists():
        return []
    with open(TRADE_LOG) as f:
        reader = csv.DictReader(f)
        return list(reader)

def calc_pnl(trades: list[dict]) -> dict:
    """Compute daily and total P&L from trade log.

    "Today" is the US/Eastern calendar date, not the host's local date.
    The host runs on BST and the US session closes at 16:00 ET = 21:00
    BST, so the two agree during the trading day — but the report also
    runs at other times, and a trade bucketed under the wrong day makes
    the daily P&L line wrong. `exit_date` in the trade log is already an
    ET date.
    """
    today = str(datetime.now(_ET).date())
    total_pnl = 0.0
    daily_pnl = 0.0
    total_bps = 0.0
    daily_bps = 0.0
    closed_trades = 0
    unfilled = 0
    wins = 0
    losses = 0

    for t in trades:
        pnl = float(t.get("pnl", 0) or 0)
        bps = float(t.get("pnl_bps", 0) or 0)
        status = t.get("status", "")
        total_pnl += pnl
        total_bps += bps
        if t.get("exit_date", "") == today:
            daily_pnl += pnl
            daily_bps += bps
        if status == "unfilled":
            unfilled += 1
        elif status in ("closed", "stopped", "target"):
            closed_trades += 1
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    return {
        "total_pnl": total_pnl,
        "daily_pnl": daily_pnl,
        "total_bps": total_bps,
        "daily_bps": daily_bps,
        "closed_trades": closed_trades,
        "unfilled": unfilled,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
    }

def build_report() -> str:
    account = get_account()
    positions = get_positions()
    trades = read_trades()
    pnl = calc_pnl(trades)

    # The label said "ET" but datetime.now() is HOST-LOCAL time (BST),
    # so the report was timestamped 15:36 "ET" when it was 10:36 ET.
    # Use the actual ET conversion.
    now = datetime.now(_ET).strftime("%Y-%m-%d %H:%M ET")

    lines = [
        f"📊 FVG Paper Trading — P&L Report  ({now})",
        "=" * 48,
        "",
        f"**Account**  (Alpaca paper)",
    ]

    if "error" in account:
        lines.append(f"  ⚠️  Could not fetch account: {account['error']}")
    else:
        lines.append(f"  Equity : **${account['equity']:>12,.2f}**")
        lines.append(f"  Cash   :   ${account['cash']:>12,.2f}")

    lines += ["", "**Open Positions**"]
    if not positions:
        lines.append("  None")
    else:
        for p in positions:
            pnl_pct = (p["current_price"] - p["avg_entry"]) / p["avg_entry"] * 100
            lines.append(
                f"  {p['symbol']:6s}  qty={p['qty']:>7.3f}  "
                f"avg=${p['avg_entry']:>8.2f}  cur=${p['current_price']:>8.2f}  "
                f"{'▲' if pnl_pct >= 0 else '▼'}{abs(pnl_pct):.2f}%  "
                f"pnl=${p['unrealised_pnl']:>+8.2f}"
            )

    # Unrealised total from positions
    unrealised_total = sum(p["unrealised_pnl"] for p in positions)
    lines += [
        "",
        f"  Unrealised P&L (open): **${unrealised_total:>+9,.2f}**",
        "",
        "**Closed Trade Log**",
        f"  Total closed : {pnl['closed_trades']} trades",
        f"  Wins / Losses: {pnl['wins']} / {pnl['losses']}",
        f"  Win rate     : {pnl['win_rate']*100:.1f}%",
        f"  Daily P&L    : **${pnl['daily_pnl']:>+9,.2f}**  ({pnl['daily_bps']:+.1f} bps)",
        f"  Cumulative   : **${pnl['total_pnl']:>+9,.2f}**  ({pnl['total_bps']:+.1f} bps)",
    ]

    if pnl["unfilled"]:
        lines.append(f"  Unfilled     : {pnl['unfilled']} signal(s) never filled")

    # Recent trades, so the report shows what actually happened rather
    # than just an aggregate.
    recent = [t for t in trades if t.get("status") in ("closed", "unfilled")][-8:]
    if recent:
        lines += ["", "**Recent Trades**"]
        for t in recent:
            bps = float(t.get("pnl_bps", 0) or 0)
            pnl_d = float(t.get("pnl", 0) or 0)
            lines.append(
                f"  {t.get('exit_date','?'):10s}  {t.get('symbol','?'):6s} "
                f"{(t.get('direction','') or '')[:4]:4s}  "
                f"qty={float(t.get('qty',0) or 0):>6.2f}  "
                f"entry={t.get('entry_price','?')}  "
                f"exit={t.get('exit_price','-')}  "
                f"{bps:>+7.1f} bps  ${pnl_d:>+7.2f}  "
                f"{t.get('exit_reason','?')}"
            )

    lines += [
        "",
        f"  Paper trades log: {TRADE_LOG}",
    ]

    return "\n".join(lines)

if __name__ == "__main__":
    print(build_report())
