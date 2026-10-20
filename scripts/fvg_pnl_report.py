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
    """Compute daily and total P&L from trade log."""
    today = str(date.today())
    total_pnl = 0.0
    daily_pnl = 0.0
    closed_trades = 0
    wins = 0
    losses = 0

    for t in trades:
        pnl = float(t.get("pnl", 0) or 0)
        total_pnl += pnl
        if t.get("exit_date", "") == today:
            daily_pnl += pnl
        if t.get("status", "") in ("closed", "stopped", "target"):
            closed_trades += 1
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    return {
        "total_pnl": total_pnl,
        "daily_pnl": daily_pnl,
        "closed_trades": closed_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
    }

def build_report() -> str:
    account = get_account()
    positions = get_positions()
    trades = read_trades()
    pnl = calc_pnl(trades)

    now = datetime.now().strftime("%Y-%m-%d %H:%M ET")

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
        f"  Daily P&L    : **${pnl['daily_pnl']:>+9,.2f}**",
        f"  Cumulative   : **${pnl['total_pnl']:>+9,.2f}**",
        "",
        f"  Paper trades log: {TRADE_LOG}",
    ]

    return "\n".join(lines)

if __name__ == "__main__":
    print(build_report())
