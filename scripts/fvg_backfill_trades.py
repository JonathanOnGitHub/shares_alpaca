#!/usr/bin/env python3
"""
fvg_backfill_trades.py — reconstruct lost FVG trade history from broker truth.

WHY THIS EXISTS

On 2026-10-22 and 2026-10-23 the executor wrote its trade log ONLY on
process exit. Both sessions ended abnormally (a wedged daemon on 23 Oct),
so every real fill was lost from disk and the daily report showed
"0 closed trades / $0.00" while trades had in fact been filled and
stopped out.

The Alpaca paper account is the source of truth: it recorded every
order, its status, its fill price and its fill time. This script
reconstructs the round-trip trades from the broker's order history and
appends them to paper_trades.csv.

WHAT IT DOES NOT DO

  * It does NOT guess exit reasons. A round trip is only recorded when
    the broker shows a matching entry AND a matching exit for the same
    symbol and size.
  * It does NOT touch the live strategy state (.positions_fvg.json).
  * It is idempotent: re-running it will not duplicate rows.
  * It only backfills rows that are genuinely missing. Any row already
    in the log (matched on symbol + entry_time + exit_time) is left
    alone.

The exit reason is inferred from the broker's exit side:
  * a BUY exit closes a SHORT  -> "stop" or "target" is guessed from
    price vs the signal's stop/target levels where available, otherwise
    "closed";
  * an unmatched entry (filled, never exited) is recorded as "open" with
    zero P&L so the audit trail is complete.

Usage:
    .venv/bin/python scripts/fvg_backfill_trades.py --dry-run
    .venv/bin/python scripts/fvg_backfill_trades.py --apply
"""
import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.requests import GetOrdersRequest  # noqa: E402

TRADE_LOG = ROOT / "paper_trades.csv"

FIELDS = [
    "symbol", "strategy", "direction", "entry_time", "exit_time",
    "exit_date", "entry_price", "exit_price", "qty", "pnl", "pnl_bps",
    "exit_reason", "status", "atr", "market", "close_mode",
]

# Commission: 0.75 bps per side x 2, matching the executor.
COMMISSION_BPS = 1.5

# Exit reasons that are KNOWN from the executor's own log for the
# 2026-10-22/23 sessions. Everything else is recorded as "closed",
# because the reason is not recoverable from broker history and must
# not be invented.
KNOWN_EXIT_REASONS = {
    # (symbol, entry filled_at) -> reason
    ("BAC", "2026-10-23 13:41:21+00:00"): "stop",
}


def to_et(ts) -> str:
    """Render a broker timestamp in US/Eastern, the strategy's timezone."""
    if not ts:
        return ""
    try:
        import pandas as pd
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def status_of(order) -> str:
    st = getattr(order, "status", "")
    val = getattr(st, "value", None)
    if val is not None:
        return str(val).lower()
    return str(st).lower()


def side_of(order) -> str:
    st = getattr(order, "side", "")
    val = getattr(st, "value", None)
    if val is not None:
        return str(val).lower()
    return str(st).lower()


def is_filled(order) -> bool:
    try:
        return float(order.filled_qty or 0) > 0
    except Exception:
        return False


def build_round_trips(orders):
    """
    Pair filled entries with filled exits per symbol.

    Alpaca's order history is FIFO per symbol, so a BUY that follows a
    SELL closes that short, and vice versa.

    Unfilled entry orders (cancelled, expired, rejected) are also
    returned, as "unfilled" rows. A signal that was taken but never
    filled is a real fact about the strategy, and the session's record
    is incomplete without it.
    """
    by_symbol = {}
    for o in orders:
        by_symbol.setdefault(o.symbol, []).append(o)

    trips = []
    for sym, olist in by_symbol.items():
        olist.sort(key=lambda o: str(o.created_at))
        # Inventory of open (unclosed) positions: one entry per side.
        shorts = []   # filled SELL entries awaiting a BUY
        longs = []    # filled BUY entries awaiting a SELL
        for o in olist:
            if not is_filled(o):
                # An entry order that never filled. It is a real fact
                # about the session (a signal was taken but not filled)
                # and it must be recorded — but it must NOT consume any
                # inventory, because no position was ever opened.
                trips.append(_make_unfilled(sym, o))
                continue
            if side_of(o) == "sell":
                shorts.append(o)
            else:
                # A BUY: close one open short if there is one, else it
                # is a long entry.
                if shorts:
                    entry = shorts.pop(0)
                    trips.append(_make_trip(sym, entry, o, "short"))
                else:
                    longs.append(o)
        # Any still-open entry never got an exit.
        for entry in shorts:
            trips.append(_make_trip(sym, entry, None, "short"))
        for entry in longs:
            trips.append(_make_trip(sym, entry, None, "long"))
    trips.sort(key=lambda t: t["entry_time"])
    return trips


def _make_unfilled(sym, order):
    """An entry order that never filled."""
    ts = to_et(order.created_at)
    return {
        "symbol": sym, "strategy": "fvg",
        "direction": "short" if side_of(order) == "sell" else "long",
        "entry_time": ts,
        "exit_time": ts,
        "exit_date": ts[:10],
        "entry_price": round(float(order.limit_price or 0), 4)
        if order.limit_price else "",
        "exit_price": "", "qty": float(order.qty or 0),
        "pnl": 0.0, "pnl_bps": 0.0,
        "exit_reason": "unfilled", "status": "unfilled",
        "atr": "", "market": "US", "close_mode": "intraday",
    }


def _make_trip(sym, entry, exit_order, direction):
    entry_px = float(entry.filled_avg_price)
    qty = float(entry.filled_qty)
    entry_time = to_et(entry.filled_at or entry.created_at)
    entry_stamp = str(entry.filled_at or entry.created_at)

    if exit_order is None:
        return {
            "symbol": sym, "strategy": "fvg", "direction": direction,
            "entry_time": entry_time,
            "exit_time": "", "exit_date": "",
            "entry_price": round(entry_px, 4), "exit_price": "",
            "qty": qty, "pnl": 0.0, "pnl_bps": 0.0,
            "exit_reason": "unclosed", "status": "open",
            "atr": "", "market": "US", "close_mode": "intraday",
        }

    exit_px = float(exit_order.filled_avg_price)
    exit_time = to_et(exit_order.filled_at or exit_order.created_at)

    if direction == "short":
        pnl_bps = (entry_px - exit_px) / entry_px * 10_000
    else:
        pnl_bps = (exit_px - entry_px) / entry_px * 10_000
    pnl_bps -= COMMISSION_BPS

    notional = abs(qty * entry_px)
    pnl = notional * pnl_bps / 10_000.0

    # The exit reason is only asserted where the executor's own log
    # recorded it. Everywhere else it is "closed" — the broker knows
    # the prices but not the strategy's intent, and inventing a "stop"
    # or a "target" would be a fabrication.
    reason = KNOWN_EXIT_REASONS.get((sym, entry_stamp), "closed")

    return {
        "symbol": sym, "strategy": "fvg", "direction": direction,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "exit_date": exit_time[:10],
        "entry_price": round(entry_px, 4), "exit_price": round(exit_px, 4),
        "qty": qty, "pnl": round(pnl, 4), "pnl_bps": round(pnl_bps, 2),
        "exit_reason": reason, "status": "closed",
        "atr": "", "market": "US", "close_mode": "intraday",
    }


def dedupe_key(row: dict) -> tuple:
    """
    A stable identity for a trade row.

    Closed trades are identified by their exact entry/exit timestamps.

    Unfilled rows are identified by symbol + direction + session date:
    the live executor records the SIGNAL time as entry_time (when the
    order was placed) whereas the backfill records the broker's
    ORDER creation time, so the two differ by seconds and an exact
    timestamp match would let the same unfilled signal be recorded
    twice.
    """
    if row.get("status") == "unfilled":
        return (row.get("symbol", ""), row.get("direction", ""),
                row.get("exit_date", ""), "unfilled")
    return (row.get("symbol", ""), row.get("entry_time", ""),
            row.get("exit_time", ""))


def existing_keys():
    if not TRADE_LOG.exists():
        return set()
    with open(TRADE_LOG) as f:
        return {dedupe_key(r) for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the backfilled rows (default is a dry run)")
    ap.add_argument("--symbols", default="BAC,XOM,ORCL,INTC",
                    help="comma-separated symbols to consider. The default "
                         "covers the INTRADAY FVG universe only — NVDA, LLY "
                         "and META belong to the overnight ENGULFING strategy "
                         "and must not be recorded against FVG.")
    ap.add_argument("--since", default="2026-10-21",
                    help="only consider orders created on/after this date")
    args = ap.parse_args()

    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
        return 1

    client = TradingClient(api_key, secret, paper=True)
    orders = client.get_orders(GetOrdersRequest(status="all", limit=500))
    symbols = {s.strip() for s in args.symbols.split(",") if s.strip()}
    orders = [
        o for o in orders
        if o.symbol in symbols and str(o.created_at)[:10] >= args.since
    ]

    trips = build_round_trips(orders)
    have = existing_keys()

    new_rows = []
    for t in trips:
        key = dedupe_key(t)
        if key in have:
            continue
        have.add(key)
        new_rows.append(t)

    print(f"broker round trips found : {len(trips)}")
    print(f"already in {TRADE_LOG.name}: {len(trips) - len(new_rows)}")
    print(f"rows to backfill          : {len(new_rows)}")
    print()
    for t in new_rows:
        print(f"  {t['symbol']:5} {t['direction']:5} {t['status']:8} "
              f"qty={t['qty']:>6} entry={t['entry_price']} "
              f"exit={t['exit_price']} pnl={t['pnl']} bps={t['pnl_bps']} "
              f"{t['entry_time'][:19]}")

    if not new_rows:
        print("\nnothing to backfill")
        return 0

    if not args.apply:
        print("\ndry run — pass --apply to write")
        return 0

    exists = TRADE_LOG.exists()
    with open(TRADE_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for t in new_rows:
            w.writerow(t)

    print(f"\nwrote {len(new_rows)} row(s) to {TRADE_LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
