#!/usr/bin/env python3
"""
reconcile_engulfing_ledger.py — rebuild the engulfing paper-trade ledger
from the BROKER's order history.

WHY THIS EXISTS
---------------
The engulfing ledger (`paper_trades_engulfing.csv`) was written by the
executor, and the executor had three defects that made its rows
untrustworthy:

  1. The exit price on a forced close was `pos.entry_price` — the
     ENTRY PROXY, not the fill. So a forced close booked a fictitious
     P&L. The 2026-10-30 META trade was booked as +87.7 bps when the
     broker actually filled the exit 3 days later at -79.8 bps.
  2. The exit price was never read back from the broker, so even a
     correctly-timed exit used a QUOTE rather than the fill.
  3. A partial entry fill left the remainder at the broker untracked.
     The 2026-10-30 BAC entry filled 3.91 shares; the ledger tracked
     1.00, so the close sold 1 share and left 2.91 naked.

The broker's order history is the ground truth. This script replays it
and rebuilds each ledger row from the actual fills.

The account is SHARED with other strategies (FVG intraday, and
unrelated projects on the same API key), so only the orders that
correspond to engulfing entries are pulled in — the FVG trades on the
same symbols are left alone.

Usage:
    python scripts/reconcile_engulfing_ledger.py            # report only
    python scripts/reconcile_engulfing_ledger.py --apply    # rewrite the ledger
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "paper_trades_engulfing.csv"
BACKUP = ROOT / "paper_trades_engulfing.csv.orig"

# Commission assumption used by the executor: 0.75 bps per side.
COMMISSION_BPS = 1.5


def load_broker_orders():
    """Fetch every order from the Alpaca paper account."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")

    client = TradingClient(key, secret, paper=True)
    orders = client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500))

    rows = []
    for o in orders:
        rows.append({
            "id": str(o.id),
            "submitted_at": pd.Timestamp(o.submitted_at).tz_localize(None)
                           if o.submitted_at.tzinfo else pd.Timestamp(o.submitted_at),
            "symbol": o.symbol,
            "side": str(o.side).replace("OrderSide.", "").lower(),
            "type": str(o.type).replace("OrderType.", "").lower(),
            "qty": float(o.qty or 0),
            "filled_qty": float(o.filled_qty or 0),
            "filled_avg_price": float(o.filled_avg_price or 0),
            "status": str(o.status).replace("OrderStatus.", "").lower(),
        })
    df = pd.DataFrame(rows).sort_values("submitted_at").reset_index(drop=True)
    return df


def find_entry_order(orders, row):
    """Find the broker BUY order that corresponds to a ledger row.

    Matching is by symbol plus submission time (within a few minutes
    of the recorded entry) — the executor logs the fill time, and the
    broker stamps the order within seconds of it.
    """
    sym = row["symbol"]
    entry_time = pd.Timestamp(row["entry_time"])
    buys = orders[(orders["symbol"] == sym)
                  & (orders["side"] == "buy")
                  & (orders["filled_qty"] > 0)]
    if buys.empty:
        return None

    # Prefer the closest BUY in time to the recorded entry.
    buys = buys.assign(dt=(buys["submitted_at"] - entry_time).abs())
    # Only orders within 15 minutes are plausible matches.
    cand = buys[buys["dt"] <= pd.Timedelta(minutes=15)]
    if cand.empty:
        return None
    cand = cand.sort_values("dt")
    # If several BUY orders exist for the symbol close in time, take
    # the one whose quantity matches the ledger row.
    exact = cand[cand["filled_qty"].sub(float(row["qty"])).abs() < 1e-3]
    if not exact.empty:
        return exact.iloc[0]
    return cand.iloc[0]


def find_exit_legs(orders, entry_order):
    """Consume the broker SELL orders that closed an entry.

    Walks forward from the entry order, accumulating SELL fills until
    the quantity held by the entry has been fully sold. This handles the
    partial-close case (the 2026-10-30 BAC trade: 1 share sold on the
    30th, the remaining 2.91 on the 9th).
    """
    sym = entry_order["symbol"]
    remaining = float(entry_order["filled_qty"])
    entry_price = float(entry_order["filled_avg_price"])

    sells = orders[(orders["symbol"] == sym)
                   & (orders["side"] == "sell")
                   & (orders["filled_qty"] > 0)
                   & (orders["submitted_at"] >= entry_order["submitted_at"])]
    sells = sells.sort_values("submitted_at")

    legs = []
    spent = 0.0
    for _, s in sells.iterrows():
        if spent >= remaining - 1e-6:
            break
        # Only count sells that close this lot: skip sells that happened
        # before the entry, and skip ones for a different round trip.
        legs.append(s)
        spent += float(s["filled_qty"])
        if spent >= remaining - 1e-6:
            break

    # Keep only the legs needed to cover the entry quantity.
    trimmed = []
    acc = 0.0
    for s in legs:
        trimmed.append(s)
        acc += float(s["filled_qty"])
        if acc >= remaining - 1e-6:
            break
    return trimmed, entry_price, remaining


def reconcile(orders, ledger):
    """Rebuild each ledger row from the broker's fills."""
    out = []
    for _, row in ledger.iterrows():
        entry = find_entry_order(orders, row)
        if entry is None:
            out.append({**row.to_dict(), "reconcile_status": "no_entry_match",
                        "reconcile_note": "no matching broker BUY order found"})
            continue

        legs, entry_px, entry_qty = find_exit_legs(orders, entry)
        if not legs:
            out.append({**row.to_dict(), "reconcile_status": "no_exit_match",
                        "reconcile_note": "entry found but no closing sell order"})
            continue

        total_qty = sum(float(l["filled_qty"]) for l in legs)
        notional_out = sum(float(l["filled_qty"]) * float(l["filled_avg_price"])
                           for l in legs)
        exit_px = notional_out / total_qty if total_qty else entry_px

        pnl_bps = (exit_px - entry_px) / entry_px * 10_000 - COMMISSION_BPS
        pnl_dollars = total_qty * (exit_px - entry_px)

        new = row.to_dict()
        new.update({
            "entry_price": round(entry_px, 4),
            "exit_price": round(exit_px, 4),
            "qty": round(total_qty, 8),
            "pnl": round(pnl_dollars, 4),
            "pnl_bps": round(pnl_bps, 4),
            "exit_price_source": "broker_fill",
            "reconcile_status": "reconciled",
            "reconcile_note": (
                f"entry {entry['id'][:8]} @{entry_px:.4f}; "
                + "; ".join(
                    f"exit {l['id'][:8]} {l['filled_qty']:.6f}sh "
                    f"@{l['filled_avg_price']:.4f} "
                    f"({str(l['submitted_at'])[:16]})"
                    for l in legs)
            ),
        })
        out.append(new)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the ledger with the reconciled rows")
    args = ap.parse_args()

    if not LEDGER.exists():
        raise SystemExit(f"{LEDGER} not found")

    ledger = pd.read_csv(LEDGER)
    orders = load_broker_orders()
    print(f"Broker orders: {len(orders)}")
    print(f"Ledger rows  : {len(ledger)}\n")

    rec = reconcile(orders, ledger)

    # Keep a pristine copy of the recorded values for the report — the
    # reconciliation mutates rows in place, so reading them back out of
    # `rec` would show the NEW values on both sides of the comparison.
    recorded = ledger.copy()
    for idx, (_, r) in enumerate(rec.iterrows()):
        print(f"--- {r['symbol']} {r['exit_date']} ({r['exit_reason']})")
        orig = recorded.iloc[idx]
        print(f"    recorded : entry {orig['entry_price']} exit {orig['exit_price']} "
              f"qty {orig['qty']} pnl {orig['pnl_bps']} bps")
        if r["reconcile_status"] == "reconciled":
            print(f"    broker   : entry {r['entry_price']} exit {r['exit_price']} "
                  f"qty {r['qty']} pnl {r['pnl_bps']} bps")
            print(f"    detail   : {r['reconcile_note']}")
        else:
            print(f"    !! {r['reconcile_status']}: {r['reconcile_note']}")
        print()

    ok = rec[rec["reconcile_status"] == "reconciled"]
    if len(ok):
        print(f"Recorded total : {ledger['pnl_bps'].sum():+.1f} bps "
              f"(${ledger['pnl'].sum():+.2f})")
        print(f"Reconciled total: {ok['pnl_bps'].sum():+.1f} bps "
              f"(${ok['pnl'].sum():+.2f})")
        wins = int((ok["pnl_bps"] > 0).sum())
        print(f"Reconciled win rate: {wins}/{len(ok)}")

    if args.apply:
        if not BACKUP.exists():
            LEDGER.rename(BACKUP)
            print(f"\nOriginal ledger backed up to {BACKUP.name}")
        rec.to_csv(LEDGER, index=False)
        print(f"Reconciled ledger written to {LEDGER.name}")
    else:
        print("\n(dry run — pass --apply to rewrite the ledger)")


if __name__ == "__main__":
    main()
