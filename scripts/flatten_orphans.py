#!/usr/bin/env python3
"""
flatten_orphans.py — close broker positions that no strategy is managing.

On 2026-11-07 the audit found three positions at the Alpaca paper
account (INTC, BAC, ORCL) that NEITHER the FVG nor the engulfing
executor owned: both `.positions_*.json` files were empty, so no stop,
no target and no flat deadline applied to them. They sat unmanaged
for days.

This script is the safe, auditable way to close them:
  * it re-reads the live broker position list (never a cached one),
  * it closes each position by exact symbol,
  * it verifies afterwards that the account is flat,
  * and it prints a JSON summary of what it did.

Run it with --dry-run first to see what WOULD be closed.

Usage:
    .venv/bin/python scripts/flatten_orphans.py --dry-run
    .venv/bin/python scripts/flatten_orphans.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from alpaca.trading.client import TradingClient  # noqa: E402


def get_client():
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
    return TradingClient(key, secret, paper=True)


def main():
    ap = argparse.ArgumentParser(description="Close unmanaged Alpaca paper positions")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="Only these symbols (default: every open position)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be closed without closing it")
    args = ap.parse_args()

    client = get_client()
    positions = [p for p in client.get_all_positions() if abs(float(p.qty)) > 1e-9]
    if args.symbols:
        wanted = {s.upper() for s in args.symbols}
        positions = [p for p in positions if p.symbol.upper() in wanted]

    summary = {
        "dry_run": args.dry_run,
        "found": [],
        "closed": [],
        "errors": [],
    }

    for p in positions:
        qty = float(p.qty)
        info = {
            "symbol": p.symbol,
            "qty": qty,
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "unrealized_pl": round(float(p.unrealized_pl), 2),
        }
        summary["found"].append(info)

        if args.dry_run:
            print(f"DRY-RUN  would close {info}")
            continue

        try:
            client.close_position(p.symbol)
            info["action"] = "close_position called"
            summary["closed"].append(info)
            print(f"CLOSED  {p.symbol} qty={qty} "
                  f"avg={info['avg_entry_price']} cur={info['current_price']} "
                  f"unrealised={info['unrealized_pl']}")
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
            summary["errors"].append(info)
            print(f"ERROR   {p.symbol}: {type(e).__name__}: {e}", file=sys.stderr)

    # Verify.
    if not args.dry_run:
        still = [p.symbol for p in client.get_all_positions()
                 if abs(float(p.qty)) > 1e-9]
        summary["positions_after"] = still
        print(f"\nPositions remaining: {still if still else 'none (flat)'}")

    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
