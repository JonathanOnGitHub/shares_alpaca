#!/usr/bin/env python3
"""
fvg_cleanup_check.py — end-of-session safety check for the Alpaca paper account.

Run at the close of the US session by scripts/fvg_eod_report.sh. It:

  1. Cancels any order still working at the broker. A DAY limit order
     left behind by a killed process could fill overnight with no stop
     attached, turning a paper short into an unmanaged position.

  2. Reports any open positions. Positions left open are a warning: the
     intraday strategy is meant to be flat by the end of the session.

NOTE: this must be a real file rather than a `python - <<PY` heredoc.
python-dotenv's find_dotenv() walks the call stack and asserts on
`frame.f_back`, which is None when the source is piped in on stdin.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.requests import GetOrdersRequest  # noqa: E402

# Statuses that mean the order can still fill.
LIVE_STATUSES = {"new", "partially_filled", "pending_new", "accepted", "held"}


def status_of(order) -> str:
    """Normalise an order status to a lowercase string.

    alpaca-py returns an `OrderStatus` enum whose `str()` is
    "OrderStatus.NEW", not "new". The wire value is in `.value`.
    """
    st = getattr(order, "status", "")
    val = getattr(st, "value", None)
    if val is not None:
        return str(val).lower()
    return str(st).lower()


def main() -> int:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
        return 1

    client = TradingClient(api_key, secret, paper=True)

    cancelled = 0
    try:
        orders = client.get_orders(GetOrdersRequest(status="all", limit=100))
    except Exception as e:
        print(f"ERROR: could not fetch orders: {e}")
        return 1

    for o in orders or []:
        if status_of(o) in LIVE_STATUSES:
            try:
                client.cancel_order_by_id(o.id)
                print(f"cancelled stale order {o.symbol} qty={o.qty} ({status_of(o)})")
                cancelled += 1
            except Exception as e:
                print(f"could not cancel {o.symbol}: {e}")

    positions = client.get_all_positions()
    if positions:
        print("WARNING — open positions remain:")
        for p in positions:
            print(f"  {p.symbol} qty={p.qty} avg={p.avg_entry_price}")
    else:
        print("Account is flat.")

    print(f"cancelled {cancelled} stale order(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
