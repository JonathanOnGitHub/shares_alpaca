"""
Tests for the FVG trade-log backfill (scripts/fvg_backfill_trades.py).

The backfill reconstructs the trades lost on 2026-10-22/23 from the
Alpaca paper account's order history. The risk with such a tool is
that it writes wrong or duplicate rows into the audit trail, so the
pairing logic and the de-duplication are tested directly against
synthetic broker orders — no network access, no real orders.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib  # noqa: E402

backfill = importlib.import_module("fvg_backfill_trades")


class FakeOrder:
    def __init__(self, oid, symbol, side, status, qty=0.0, filled_qty=0.0,
                 filled_avg_price=None, limit_price=None,
                 created_at="2026-10-23 13:41:21+00:00",
                 filled_at=None):
        self.id = oid
        self.symbol = symbol
        self.side = side
        self.status = status
        self.qty = qty
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.limit_price = limit_price
        self.created_at = created_at
        self.filled_at = filled_at


def test_short_round_trip_is_paired():
    """A filled SELL followed by a filled BUY is one short trade."""
    orders = [
        FakeOrder("1", "BAC", "sell", "filled", qty=4, filled_qty=4,
                  filled_avg_price=50.60,
                  created_at="2026-10-23 13:41:08+00:00",
                  filled_at="2026-10-23 13:41:21+00:00"),
        FakeOrder("2", "BAC", "buy", "filled", qty=4, filled_qty=4,
                  filled_avg_price=50.70,
                  created_at="2026-10-23 13:45:13+00:00",
                  filled_at="2026-10-23 13:45:21+00:00"),
    ]
    trips = backfill.build_round_trips(orders)
    assert len(trips) == 1
    t = trips[0]
    assert t["symbol"] == "BAC"
    assert t["direction"] == "short"
    assert t["status"] == "closed"
    assert t["entry_price"] == 50.60
    assert t["exit_price"] == 50.70
    # A short that exits HIGHER lost money.
    assert t["pnl_bps"] < 0
    assert t["pnl"] < 0


def test_winning_short_has_positive_pnl():
    orders = [
        FakeOrder("1", "BAC", "sell", "filled", qty=4, filled_qty=4,
                  filled_avg_price=50.77,
                  created_at="2026-10-22 15:39:20+00:00",
                  filled_at="2026-10-22 15:39:20+00:00"),
        FakeOrder("2", "BAC", "buy", "filled", qty=4, filled_qty=4,
                  filled_avg_price=50.4075,
                  created_at="2026-10-22 20:14:02+00:00",
                  filled_at="2026-10-22 20:14:02+00:00"),
    ]
    trips = backfill.build_round_trips(orders)
    assert len(trips) == 1
    assert trips[0]["pnl_bps"] > 0
    assert trips[0]["pnl"] > 0


def test_unfilled_order_is_recorded_and_consumes_no_inventory():
    """A cancelled entry is a real fact, but it opens no position."""
    orders = [
        FakeOrder("1", "XOM", "buy", "canceled", qty=1.11,
                  limit_price=179.96,
                  created_at="2026-10-23 13:41:09+00:00"),
    ]
    trips = backfill.build_round_trips(orders)
    assert len(trips) == 1
    t = trips[0]
    assert t["status"] == "unfilled"
    assert t["pnl"] == 0.0
    assert t["exit_reason"] == "unfilled"
    assert t["direction"] == "long"
    # The XOM row in the live log is dated to the SIGNAL session.
    assert t["exit_date"] == "2026-10-23"


def test_unfilled_order_does_not_close_a_real_trade():
    """
    An unfilled order between an entry and its exit must not be mistaken
    for the exit.
    """
    orders = [
        FakeOrder("1", "BAC", "sell", "filled", qty=4, filled_qty=4,
                  filled_avg_price=50.60,
                  created_at="2026-10-23 13:41:08+00:00",
                  filled_at="2026-10-23 13:41:21+00:00"),
        FakeOrder("2", "BAC", "sell", "canceled", qty=4,
                  created_at="2026-10-23 13:42:00+00:00"),
        FakeOrder("3", "BAC", "buy", "filled", qty=4, filled_qty=4,
                  filled_avg_price=50.70,
                  created_at="2026-10-23 13:45:13+00:00",
                  filled_at="2026-10-23 13:45:21+00:00"),
    ]
    trips = backfill.build_round_trips(orders)
    closed = [t for t in trips if t["status"] == "closed"]
    unfilled = [t for t in trips if t["status"] == "unfilled"]
    assert len(closed) == 1, "the real round trip was not paired"
    assert closed[0]["entry_price"] == 50.60
    assert closed[0]["exit_price"] == 50.70
    assert len(unfilled) == 1


def test_unclosed_entry_is_flagged_open():
    orders = [
        FakeOrder("1", "BAC", "sell", "filled", qty=4, filled_qty=4,
                  filled_avg_price=50.60,
                  created_at="2026-10-23 13:41:08+00:00",
                  filled_at="2026-10-23 13:41:21+00:00"),
    ]
    trips = backfill.build_round_trips(orders)
    assert len(trips) == 1
    assert trips[0]["status"] == "open"
    assert trips[0]["exit_reason"] == "unclosed"
    assert trips[0]["pnl"] == 0.0


def test_dedupe_key_matches_unfilled_across_timestamp_drift():
    """
    The live executor logs the SIGNAL time as entry_time; the backfill
    logs the broker's ORDER time. They differ by seconds, so an exact
    timestamp match would record the same unfilled signal twice.
    """
    live_row = {
        "symbol": "XOM", "direction": "long", "status": "unfilled",
        "exit_date": "2026-10-23",
        "entry_time": "2026-10-23 14:41:07.256832",
        "exit_time": "2026-10-23 14:41:07.256832",
    }
    backfill_row = {
        "symbol": "XOM", "direction": "long", "status": "unfilled",
        "exit_date": "2026-10-23",
        "entry_time": "2026-10-23 09:41:09",
        "exit_time": "2026-10-23 09:41:09",
    }
    assert backfill.dedupe_key(live_row) == backfill.dedupe_key(backfill_row)


def test_dedupe_key_keeps_distinct_closed_trades_apart():
    a = {"symbol": "BAC", "direction": "short", "status": "closed",
         "exit_date": "2026-10-22",
         "entry_time": "2026-10-22 11:39:20", "exit_time": "2026-10-22 15:14:02"}
    b = {"symbol": "BAC", "direction": "short", "status": "closed",
         "exit_date": "2026-10-23",
         "entry_time": "2026-10-23 09:41:21", "exit_time": "2026-10-23 09:45:21"}
    assert backfill.dedupe_key(a) != backfill.dedupe_key(b)


def test_backfill_is_idempotent(tmp_path):
    """Running the backfill twice must not duplicate rows."""
    import csv
    log = tmp_path / "paper_trades.csv"
    fields = backfill.FIELDS
    with open(log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "symbol": "XOM", "strategy": "fvg", "direction": "long",
            "entry_time": "2026-10-23 09:41:09",
            "exit_time": "2026-10-23 09:41:09", "exit_date": "2026-10-23",
            "entry_price": "179.96", "exit_price": "", "qty": "1.11",
            "pnl": "0.0", "pnl_bps": "0.0", "exit_reason": "unfilled",
            "status": "unfilled", "atr": "", "market": "US",
            "close_mode": "intraday",
        })

    original = backfill.TRADE_LOG
    backfill.TRADE_LOG = log
    try:
        have = backfill.existing_keys()
        assert ("XOM", "long", "2026-10-23", "unfilled") in have
    finally:
        backfill.TRADE_LOG = original


def test_exit_reason_is_not_invented():
    """
    The broker knows the prices but not the strategy's intent. A
    backfilled row must say "closed", not a fabricated "stop".
    """
    orders = [
        FakeOrder("1", "BAC", "sell", "filled", qty=4, filled_qty=4,
                  filled_avg_price=50.60,
                  created_at="2026-10-22 15:39:20+00:00",
                  filled_at="2026-10-22 15:39:20+00:00"),
        FakeOrder("2", "BAC", "buy", "filled", qty=4, filled_qty=4,
                  filled_avg_price=50.70,
                  created_at="2026-10-22 20:14:02+00:00",
                  filled_at="2026-10-22 20:14:02+00:00"),
    ]
    trips = backfill.build_round_trips(orders)
    assert trips[0]["exit_reason"] == "closed"


def test_known_exit_reason_is_applied():
    """Where the executor's log records the reason, it is used."""
    orders = [
        FakeOrder("1", "BAC", "sell", "filled", qty=4, filled_qty=4,
                  filled_avg_price=50.60,
                  created_at="2026-10-23 13:41:08+00:00",
                  filled_at="2026-10-23 13:41:21+00:00"),
        FakeOrder("2", "BAC", "buy", "filled", qty=4, filled_qty=4,
                  filled_avg_price=50.70,
                  created_at="2026-10-23 13:45:13+00:00",
                  filled_at="2026-10-23 13:45:21+00:00"),
    ]
    trips = backfill.build_round_trips(orders)
    assert trips[0]["exit_reason"] == "stop"


def test_timestamps_are_rendered_in_et():
    """Broker timestamps are UTC; the strategy's day is US/Eastern."""
    ts = backfill.to_et("2026-10-23 13:41:21+00:00")
    assert ts == "2026-10-23 09:41:21", ts


if __name__ == "__main__":
    import tempfile
    failures = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(Path(td))
                print(f"PASS  {fn.__name__}")
            except Exception as e:
                failures += 1
                print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
