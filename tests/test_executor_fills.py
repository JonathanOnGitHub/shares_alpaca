"""
Tests for the PaperExecutor fill / order-lifecycle fixes.

Covers the bugs found in the 2026-10-22 audit:

  1. A PARTIALLY filled entry order was never transitioned to "entered",
     so the position sat with no stop, no target and no deadline.
  2. The tracked qty was the REQUESTED qty, not the FILLED qty, so a
     close would move more shares than were actually held.
  3. On a fill, the qty was recomputed as (capital * 1%) / risk, which
     silently discarded the fixed-$100-risk + notional-cap sizing the
     backtest uses.
  4. A resting entry order was not cancelled when the position closed,
     so it could fill later and open an unmanaged position.
  5. The same intraday signal was re-entered on every poll after a
     close, because the scanner re-derives the day's signals.
"""
import logging
import re
import sys
from pathlib import Path
from threading import Event

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.trading.executor_live import PaperExecutor  # noqa: E402


class FakeOrder:
    def __init__(self, oid, symbol, status, filled_qty=0.0, filled_avg_price=None):
        self.id = oid
        self.symbol = symbol
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price


class FakeClient:
    def __init__(self, orders=None):
        self.orders = orders or []
        self.cancelled = []

    def get_orders(self, status="all", limit=50):
        return list(self.orders)

    def get_all_positions(self):
        return []


class FakeBroker:
    """Minimal broker backend for exercising the executor's order paths.

    `price` is the current market price for every symbol; `order_status`
    is the status reported for the order the broker accepts.
    """

    def __init__(self, orders=None, price=100.0, order_status="filled",
                 filled_qty=0.0, filled_avg_price=None, prices=None):
        self.client = FakeClient(orders)
        self.price = price
        self.prices = prices or {}
        self.order_status = order_status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.submitted = []
        self.cancelled = []
        self.closed = []
        self._next_id = 0
    def _record_order(self, symbol, qty):
        """Register the submitted order with the fake client so that
        _get_order()/_is_filled() can find it, exactly as Alpaca would."""
        self._next_id += 1
        oid = f"ord-{self._next_id}"
        self.client.orders.append(
            FakeOrder(oid, symbol, self.order_status,
                      filled_qty=self.filled_qty,
                      filled_avg_price=self.filled_avg_price))
        return oid

    def submit_market_order(self, symbol, qty, side):
        self.submitted.append((symbol, qty, side, "market"))
        return self._record_order(symbol, qty)

    def submit_limit_order(self, symbol, qty, limit_price, side):
        self.submitted.append((symbol, qty, side, "limit"))
        return self._record_order(symbol, qty)

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        for o in self.client.orders:
            if o.id == order_id:
                o.status = "cancelled"

    def close_position(self, symbol, qty=None):
        self.closed.append((symbol, qty))

    def get_positions(self):
        return {}

    def get_cash(self):
        return 100_000.0

    def get_current_price(self, symbol):
        # Per-symbol overrides let a test give one symbol a price that
        # satisfies the stop-geometry guard while another does not.
        return self.prices.get(symbol, self.price)

    def market_hours_open(self):
        return True


def make_executor(broker, tmp_path):
    ex = PaperExecutor.__new__(PaperExecutor)
    ex.broker = broker
    ex.market = "US"
    ex.close_mode = "intraday"
    ex.strategy = "fvg"
    ex.capital = 10_000.0
    ex.positions = {}
    ex.trade_log = []
    ex._bar_count = 0
    ex._acted_signals = set()
    ex._acted_file = Path(tmp_path) / ".acted_fvg.json"
    ex._positions_file = Path(tmp_path) / ".positions_fvg.json"
    ex._foreign_positions_seen = set()
    # Mirror the real constructor: restore any previously acted-on keys
    # from disk so a restart does not re-enter a setup.
    ex._load_acted_signals()
    ex._load_positions()
    ex._stop = Event()
    ex.signals_file = Path(tmp_path) / "signals.json"
    ex.log_file = Path(tmp_path) / "trades.csv"
    ex.poll_seconds = 60
    return ex


# ── 1 & 2 & 3: partial fill transitions to entered with the filled qty ──────

def test_partial_fill_becomes_entered_with_filled_qty(tmp_path):
    # Price must sit BELOW the stop (a short's stop) so the pre-entry
    # gap guard does not skip the signal.
    broker = FakeBroker(price=126.72, order_status="partially_filled",
                        filled_qty=1.0, filled_avg_price=127.02)
    ex = make_executor(broker, tmp_path)

    sig = {
        "symbol": "ORCL", "direction": "short",
        "entry_price": 127.01, "stop_price": 127.3468, "target_price": 126.3364,
        "atr": 0.6736, "risk_dollars": 100.0, "max_risk_mult": 2.0,
        "bar_date": "2026-10-22", "bar_time_et": "09:35",
        "timestamp": pd.Timestamp.now().isoformat(), "flat_time_et": "14:00",
    }
    ex._open_position(sig, pd.Timestamp.now(), "ORCL|2026-10-22|09:35")
    assert "ORCL" in ex.positions
    pos = ex.positions["ORCL"]
    assert pos.status == "pending"

    # The partially filled order MUST transition to "entered".
    ex._monitor_positions()
    pos = ex.positions["ORCL"]
    assert pos.status == "entered", "partial fill left position unmanaged"
    # The tracked qty must be the FILLED qty, not the requested qty.
    assert pos.qty == 1.0, f"expected filled qty 1.0, got {pos.qty}"
    # The fill price must be recorded for P&L.
    assert pos.entry_filled_price == 127.02


def test_partial_fill_does_not_recompute_qty_from_capital(tmp_path):
    """The old code reset qty to (capital*1%)/risk, ignoring the signal's
    fixed risk_dollars and notional cap."""
    broker = FakeBroker(price=126.72, order_status="partially_filled",
                        filled_qty=2.0, filled_avg_price=127.02)
    ex = make_executor(broker, tmp_path)

    sig = {
        "symbol": "ORCL", "direction": "short",
        "entry_price": 127.01, "stop_price": 127.3468, "target_price": 126.3364,
        "atr": 0.6736, "risk_dollars": 100.0, "max_risk_mult": 2.0,
        "bar_date": "2026-10-22", "bar_time_et": "09:35",
        "timestamp": pd.Timestamp.now().isoformat(), "flat_time_et": "14:00",
    }
    ex._open_position(sig, pd.Timestamp.now(), "ORCL|2026-10-22|09:35")
    ex._monitor_positions()
    pos = ex.positions["ORCL"]
    # (10000 * 0.01) / abs(127.01 - 127.35) would be ~294 shares.
    # The filled qty of 2 must survive.
    assert pos.qty == 2.0, f"qty was overwritten: {pos.qty}"


# ── 4: a resting entry order is cancelled when the position closes ─────────

def test_close_cancels_resting_entry_order(tmp_path):
    broker = FakeBroker(price=87.36, order_status="new")
    ex = make_executor(broker, tmp_path)

    sig = {
        "symbol": "INTC", "direction": "short",
        "entry_price": 88.30, "stop_price": 88.645, "target_price": 87.6101,
        "atr": 0.6899, "risk_dollars": 100.0, "max_risk_mult": 2.0,
        "bar_date": "2026-10-22", "bar_time_et": "09:35",
        "timestamp": pd.Timestamp.now().isoformat(), "flat_time_et": "14:00",
    }
    ex._open_position(sig, pd.Timestamp.now(), "INTC|2026-10-22|09:35")
    assert "INTC" in ex.positions
    pos = ex.positions["INTC"]
    pos.status = "entered"
    pos.entry_time = pd.Timestamp.now()
    pos.entry_filled_price = 88.30

    ex._close_position("INTC", "stop", 88.645)

    assert broker.cancelled, "resting entry order was not cancelled"
    assert any(sym == "INTC" for sym, _ in broker.closed)


# ── 5: the same setup is not re-entered after a close ──────────────────────

def test_signal_not_reentered_after_close(tmp_path):
    import json
    broker = FakeBroker(price=50.64)
    ex = make_executor(broker, tmp_path)

    sig = {
        "symbol": "BAC", "direction": "short",
        "entry_price": 50.765, "stop_price": 50.8224, "target_price": 50.6501,
        "atr": 0.1149, "risk_dollars": 100.0, "max_risk_mult": 2.0,
        "bar_date": "2026-10-22", "bar_time_et": "09:35",
        "timestamp": pd.Timestamp.now().isoformat(), "flat_time_et": "14:00",
    }
    ex._open_position(sig, pd.Timestamp.now(), "BAC|2026-10-22|09:35")
    assert "BAC|2026-10-22|09:35" in ex._acted_signals

    # Simulate the position having closed.
    ex.positions.pop("BAC", None)

    # A second poll must NOT place a second order for the same setup.
    ex.signals_file.write_text(json.dumps({"signals": [sig]}))
    n_before = len(broker.submitted)
    ex._process_signals()
    assert len(broker.submitted) == n_before, \
        "the same signal was entered twice"


def test_acted_signals_persist_across_restart(tmp_path):
    """An executor restart mid-session must not re-enter a setup it
    already traded — the intraday scanner re-derives the same signal
    on every poll."""
    import json
    broker = FakeBroker(price=50.64)
    ex = make_executor(broker, tmp_path)

    sig = {
        "symbol": "BAC", "direction": "short",
        "entry_price": 50.765, "stop_price": 50.8224, "target_price": 50.6501,
        "atr": 0.1149, "risk_dollars": 100.0, "max_risk_mult": 2.0,
        "bar_date": "2026-10-22", "bar_time_et": "09:35",
        "timestamp": pd.Timestamp.now().isoformat(), "flat_time_et": "14:00",
    }
    ex._open_position(sig, pd.Timestamp.now(), "BAC|2026-10-22|09:35")
    assert ex._acted_file.exists(), "acted-signal file was not written"

    # A brand-new executor (as after a daemon restart) must load it.
    ex2 = make_executor(broker, tmp_path)
    assert "BAC|2026-10-22|09:35" in ex2._acted_signals, \
        "acted signals were not restored on restart"

    ex2.positions.pop("BAC", None)
    ex2.signals_file.write_text(json.dumps({"signals": [sig]}))
    n_before = len(broker.submitted)
    ex2._process_signals()
    assert len(broker.submitted) == n_before, \
        "the same signal was re-entered after a restart"


def test_order_status_normalises_enum(tmp_path):
    """alpaca-py returns an OrderStatus enum whose str() is
    "OrderStatus.FILLED", not "filled". A naive comparison never
    matches, so a filled order was never recognised."""
    from src.trading.executor_live import PaperExecutor

    class FakeEnum:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return f"OrderStatus.{self.value.upper()}"

    o = type("O", (), {"status": FakeEnum("filled")})()
    assert PaperExecutor._order_status(o) == "filled"

    o2 = type("O", (), {"status": "partially_filled"})()
    assert PaperExecutor._order_status(o2) == "partially_filled"

    o3 = type("O", (), {"status": FakeEnum("new")})()
    assert PaperExecutor._order_status(o3) == "new"


def test_get_order_survives_alpaca_get_orders_signature(tmp_path):
    """alpaca-py's get_orders() takes NO keyword args. Calling it as
    get_orders(status='all', limit=50) raises TypeError, which the old
    bare `except Exception` swallowed — so every fill check silently
    returned False and no position was ever managed."""
    import inspect
    from src.trading.executor_live import PaperExecutor

    src = inspect.getsource(PaperExecutor._get_order)
    # Strip comments/docstrings so the prose in the docstring does not
    # trip the check — we care about the actual call expression.
    code = "\n".join(
        line for line in src.splitlines()
        if not line.strip().startswith("#")
    )
    # Remove the triple-quoted docstring block.
    code = re.sub(r'"""(?:.|\n)*?"""', "", code)
    assert "get_orders(status=" not in code, \
        "get_orders() must not be called with keyword arguments"
    assert "GetOrdersRequest" in code, \
        "must use the request-object API"


def test_reconcile_flags_foreign_broker_position(tmp_path):
    """A position at the broker that this executor did not open (e.g.
    from another strategy sharing the account) must be flagged — it has
    no stop, no target and no deadline."""
    import logging
    broker = FakeBroker(price=50.64)
    ex = make_executor(broker, tmp_path)

    class P:
        def __init__(self, sym, qty):
            self.symbol = sym
            self.qty = qty
    broker.client.get_all_positions = lambda: [P("NFLX", 10.0)]

    with caplog_at("WARNING"):
        ex._reconcile_with_broker()

    assert "NFLX" in ex._foreign_positions_seen, \
        "a foreign broker position was not flagged"


class _Handler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def caplog_at(level):
    import logging
    h = _Handler()
    root = logging.getLogger()
    root.addHandler(h)
    old = root.level
    root.setLevel(level)
    return _Ctx(h, root, old)


class _Ctx:
    def __init__(self, handler, root, old_level):
        self.h = handler
        self.root = root
        self.old = old_level

    def __enter__(self):
        return self.h

    def __exit__(self, *a):
        self.root.removeHandler(self.h)
        self.root.setLevel(self.old)
        return False


def test_open_positions_survive_restart(tmp_path):
    """A daemon restart must re-adopt the positions it already opened.

    Otherwise the positions sit at the broker with no stop, no target
    and no deadline — an unbounded risk — and the restarted process
    mis-classifies them as belonging to another strategy."""
    broker = FakeBroker(price=50.64)
    ex = make_executor(broker, tmp_path)

    sig = {
        "symbol": "BAC", "direction": "short",
        "entry_price": 50.765, "stop_price": 50.8224, "target_price": 50.6501,
        "atr": 0.1149, "risk_dollars": 100.0, "max_risk_mult": 2.0,
        "bar_date": "2026-10-22", "bar_time_et": "09:35",
        "timestamp": pd.Timestamp.now().isoformat(), "flat_time_et": "14:00",
    }
    ex._open_position(sig, pd.Timestamp.now(), "BAC|2026-10-22|09:35")
    assert ex._positions_file.exists(), "position file was not written"

    ex2 = make_executor(broker, tmp_path)
    assert "BAC" in ex2.positions, \
        "open position was not re-adopted after a restart"
    p = ex2.positions["BAC"]
    assert p.direction == "short"
    # Levels are stored at Alpaca's penny increment (2dp), so compare
    # against the rounded value rather than the raw signal precision.
    assert abs(p.stop_price - round(50.8224, 2)) < 1e-9, p.stop_price
    assert abs(p.target_price - round(50.6501, 2)) < 1e-9, p.target_price
    assert p.status == "pending"


if __name__ == "__main__":
    import tempfile
    failures = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(td)
                print(f"PASS {fn.__name__}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {fn.__name__}: {e}")
            except Exception as e:
                failures += 1
                print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
