"""
Regression tests for the engulfing live paper-trading fixes (2026-11-10).

Covers the six defects found in the end-to-end audit of the engulfing
paper-trading pipeline:

  1. The runner had LOST its SIGTERM/SIGINT handler (dropped in commit
     7aa7b4c). systemd stops the unit with SIGTERM, so the daemon was
     killed outright: no trade log written, no position flattened, no
     protective stop cancelled.

  2. `_close_all()` passed `pos.entry_price` as the EXIT price, so a
     forced close booked the position as exiting at its entry proxy —
     a guaranteed zero-P&L trade, or a fictitious gain/loss when the
     proxy differed from the real fill.

  3. The exit price was never read back from the broker. The ledger
     must use the price the broker ACTUALLY executed at.

  4. No broker-side protective stop ever existed. Stops lived only in
     the Python polling loop, so a daemon crash left an overnight
     position completely unprotected.

  5. A partial/extra fill left ORPHANED shares at the broker: the
     position tracked 1.00 share while the broker held 3.91, so the
     close sold 1 share and left 2.91 naked.

  6. The daily wrapper restarted the daemon unconditionally. For an
     OVERNIGHT strategy that force-closed the position the backtest
     expects to be held across the close.
"""
import inspect
import json
import re
import sys
from pathlib import Path
from threading import Event

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.trading.executor_live import PaperExecutor  # noqa: E402


# ── fakes ───────────────────────────────────────────────────────────────────

class FakeOrder:
    def __init__(self, oid, symbol, status, filled_qty=0.0,
                 filled_avg_price=None):
        self.id = oid
        self.symbol = symbol
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price


class FakeClient:
    def __init__(self, orders=None, positions=None):
        self.orders = orders or []
        self.positions = positions or {}

    def get_orders(self, request=None):
        return list(self.orders)

    def get_all_positions(self):
        class P:
            def __init__(self, symbol, qty):
                self.symbol = symbol
                self.qty = qty
        return [P(s, q) for s, q in self.positions.items()]


class FakeBroker:
    """Broker fake that records stop orders and close-order fill prices.

    `close_fill_price` is the price the broker ACTUALLY executes the
    close at. It is deliberately different from `price` (the quote) so
    a test can prove the ledger used the FILL and not the quote.
    """

    def __init__(self, orders=None, price=100.0, order_status="filled",
                 filled_qty=0.0, filled_avg_price=None,
                 close_fill_price=None, close_fill_status="filled",
                 positions=None, prices=None):
        self.client = FakeClient(orders, positions)
        self.price = price
        self.prices = prices or {}
        self.order_status = order_status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.close_fill_price = close_fill_price
        self.close_fill_status = close_fill_status
        self.submitted = []
        self.stops = []
        self.cancelled = []
        self.closed = []
        self._next_id = 0

    def _record_order(self, symbol, qty, status=None, filled_qty=None,
                      filled_avg_price=None):
        self._next_id += 1
        oid = f"ord-{self._next_id}"
        self.client.orders.append(
            FakeOrder(oid, symbol,
                      status or self.order_status,
                      filled_qty=self.filled_qty if filled_qty is None else filled_qty,
                      filled_avg_price=(self.filled_avg_price
                                        if filled_avg_price is None
                                        else filled_avg_price)))
        return oid

    def submit_market_order(self, symbol, qty, side):
        self.submitted.append((symbol, qty, side, "market"))
        return self._record_order(symbol, qty)

    def submit_limit_order(self, symbol, qty, limit_price, side):
        self.submitted.append((symbol, qty, side, "limit"))
        return self._record_order(symbol, qty)

    def submit_stop_order(self, symbol, qty, stop_price, side):
        self.stops.append((symbol, qty, stop_price, side))
        return self._record_order(symbol, qty, status="new")

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        for o in self.client.orders:
            if o.id == order_id:
                o.status = "cancelled"

    def close_position(self, symbol, qty=None):
        self.closed.append((symbol, qty))
        # The close order is a market order that fills at the broker's
        # actual execution price.
        return self._record_order(
            symbol, qty, status=self.close_fill_status,
            filled_qty=(qty or 0),
            filled_avg_price=self.close_fill_price)

    def get_positions(self):
        return dict(self.client.positions)

    def get_cash(self):
        return 100_000.0

    def get_current_price(self, symbol):
        return self.prices.get(symbol, self.price)

    def market_hours_open(self):
        return True


def make_executor(broker, tmp_path, strategy="engulfing",
                  close_mode="overnight"):
    ex = PaperExecutor.__new__(PaperExecutor)
    ex.broker = broker
    ex.market = "US"
    ex.close_mode = close_mode
    ex.strategy = strategy
    ex.capital = 100_000.0
    ex.positions = {}
    ex.trade_log = []
    ex._bar_count = 0
    ex._acted_signals = set()
    ex._acted_file = Path(tmp_path) / f".acted_{strategy}.json"
    ex._positions_file = Path(tmp_path) / f".positions_{strategy}.json"
    ex._foreign_positions_seen = set()
    ex._load_acted_signals()
    ex._load_positions()
    ex._stop = Event()
    ex.signals_file = Path(tmp_path) / "signals.json"
    ex.log_file = Path(tmp_path) / "trades.csv"
    ex.poll_seconds = 60
    return ex


def make_signal(sym="META", direction="long", entry=100.0,
                stop=95.0, target=110.0, bar_date="2026-11-06"):
    return {
        "symbol": sym, "direction": direction,
        "entry_price": entry, "stop_price": stop, "target_price": target,
        "atr": 5.0, "risk_dollars": 100.0, "max_risk_mult": 2.0,
        "bar_date": bar_date,
        "timestamp": pd.Timestamp.now().isoformat(),
    }


def enter_position(ex, broker, sym="META", direction="long", entry=100.0,
                   stop=95.0, target=110.0, qty=1.0, fill_price=None):
    """Put a position into the 'entered' state directly.

    The position is built directly rather than via `_open_position`,
    because `_open_position` refuses to open a symbol that is already
    held at the broker — which is exactly the situation the orphaned-
    remainder tests need to set up.
    """
    from src.trading.executor_live import ActivePosition
    sig = make_signal(sym=sym, direction=direction, entry=entry,
                      stop=stop, target=target)
    now = pd.Timestamp.now()
    ex.positions[sym] = ActivePosition(
        symbol=sym,
        direction=direction,
        entry_price=round(entry, 2),
        stop_price=round(stop, 2),
        target_price=round(target, 2),
        atr=5.0,
        qty=qty,
        signal_time=now,
        entry_time=now,
        order_id="entry-test",
        status="entered",
        bar_date=sig["bar_date"],
        close_mode=ex.close_mode,
        close_deadline_ts=now + pd.Timedelta(days=1),
        entry_filled_price=fill_price if fill_price is not None else entry,
    )
    return ex.positions[sym]


def _code_only(src: str) -> str:
    """Strip comments and docstrings so a source check is not fooled by
    prose describing the very bug it is checking for."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    out = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


# ── 1: the runner must handle SIGTERM ───────────────────────────────────────

def test_runner_handles_sigterm(tmp_path=None):
    """The engulfing runner MUST install a SIGTERM/SIGINT handler.

    systemd stops the unit with SIGTERM. Without a handler the process
    is killed outright: the trade log is not written and open positions
    are left untouched at the broker. The handler was dropped in commit
    7aa7b4c and must not be dropped again.
    """
    src = (ROOT / "scripts" / "run_engulfing_paper.py").read_text()

    assert "signal.signal(signal.SIGTERM" in src, \
        "the runner does not handle SIGTERM — systemd kills it outright"
    assert "signal.signal(signal.SIGINT" in src, \
        "the runner does not handle SIGINT"
    # The handler must stop the executor, which is what triggers the
    # graceful shutdown (write the log, keep/flatten positions).
    assert "executor.stop()" in src, \
        "the signal handler does not stop the executor"


def test_runner_sigterm_handler_calls_stop(tmp_path=None):
    """A source-level check that the handler is wired to executor.stop()."""
    src = (ROOT / "scripts" / "run_engulfing_paper.py").read_text()
    idx = src.index("def _sig_handler")
    body = src[idx:idx + 600]
    assert "executor.stop()" in body, \
        "the SIGTERM handler does not stop the executor"


# ── 2: force_close must NOT use the entry proxy as the exit price ─────────

def test_close_all_does_not_use_entry_price_as_exit(tmp_path):
    """`_close_all` must not pass `pos.entry_price` as the exit price.

    That bug made every forced close book a zero-P&L trade at the entry
    proxy — and, when the proxy differed from the real fill, a
    fictitious gain or loss.
    """
    src = _code_only(inspect.getsource(PaperExecutor._close_all))
    assert "pos.entry_price" not in src, \
        "_close_all still passes the entry proxy as the exit price"
    assert '"force_close", None' in src, \
        "force_close must pass exit_price=None so the broker fill is used"


def test_force_close_uses_broker_fill_not_entry_proxy(tmp_path):
    """A forced close must book the broker's ACTUAL fill price.

    Setup: entered at a 621.03 proxy, but the broker really filled the
    entry at 642.842 and the close at 629.756. The old code produced a
    -340.8 bps "loss" from the proxies; the correct figure from the
    fills is about -203.6 bps.
    """
    broker = FakeBroker(price=629.756, close_fill_price=629.756)
    ex = make_executor(broker, tmp_path)

    enter_position(ex, broker, sym="META", entry=621.03, stop=600.0,
                   target=660.0, qty=0.32205, fill_price=642.842)

    ex._close_all()

    assert len(ex.trade_log) == 1, "the forced close was not recorded"
    row = ex.trade_log[0]
    assert row["exit_reason"] == "force_close"
    # The exit price must be the broker's fill, NOT the entry proxy.
    assert row["exit_price"] == 629.756, \
        f"exit price was {row['exit_price']}, expected the broker fill 629.756"
    assert row["exit_price"] != 621.03, \
        "the entry proxy was booked as the exit price"
    assert row["exit_price_source"] == "broker_fill"


def test_force_close_does_not_book_a_fictitious_profit(tmp_path):
    """Regression on the 2026-11-04 META trade.

    The old code exited at the entry proxy, which for a force_close
    produced a 0% trade, and in the META case produced a LOSS larger
    than the broker's actual loss. The ledger must reflect the fills.
    """
    broker = FakeBroker(price=629.756, close_fill_price=629.756)
    ex = make_executor(broker, tmp_path)

    enter_position(ex, broker, sym="META", entry=621.03, stop=600.0,
                   target=660.0, qty=0.32205, fill_price=642.842)
    ex._close_all()

    row = ex.trade_log[0]
    expected_bps = (629.756 - 642.842) / 642.842 * 10_000 - 1.5
    assert abs(row["pnl_bps"] - expected_bps) < 0.5, \
        f"pnl_bps {row['pnl_bps']} != expected {expected_bps:.2f}"


# ── 3: the exit price must come from the broker's fill ─────────────────────

def test_close_uses_broker_fill_not_market_quote(tmp_path):
    """The exit price is the broker's FILL, not the last quote.

    A market quote and the fill differ, and the fill is what actually
    happened.
    """
    broker = FakeBroker(price=101.50, close_fill_price=100.25)
    ex = make_executor(broker, tmp_path)

    enter_position(ex, broker, sym="META", entry=100.0, stop=95.0,
                   target=110.0, qty=2.0, fill_price=100.0)
    ex._close_position("META", "target", 110.0)

    row = ex.trade_log[0]
    assert row["exit_price"] == 100.25, \
        f"exit price {row['exit_price']} != broker fill 100.25"


def test_close_falls_back_to_quote_when_broker_has_no_fill(tmp_path):
    """If the broker never reports a fill, the quote is used — but the
    row is LABELLED as a quote so the audit can see it is not a fill."""
    broker = FakeBroker(price=101.50, close_fill_price=None,
                        close_fill_status="new")
    ex = make_executor(broker, tmp_path)

    enter_position(ex, broker, sym="META", entry=100.0, stop=95.0,
                   target=110.0, qty=2.0, fill_price=100.0)
    ex._close_position("META", "target", 110.0)

    row = ex.trade_log[0]
    assert row["exit_price_source"] == "quote", \
        "a non-fill exit price must be labelled 'quote'"
    assert row["exit_price"] == 110.0


def test_ledger_records_exit_price_source(tmp_path):
    """Every closed row must record where its exit price came from."""
    broker = FakeBroker(price=101.50, close_fill_price=100.25)
    ex = make_executor(broker, tmp_path)

    enter_position(ex, broker, sym="META", entry=100.0, stop=95.0,
                   target=110.0, qty=2.0, fill_price=100.0)
    ex._close_position("META", "target", 110.0)

    assert "exit_price_source" in ex.trade_log[0]


# ── 4: a broker-side protective stop must exist ────────────────────────────

def test_broker_backend_implements_submit_stop_order(tmp_path=None):
    """The Alpaca backend must implement the broker-side stop."""
    from src.trading.executor_live import AlpacaBackend, BrokerBackend
    assert hasattr(AlpacaBackend, "submit_stop_order"), \
        "the Alpaca backend cannot place a protective stop"
    assert hasattr(BrokerBackend, "submit_stop_order"), \
        "the broker interface has no stop-order method"
    # The Alpaca implementation must not be the ABC's raising stub.
    assert (AlpacaBackend.submit_stop_order is not
            BrokerBackend.submit_stop_order), \
        "the Alpaca backend does not actually implement submit_stop_order"
    # The executor must call it.
    src = _code_only(inspect.getsource(PaperExecutor._place_protective_stop))
    assert "submit_stop_order" in src, \
        "the executor never places a broker-side protective stop"


def test_protective_stop_placed_on_fill(tmp_path):
    """When a position fills, a protective stop MUST rest at the broker.

    Without it, a daemon crash leaves the position naked — which is
    exactly what happened to every overnight position before this fix.
    """
    broker = FakeBroker(price=100.0, order_status="filled",
                        filled_qty=2.0, filled_avg_price=100.0)
    ex = make_executor(broker, tmp_path)

    sig = make_signal(sym="META", entry=100.0, stop=95.0, target=110.0)
    ex._open_position(sig, pd.Timestamp.now(), "META|test")
    ex._monitor_positions()

    assert ex.positions["META"].status == "entered"
    assert broker.stops, "no protective stop was placed at the broker"
    sym, qty, stop_px, side = broker.stops[0]
    assert sym == "META"
    assert qty == 2.0, "the stop must cover the FILLED quantity"
    assert stop_px == 95.0
    assert side == "sell", "a long position is protected by a SELL stop"


def test_protective_stop_is_cancelled_when_position_closes(tmp_path):
    """The stop must be cancelled when the position is closed.

    Leaving it working would let it fire against a position that no
    longer exists — opening a naked SHORT.
    """
    broker = FakeBroker(price=110.0, close_fill_price=110.0)
    ex = make_executor(broker, tmp_path)

    enter_position(ex, broker, sym="META", entry=100.0, stop=95.0,
                   target=110.0, qty=2.0, fill_price=100.0)
    ex._place_protective_stop("META", ex.positions["META"])
    stop_id = ex.positions["META"].stop_order_id
    assert stop_id is not None

    ex._close_position("META", "target", 110.0)
    assert stop_id in broker.cancelled, \
        "the protective stop was left working after the position closed"


def test_protective_stop_side_is_buy_for_a_short(tmp_path):
    """A short position is protected by a BUY stop above the entry."""
    broker = FakeBroker(price=100.0)
    ex = make_executor(broker, tmp_path)

    pos = enter_position(ex, broker, sym="META", direction="short",
                          entry=100.0, stop=105.0, target=90.0,
                          qty=2.0, fill_price=100.0)
    ex._place_protective_stop("META", pos)

    assert broker.stops, "no protective stop placed for a short"
    assert broker.stops[0][3] == "buy", \
        "a short position must be protected by a BUY stop"


def test_protective_stop_resized_after_a_partial_fill(tmp_path):
    """A partially filled entry must end up with a stop for the FULL size.

    Otherwise the extra shares are held with no protection.
    """
    broker = FakeBroker(price=100.0, order_status="partially_filled",
                        filled_qty=1.0, filled_avg_price=100.0)
    ex = make_executor(broker, tmp_path)

    sig = make_signal(sym="META", entry=100.0, stop=95.0, target=110.0)
    ex._open_position(sig, pd.Timestamp.now(), "META|test")
    ex._monitor_positions()

    pos = ex.positions["META"]
    assert pos.qty == 1.0
    assert broker.stops[-1][1] == 1.0

    # The order fills further on a later poll.
    for o in broker.client.orders:
        if o.id == pos.order_id:
            o.status = "partially_filled"
            o.filled_qty = 3.0
    ex._monitor_positions()

    assert ex.positions["META"].qty == 3.0
    assert broker.stops[-1][1] == 3.0, \
        "the protective stop was not resized to cover the full fill"


def test_protective_stop_is_rearmed_on_restart(tmp_path):
    """A restored position must get a working stop again at start-up."""
    broker = FakeBroker(price=100.0, order_status="filled",
                        filled_qty=2.0, filled_avg_price=100.0)
    ex = make_executor(broker, tmp_path)

    enter_position(ex, broker, sym="META", entry=100.0, stop=95.0,
                   target=110.0, qty=2.0, fill_price=100.0)
    ex._save_positions()

    # A fresh executor re-adopts the position from disk.
    ex2 = make_executor(broker, tmp_path)
    assert "META" in ex2.positions

    before = len(broker.stops)
    ex2._rearm_protective_stops()
    assert len(broker.stops) > before, \
        "the restored position did not get a protective stop re-armed"
    assert ex2.positions["META"].stop_order_id is not None


def test_broker_side_stop_fill_is_recorded_without_a_second_close(tmp_path):
    """If the broker-side stop fires, the trade is recorded but the
    executor must NOT submit another close order."""
    broker = FakeBroker(price=94.0, close_fill_price=94.0)
    ex = make_executor(broker, tmp_path)

    enter_position(ex, broker, sym="META", entry=100.0, stop=95.0,
                   target=110.0, qty=2.0, fill_price=100.0)
    ex._place_protective_stop("META", ex.positions["META"])
    stop_id = ex.positions["META"].stop_order_id

    # The stop fills at the broker.
    for o in broker.client.orders:
        if o.id == stop_id:
            o.status = "filled"
            o.filled_qty = 2.0
            o.filled_avg_price = 94.5

    n_closed_before = len(broker.closed)
    ex._monitor_positions()

    assert len(ex.trade_log) == 1, "the broker-side stop exit was not recorded"
    assert ex.trade_log[0]["exit_reason"] == "stop"
    assert ex.trade_log[0]["exit_price"] == 94.5, \
        "the broker-side stop fill price was not used"
    assert len(broker.closed) == n_closed_before, \
        "a duplicate close order was submitted for an already-closed position"


# ── 5: orphaned remainder shares must be adopted ───────────────────────────

def test_orphaned_remainder_shares_are_adopted(tmp_path):
    """Broker holds MORE of our symbol than we track → adopt the excess.

    This is the 2026-10-30 BAC case: the entry filled 3.91 shares but
    the position tracked 1.00, so the close sold 1 share and left
    2.91 naked at the broker.
    """
    broker = FakeBroker(price=50.09, positions={"BAC": 3.91})
    ex = make_executor(broker, tmp_path)

    enter_position(ex, broker, sym="BAC", entry=50.50, stop=51.0,
                   target=49.0, qty=1.0, fill_price=50.5047)

    ex._reconcile_with_broker()

    assert ex.positions["BAC"].qty == 3.91, \
        "the orphaned remainder was not adopted"
    assert broker.stops, "the adopted shares are not protected by a stop"
    assert broker.stops[-1][1] == 3.91, \
        "the protective stop does not cover the adopted shares"


def test_reconcile_does_not_close_foreign_positions(tmp_path):
    """A position from another strategy must be flagged, NOT closed.

    The Alpaca paper account is shared.
    """
    broker = FakeBroker(price=50.0, positions={"NFLX": 10.0})
    ex = make_executor(broker, tmp_path)

    ex._reconcile_with_broker()

    assert "NFLX" in ex._foreign_positions_seen
    assert not broker.closed, \
        "a foreign strategy's position was closed by this executor"
    assert "NFLX" not in ex.positions


def test_reconcile_ignores_positions_we_hold_less_of(tmp_path):
    """If the broker holds FEWER than we track, that is not an orphan."""
    broker = FakeBroker(price=50.0, positions={"BAC": 0.5})
    ex = make_executor(broker, tmp_path)

    enter_position(ex, broker, sym="BAC", entry=50.50, stop=51.0,
                   target=49.0, qty=1.0, fill_price=50.5047)

    ex._reconcile_with_broker()

    assert ex.positions["BAC"].qty == 1.0, \
        "the tracked quantity was reduced by a smaller broker holding"


# ── 6: an overnight position must survive a daemon restart ─────────────────

def test_close_all_can_leave_overnight_positions_open(tmp_path):
    """An overnight position must NOT be flattened by a routine stop."""
    broker = FakeBroker(price=100.0, close_fill_price=100.0)
    ex = make_executor(broker, tmp_path, close_mode="overnight")

    enter_position(ex, broker, sym="META", entry=100.0, stop=95.0,
                   target=110.0, qty=2.0, fill_price=100.0)

    ex._close_all(flatten=False)

    assert not broker.closed, \
        "the overnight position was flattened by a routine shutdown"
    assert ex.positions["META"].status == "entered", \
        "the overnight position was retired by a routine shutdown"


def test_intraday_positions_are_still_flattened(tmp_path):
    """The intraday strategy must still be flat by the close."""
    broker = FakeBroker(price=100.0, close_fill_price=100.0)
    ex = make_executor(broker, tmp_path, close_mode="intraday")

    enter_position(ex, broker, sym="META", entry=100.0, stop=95.0,
                   target=110.0, qty=2.0, fill_price=100.0)

    ex._close_all(flatten=True)

    assert broker.closed, "the intraday position was not flattened"
    assert ex.positions["META"].status == "exited"


def test_run_uses_flatten_false_for_overnight(tmp_path):
    """The run loop must not flatten overnight positions on exit."""
    src = inspect.getsource(PaperExecutor.run)
    assert "flatten_on_exit" in src, \
        "run() does not distinguish overnight from intraday shutdown"
    assert "self.close_mode != \"overnight\"" in src, \
        "run() does not treat the overnight close mode specially"


def test_wrapper_does_not_restart_a_daemon_holding_a_position(tmp_path=None):
    """The daily wrapper must not restart the daemon mid-position.

    For an overnight strategy the daemon is supposed to be alive across
    the close, holding the position.
    """
    src = (ROOT / "scripts" / "engulfing_daily_wrapper.sh").read_text()
    assert "HOLDING" in src, \
        "the wrapper does not check for an open position before restarting"
    assert "NOT restarting" in src, \
        "the wrapper restarts the daemon unconditionally"


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
