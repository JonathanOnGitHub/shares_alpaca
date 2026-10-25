"""
Tests for the FVG live-trading reliability fixes (2026-10-25 audit).

The bugs these cover:

  1. The trade log was only written on process exit, so a daemon that
     was killed or wedged at the close left every trade of that session
     unrecorded — the daily report showed "0 trades / $0.00".
  2. The trade log was OVERWRITTEN with the in-memory list only, so a
     restart mid-session wiped the trades already recorded.
  3. A pending position whose entry order had been cancelled sat in
     "pending" forever: it blocked the "session over and flat" exit,
     wedged the daemon across the close into the next session, and was
     re-adopted as a phantom position on the next restart.
  4. The capital update applied the PRICE MOVE IN BPS to the whole
     account instead of the actual dollar P&L — a ~50x overstatement
     of every loss.
  5. The P&L report read columns the log never wrote ("pnl" in dollars,
     "status", "exit_date"), so it reported zero trades and $0.00 even
     when trades existed.
  6. Intraday entries used LIMIT orders, so only signals that retraced
     to the entry price were traded — a selection bias against the
     backtest's "enter at the signal bar's close" rule.
"""
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.trading.executor_live import PaperExecutor  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from test_executor_fills import FakeBroker, make_executor  # noqa: E402


def _sig(symbol="BAC", direction="short", entry=50.60, stop=50.69,
         target=50.42, bar_date="2026-10-23"):
    return {
        "symbol": symbol, "direction": direction,
        "entry_price": entry, "stop_price": stop, "target_price": target,
        "atr": 0.14, "risk_dollars": 100.0, "max_risk_mult": 2.0,
        "bar_date": bar_date, "bar_time_et": "09:35",
        "timestamp": pd.Timestamp.now().isoformat(), "flat_time_et": "14:00",
    }


def _broker(price, symbol="BAC", prices=None, **kw):
    """
    A broker whose quoted price sits INSIDE the stop band.

    The executor refuses to enter when the market has already gapped
    through the stop, so a short needs a price BELOW its stop and a
    long needs one ABOVE its stop.
    """
    kw.setdefault("order_status", "filled")
    kw.setdefault("filled_qty", 4.0)
    kw.setdefault("filled_avg_price", 50.60)
    price_map = {symbol: price}
    price_map.update(prices or {})
    return FakeBroker(price=price, prices=price_map, **kw)


# ── 1: the trade log is written IMMEDIATELY on close, not only at exit ──────

def test_trade_log_written_immediately_on_close(tmp_path):
    broker = _broker(50.60)
    ex = make_executor(broker, tmp_path)

    ex._open_position(_sig(), pd.Timestamp.now(), "BAC|2026-10-23|09:35")
    ex._monitor_positions()
    assert ex.positions["BAC"].status == "entered"

    ex._close_position("BAC", "stop", 50.69)

    # The log must exist NOW, before any process exit.
    assert ex.log_file.exists(), "trade log was not written on close"
    df = pd.read_csv(ex.log_file)
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "BAC"
    assert df.iloc[0]["exit_reason"] == "stop"


# ── 2: the log MERGES with existing rows instead of overwriting ────────────

def test_trade_log_merges_with_existing_rows(tmp_path):
    broker = _broker(50.60)
    ex = make_executor(broker, tmp_path)

    # A first executor writes a trade.
    ex._open_position(_sig("BAC"), pd.Timestamp.now(), "BAC|2026-10-23|09:35")
    ex._monitor_positions()
    ex._close_position("BAC", "stop", 50.69)
    assert len(pd.read_csv(ex.log_file)) == 1

    # A second executor (restart) has an EMPTY in-memory log but the file
    # already holds a row. Its own trade must be ADDED, not replace it.
    # XOM is a LONG, so it needs a price ABOVE its stop (179.60).
    ex2 = make_executor(_broker(50.60, prices={"XOM": 180.00}), tmp_path)
    ex2._open_position(_sig("XOM", direction="long", entry=179.88,
                            stop=179.60, target=180.45),
                       pd.Timestamp.now(), "XOM|2026-10-23|09:35")
    ex2._monitor_positions()
    ex2._close_position("XOM", "target", 180.45)

    df = pd.read_csv(ex.log_file)
    assert len(df) == 2, f"restart wiped the existing trade row: {df}"
    assert set(df["symbol"]) == {"BAC", "XOM"}


def test_trade_log_is_idempotent(tmp_path):
    """Writing the same trade twice must not duplicate it."""
    broker = _broker(50.60)
    ex = make_executor(broker, tmp_path)

    ex._open_position(_sig(), pd.Timestamp.now(), "BAC|2026-10-23|09:35")
    ex._monitor_positions()
    ex._close_position("BAC", "stop", 50.69)

    # Re-write the same in-memory log.
    ex._write_log()
    ex._write_log()

    df = pd.read_csv(ex.log_file)
    assert len(df) == 1, f"trade was duplicated: {len(df)} rows"


# ── 3: the log schema matches what the P&L report reads ────────────────────

def test_trade_log_schema_matches_pnl_report(tmp_path):
    broker = _broker(50.60)
    ex = make_executor(broker, tmp_path)

    ex._open_position(_sig(), pd.Timestamp.now(), "BAC|2026-10-23|09:35")
    ex._monitor_positions()
    ex._close_position("BAC", "stop", 50.69)

    with open(ex.log_file) as f:
        row = list(csv.DictReader(f))[0]

    # Every field the report's calc_pnl() touches.
    assert "pnl" in row, "report reads 'pnl' but the log does not write it"
    assert "status" in row, "report reads 'status' but the log does not write it"
    assert "exit_date" in row, "report reads 'exit_date' but the log does not write it"
    assert "pnl_bps" in row

    # The values must be sane: a short from 50.60 closed at 50.69 lost money.
    assert float(row["pnl"]) < 0, "a losing short booked a profit"
    assert float(row["pnl_bps"]) < 0
    assert row["status"] == "closed"


def test_pnl_report_reads_the_log(tmp_path):
    """The report's calc_pnl() must count a trade written by the executor."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    report = importlib.import_module("fvg_pnl_report")

    broker = _broker(50.60)
    ex = make_executor(broker, tmp_path)

    ex._open_position(_sig(), pd.Timestamp.now(), "BAC|2026-10-23|09:35")
    ex._monitor_positions()
    ex._close_position("BAC", "stop", 50.69)

    trades = list(csv.DictReader(open(ex.log_file)))
    pnl = report.calc_pnl(trades)

    assert pnl["closed_trades"] == 1, f"report counted 0 closed trades: {pnl}"
    assert pnl["losses"] == 1
    assert pnl["total_pnl"] < 0
    assert pnl["total_bps"] < 0


# ── 4: capital is updated by DOLLARS, not by bps applied to the account ────

def test_capital_moves_by_dollars_not_bps_of_account(tmp_path):
    """
    The old code did `capital *= (1 + pnl_bps/10_000)`.

    # A 4-share short at 50.60 (notional $202) closed at 50.69 loses
    # ~17.8 bps of PRICE, i.e. about $0.36 of DOLLARS. The old code
    # marked the whole $10,000 account down by $17.80.
    """
    broker = _broker(50.60)
    ex = make_executor(broker, tmp_path)

    ex._open_position(_sig(), pd.Timestamp.now(), "BAC|2026-10-23|09:35")
    ex._monitor_positions()
    ex._close_position("BAC", "stop", 50.69)

    # -$0.09 of price loss plus ~$0.30 of round-trip commission.
    expected_dollars = (50.60 - 50.69) * 4 - 1.5 / 10_000 * (50.60 * 4)
    assert expected_dollars < 0, "fixture does not describe a loss"
    assert ex.capital < 10_000.0
    assert abs((ex.capital - 10_000.0) - expected_dollars) < 0.02, (
        f"capital moved by {ex.capital - 10_000.0:.4f}, "
        f"expected {expected_dollars:.4f}"
    )
    # The old bug marked the account down by ~$17.80 instead of ~$0.39.
    assert abs((ex.capital - 10_000.0)) < 1.0, \
        f"capital moved by {ex.capital - 10_000.0:.2f} — bps applied to account"


# ── 5: a dead pending order is retired, not left to wedge the daemon ───────

def test_dead_pending_order_is_retired(tmp_path):
    broker = _broker(50.60, order_status="cancelled")
    ex = make_executor(broker, tmp_path)

    ex._open_position(_sig(), pd.Timestamp.now(), "BAC|2026-10-23|09:35")
    pos = ex.positions["BAC"]
    assert pos.status == "pending"

    ex._monitor_positions()

    assert pos.status == "exited", "dead pending position was not retired"
    assert pos.exit_reason == "unfilled"
    # The position file must no longer carry the phantom position.
    assert "BAC" not in ex._positions_file.read_text()


def test_retired_pending_position_is_logged_as_unfilled(tmp_path):
    broker = _broker(50.60, order_status="cancelled")
    ex = make_executor(broker, tmp_path)

    ex._open_position(_sig(), pd.Timestamp.now(), "BAC|2026-10-23|09:35")
    ex._monitor_positions()

    df = pd.read_csv(ex.log_file)
    assert len(df) == 1
    assert df.iloc[0]["status"] == "unfilled"
    assert float(df.iloc[0]["pnl"]) == 0.0


def test_live_pending_order_is_not_retired(tmp_path):
    """A working order must NOT be retired — it may still fill."""
    broker = _broker(50.60, order_status="new")
    ex = make_executor(broker, tmp_path)

    ex._open_position(_sig(), pd.Timestamp.now(), "BAC|2026-10-23|09:35")
    ex._monitor_positions()

    assert ex.positions["BAC"].status == "pending", \
        "a working order was retired prematurely"


# ── 6: the daemon exits when the session ends, even with a stale position ──

def test_session_ends_even_with_open_position(tmp_path):
    """
    The old `_session_ended()` required an EMPTY position book. A single
    pending position whose order had been cancelled therefore wedged the
    daemon forever: it never exited, and the next day's wrapper reused
    the stale process, so the strategy silently stopped trading.
    """
    ex = make_executor(FakeBroker(), tmp_path)

    # A stale pending position that was never filled.
    ex.positions["BAC"] = type("P", (), {
        "status": "pending", "close_deadline_ts": None})()

    # Freeze the clock at 16:30 ET (after the close).
    ex._now_override = pd.Timestamp.now("America/New_York").normalize() + \
        pd.Timedelta(hours=16, minutes=30)

    assert ex._session_ended() is True, \
        "session did not end with a stale position still open"


def test_session_does_not_end_before_the_close(tmp_path):
    ex = make_executor(FakeBroker(), tmp_path)
    ex._now_override = pd.Timestamp.now("America/New_York").normalize() + \
        pd.Timedelta(hours=11, minutes=0)
    assert ex._session_ended() is False


# ── 7: intraday entries use MARKET orders (backtest parity) ─────────────────

def test_intraday_entry_uses_market_order(tmp_path):
    """
    A limit order only fills if the market retraces to the entry price.
    That silently drops every signal that ran away — a selection bias
    against the backtest's "enter at the signal bar's close" rule.
    """
    broker = _broker(50.60)
    ex = make_executor(broker, tmp_path)

    ex._open_position(_sig(), pd.Timestamp.now(), "BAC|2026-10-23|09:35")

    assert broker.submitted, "no order was submitted"
    symbol, qty, side, order_type = broker.submitted[0]
    assert order_type == "market", \
        f"intraday entry used a {order_type} order, expected market"


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
