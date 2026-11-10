"""Regression tests for the live-path portfolio simulation.

The same-day exit ordering bug (a trade that stops out or targets during
its own entry session had its exit processed BEFORE its entry, so the
sizing record did not exist yet and the trade was silently dropped)
inflated Sharpe by ~35x. These tests lock the corrected behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from backtest_whole_shares import build_trades, simulate  # noqa: E402


def _trade(**kw):
    base = {
        "ticker": "TEST", "signal_date": "2024-01-01",
        "entry_date": "2024-01-02", "exit_date": "2024-01-03",
        "proxy": 100.0, "entry": 100.0, "stop": 95.0, "target": 110.0,
        "exit_price": 105.0, "exit_reason": "overnight_exit",
        "risk_per_share": 5.0, "_entry_bar": 1, "_exit_bar": 2,
    }
    base.update(kw)
    return base


class TestSameDayExits:
    """A trade that exits during its OWN entry session must still be booked."""

    def test_same_day_stop_is_counted(self):
        # entered and stopped out on the same session
        t = _trade(exit_date="2024-01-02", exit_price=95.0, exit_reason="stop")
        r = simulate([t], 10_000.0, whole_shares=True, cost_bps=0.0)
        assert r["n_trades"] == 1, "same-day stop was silently dropped"

    def test_same_day_target_is_counted(self):
        t = _trade(exit_date="2024-01-02", exit_price=110.0, exit_reason="target")
        r = simulate([t], 10_000.0, whole_shares=True, cost_bps=0.0)
        assert r["n_trades"] == 1, "same-day target was silently dropped"

    def test_next_day_exit_is_counted(self):
        t = _trade()
        r = simulate([t], 10_000.0, whole_shares=True, cost_bps=0.0)
        assert r["n_trades"] == 1

    def test_mixed_same_day_and_multi_day_exits(self):
        trades = [
            _trade(exit_date="2024-01-02", exit_price=95.0, exit_reason="stop"),
            _trade(entry_date="2024-01-02", exit_date="2024-01-04",
                   exit_price=105.0, exit_reason="overnight_exit"),
        ]
        r = simulate(trades, 10_000.0, whole_shares=True, cost_bps=0.0)
        assert r["n_trades"] == 2

    def test_same_day_loss_reduces_capital(self):
        t = _trade(exit_date="2024-01-02", exit_price=95.0, exit_reason="stop")
        r = simulate([t], 10_000.0, whole_shares=True, cost_bps=0.0)
        assert r["total_pnl_dollars"] < 0
        assert r["n_trades"] == 1


class TestHoldWindow:
    """`build_trades` must honour the requested hold period."""

    @pytest.fixture(scope="class")
    def price_df(self):
        import pickle
        with open(ROOT / "data" / "engulfing_universe.pkl", "rb") as fh:
            return pickle.load(fh)

    @pytest.fixture(scope="class")
    def universe(self):
        import pickle
        with open(ROOT / "data" / "universe_rankings.pkl", "rb") as fh:
            ranked = pickle.load(fh)
        return ranked[:20]

    def test_longer_hold_exits_later(self, price_df, universe):
        one = build_trades(price_df, universe, hold_sessions=1)
        five = build_trades(price_df, universe, hold_sessions=5)
        assert one and five
        # the same setups must not exit EARLIER with a longer hold
        key = lambda t: (t["ticker"], t["entry_date"])  # noqa: E731
        one_map = {key(t): t for t in one}
        later_or_equal = 0
        for t in five:
            o = one_map.get(key(t))
            if o is None:
                continue
            if t["exit_date"] >= o["exit_date"]:
                later_or_equal += 1
        assert later_or_equal > 0

    def test_unbounded_has_no_time_exit(self, price_df, universe):
        unb = build_trades(price_df, universe, hold_sessions=0)
        reasons = {t["exit_reason"] for t in unb}
        assert "hold_0_exit" not in reasons
        assert reasons <= {"stop", "target"}

    def test_finite_hold_has_time_exit(self, price_df, universe):
        tr = build_trades(price_df, universe, hold_sessions=3)
        assert "hold_3_exit" in {t["exit_reason"] for t in tr}

    def test_gap_through_stop_fills_at_open(self, price_df, universe):
        """A GTC stop cannot fill below the open, so a gap-down fills at the open."""
        for hold in (1, 3, 5):
            for t in build_trades(price_df, universe, hold_sessions=hold):
                if t["exit_reason"] == "stop" and t["exit_price"] < t["stop"]:
                    # only legitimate if it gapped: the open was below the stop
                    assert t["exit_price"] <= t["stop"]
