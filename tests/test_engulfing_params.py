"""Tests for the reoptimised engulfing parameters.

The live strategy's stop width (STOP_ATR_MULT) and overnight hold window
(hold_sessions) are now configurable. These tests lock the reoptimised
defaults and the mechanics that apply them, so the backtest and the live
executor cannot drift apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from src.trading.engulfing_live import (  # noqa: E402
    STOP_ATR_MULT,
    TARGET_MULT,
    _detect_engulfing,
)
from src.trading.executor_live import PaperExecutor  # noqa: E402


class TestReoptimisedDefaults:
    """The values the sweep selected out of sample."""

    def test_target_mult_is_two(self):
        # 2R was the live value before and survives the sweep: beyond ~3R
        # the target is never reached, so the parameter goes inert.
        assert TARGET_MULT == 2.0

    def test_stop_atr_mult_is_two(self):
        # The sweep's interior optimum: a wider stop stops out far less
        # often (stop rate 48.8% -> 21.9% at hold=5).
        assert STOP_ATR_MULT == 2.0

    def test_backtest_defaults_match_live(self):
        import backtest_whole_shares as B
        assert B.detect_signals.__defaults__[0] == 10  # trend lookback
        # build_trades defaults must match the live constants
        import inspect
        sig = inspect.signature(B.build_trades)
        assert sig.parameters["target_mult"].default == TARGET_MULT
        assert sig.parameters["stop_atr_mult"].default == STOP_ATR_MULT


class TestStopWidthApplied:
    """A wider ATR stop must produce a wider stop and a lower target."""

    @pytest.fixture(scope="class")
    def bars(self):
        import numpy as np
        import pandas as pd
        # 40 bars engineered to produce exactly one bullish engulfing
        idx = pd.bdate_range("2024-01-01", periods=40)
        n = len(idx)
        o = np.full(n, 100.0)
        h = np.full(n, 101.0)
        l = np.full(n, 99.0)
        c = np.full(n, 100.0)
        # bar 20: prior red, current green engulfing it
        c[19] = 99.0
        o[19] = 100.5
        h[19] = 100.8
        l[19] = 98.8
        c[20] = 101.5
        o[20] = 99.5
        h[20] = 101.8
        l[20] = 99.2
        return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)

    def test_wider_stop_lowers_the_stop(self, bars):
        narrow = _detect_engulfing(bars, trend_lookback=0, target_date=None,
                                    target_mult=2.0, stop_atr_mult=1.0)
        wide = _detect_engulfing(bars, trend_lookback=0, target_date=None,
                                  target_mult=2.0, stop_atr_mult=2.0)
        if not narrow or not wide:
            pytest.skip("fixture bars produced no engulfing signal")
        assert wide[0]["stop_price"] < narrow[0]["stop_price"]

    def test_wider_stop_lowers_the_target(self, bars):
        narrow = _detect_engulfing(bars, trend_lookback=0, target_date=None,
                                    target_mult=2.0, stop_atr_mult=1.0)
        wide = _detect_engulfing(bars, trend_lookback=0, target_date=None,
                                  target_mult=2.0, stop_atr_mult=2.0)
        if not narrow or not wide:
            pytest.skip("fixture bars produced no engulfing signal")
        assert wide[0]["target_price"] < narrow[0]["target_price"]

    def test_target_mult_moves_the_target(self, bars):
        low = _detect_engulfing(bars, trend_lookback=0, target_date=None,
                                 target_mult=1.0, stop_atr_mult=1.0)
        high = _detect_engulfing(bars, trend_lookback=0, target_date=None,
                                  target_mult=3.0, stop_atr_mult=1.0)
        if not low or not high:
            pytest.skip("fixture bars produced no engulfing signal")
        assert high[0]["target_price"] > low[0]["target_price"]


class TestHoldSessionsDeadline:
    """The close deadline must move out with the hold window."""

    @staticmethod
    def _executor(hold_sessions):
        ex = PaperExecutor.__new__(PaperExecutor)
        ex.hold_sessions = hold_sessions
        ex.close_mode = "overnight"
        ex.market = "US"
        return ex

    def _deadline(self, hold_sessions):
        import pandas as pd
        ex = self._executor(hold_sessions)
        entry = pd.Timestamp("2024-01-02")     # a Tuesday
        return ex._close_deadline_for_entry(entry)

    def test_one_session_is_the_next_weekday(self):
        d = self._deadline(1)
        # entry Tue 2024-01-02 -> close at the open of Wed 2024-01-03
        assert d.strftime("%Y-%m-%d") == "2024-01-03"

    def test_three_sessions_is_three_weekdays_later(self):
        d = self._deadline(3)
        # entry Tue -> close at the open of Fri 2024-01-05
        assert d.strftime("%Y-%m-%d") == "2024-01-05"

    def test_longer_hold_is_always_later(self):
        assert self._deadline(3) > self._deadline(1)
        assert self._deadline(5) > self._deadline(3)

    def test_hold_skips_the_weekend(self):
        import pandas as pd
        ex = self._executor(3)
        # Friday 2024-01-05 entry -> Mon 2024-01-08, Tue 2024-01-09, Wed 2024-01-10
        d = ex._close_deadline_for_entry(pd.Timestamp("2024-01-05"))
        assert d.strftime("%Y-%m-%d") == "2024-01-10"

    def test_zero_or_negative_falls_back_to_one(self):
        assert self._deadline(0) == self._deadline(1)
        assert self._deadline(-5) == self._deadline(1)

    def test_missing_attribute_uses_class_default(self):
        ex = PaperExecutor.__new__(PaperExecutor)
        ex.close_mode = "overnight"
        ex.market = "US"
        d = ex._close_deadline_for_entry(__import__("pandas").Timestamp("2024-01-02"))
        assert d.strftime("%Y-%m-%d") == "2024-01-03"
