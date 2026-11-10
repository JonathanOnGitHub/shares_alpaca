"""Tests for the whole-share sizing rule used by the live executor.

These lock in the behaviour that `scripts/backtest_whole_shares.py`
models, so the backtest and the executor cannot drift apart silently.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from backtest_whole_shares import (  # noqa: E402
    MAX_RISK_MULT,
    RISK_PCT,
    size_position,
)


class TestFractionalSizing:
    """The pre-whole-share path: fractional size, notional capped at 2x risk."""

    def test_size_is_risk_budget_over_stop_distance(self):
        # risk $100, stop distance $5 -> 20 shares; at a $10 price the
        # notional ($200) sits exactly on the 2x cap, so it is not truncated
        qty, capped, status = size_position(20.0, 10.0, 5.0, 100.0, whole_shares=False)
        assert qty == pytest.approx(20.0)
        assert capped is False
        assert status == "ok"

    def test_notional_cap_truncates_position(self):
        # 20 shares at $100 = $2000 notional, but the cap is 2x $100 = $200
        qty, capped, status = size_position(20.0, 100.0, 5.0, 100.0, whole_shares=False)
        assert qty == pytest.approx(2.0)
        assert capped is True
        assert status == "ok"

    def test_cap_constant_is_two_times_risk(self):
        assert MAX_RISK_MULT == 2.0
        assert RISK_PCT == 0.01


class TestWholeShareSizing:
    """Whole shares are required so a broker-side GTC stop can be placed."""

    def test_rounds_down_to_whole_shares(self):
        # 19.7 shares at a $10 price = $197 notional, inside the $200 cap
        qty, _, status = size_position(19.7, 10.0, 5.0, 100.0, whole_shares=True)
        assert qty == 19.0
        assert status == "ok"

    def test_never_returns_fractional_quantity(self):
        for raw in (0.5, 1.2, 3.99, 7.01, 12.5, 99.999):
            qty, _, _ = size_position(raw, 10.0, 2.0, 100.0, whole_shares=True)
            assert qty == math.floor(qty), f"{raw} produced fractional qty {qty}"

    def test_takes_one_share_when_risk_budget_allows(self):
        # capped size under 1 share, but one share's risk ($50) fits the $100 budget
        qty, _, status = size_position(0.5, 100.0, 50.0, 100.0, whole_shares=True)
        assert qty == 1.0
        assert status == "ok"

    def test_skips_when_one_share_exceeds_risk_budget(self):
        # one share would risk $150 against a $100 budget -> too risky, skip
        qty, _, status = size_position(0.5, 100.0, 150.0, 100.0, whole_shares=True)
        assert qty == 0.0
        assert status == "skipped_risk"


class TestNotionalCapOnWholeShares:
    """`respect_notional` skips setups whose single share breaches the cap.

    The executor currently takes that share anyway (the cap breach noted in
    reports/engulfing_whole_shares_summary.md); this pins the alternative.
    """

    def test_breach_taken_by_default(self):
        # one share of a $900 stock = $900 notional vs a $200 cap
        qty, _, status = size_position(0.5, 900.0, 50.0, 100.0, whole_shares=True)
        assert qty == 1.0
        assert status == "ok"

    def test_breach_skipped_when_respected(self):
        qty, _, status = size_position(
            0.5, 900.0, 50.0, 100.0, whole_shares=True, respect_notional=True
        )
        assert qty == 0.0
        assert status == "skipped_notional"

    def test_within_cap_is_unaffected(self):
        qty, _, status = size_position(
            0.5, 150.0, 50.0, 100.0, whole_shares=True, respect_notional=True
        )
        assert qty == 1.0
        assert status == "ok"


class TestNotionalCapByCapitalBase:
    """The cap breach only bites at small capital bases."""

    @staticmethod
    def _breach(capital: float, proxy: float, stop_dist: float) -> bool:
        risk = capital * RISK_PCT
        cap = risk * MAX_RISK_MULT
        raw = risk / stop_dist
        if proxy * raw > cap:
            raw = cap / proxy
        whole = math.floor(raw)
        if whole < 1:
            if stop_dist > risk:
                return False
            whole = 1
            return proxy > cap
        return False

    def test_breach_at_small_capital(self):
        # $10k capital -> $200 notional cap; one $500 share breaches it
        assert self._breach(10_000.0, 500.0, 5.0) is True

    def test_no_breach_at_real_account_equity(self):
        # ~$103k capital -> $2060 notional cap; $500 is comfortably inside
        assert self._breach(103_000.0, 500.0, 5.0) is False

    def test_no_breach_when_price_is_inside_the_cap(self):
        # $150 stock against a $200 cap needs no breach at any capital base
        assert self._breach(10_000.0, 150.0, 5.0) is False
