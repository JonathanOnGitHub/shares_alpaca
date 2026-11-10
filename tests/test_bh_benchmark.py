"""Regression tests for the Buy & Hold benchmark comparison.

Two bugs are locked here:

1. The composite added the B&H daily returns instead of compounding them,
   so a 30x move collapsed to a fraction of it and the composite showed
   ~15% annual return against a 37% Buy & Hold.
2. The B&H daily-return array was built with a mismatched boolean mask
   (size 2764 against a 2765 array), which raised IndexError.

The benchmark is also checked for the property that actually matters:
a strategy that trades a tiny fraction of the capital should NOT be
able to beat a fully-invested Buy & Hold on a fair composite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from engulfing_vs_bh_portfolio import (  # noqa: E402
    _date_range, buy_and_hold, composite, curve_metrics,
)


def _price_df(n_days: int = 400, n_tickers: int = 3, growth: float = 1.0005):
    import pandas as pd
    import numpy as np
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    out = {}
    for t in range(n_tickers):
        close = 100.0 * (growth ** np.arange(n_days))
        out[f"T{t}"] = pd.DataFrame({
            "Open": close, "High": close * 1.01,
            "Low": close * 0.99, "Close": close, "Volume": 1_000_000,
        }, index=idx)
    return out


def test_date_range_is_inclusive():
    assert _date_range("2024-01-01", "2024-01-03") == [
        "2024-01-01", "2024-01-02", "2024-01-03"]
    assert _date_range("2024-01-03", "2024-01-01") == ["2024-01-03"]


def test_curve_metrics_handles_single_point():
    m = curve_metrics([("2024-01-01", 100.0)], 100.0)
    assert m["ann_ret_pct"] == 0.0
    assert m["sharpe"] == 0.0


def test_bh_compounds_rather_than_adds():
    """B&H over a growing market must compound to the full move."""
    price_df = _price_df(growth=1.001)
    bh = buy_and_hold(price_df, ["T0", "T1", "T2"], 10_000.0, 0.0)
    curve = bh["curve"]
    expected_mult = float(price_df["T0"]["Close"].iloc[-1]
                          / price_df["T0"]["Open"].iloc[0])
    got_mult = curve[-1][1] / curve[0][1]
    # within rounding of whole-share allocation
    assert got_mult == pytest.approx(expected_mult, rel=0.02)


def test_bh_charges_costs():
    """Round-trip cost must reduce the B&H result."""
    price_df = _price_df(growth=1.001)
    free = buy_and_hold(price_df, ["T0"], 10_000.0, 0.0)["metrics"]
    costed = buy_and_hold(price_df, ["T0"], 10_000.0, 10.0)["metrics"]
    assert costed["total_ret_pct"] < free["total_ret_pct"]


def test_composite_does_not_add_returns():
    """The idle cash must COMPOUND the B&H return, not add it.

    Regression for the bug that reported ~15% annual return against a
    37% Buy & Hold.
    """
    price_df = _price_df(n_days=400, growth=1.0015)
    bh = buy_and_hold(price_df, ["T0", "T1", "T2"], 10_000.0, 0.0)
    # a strategy that risks nothing and always holds one position
    trades = [{
        "ticker": "T0", "entry_date": "2020-01-02", "exit_date": "2021-06-30",
        "notional": 200.0, "pnl_dollars": 0.0,
    }]
    m = composite(bh, trades, 10_000.0)
    bh_mult = bh["curve"][-1][1] / bh["curve"][0][1]
    comp_mult = 1 + m["total_ret_pct"] / 100
    # the composite must be within 10% of the pure B&H multiple — an
    # ADDITIVE bug would put it far below (that was ~1.2x vs 31x)
    assert comp_mult == pytest.approx(bh_mult, rel=0.10)


def test_composite_with_idle_cash_keeps_bh_scale():
    """A near-fully-idle strategy should roughly track Buy & Hold."""
    price_df = _price_df(n_days=400, growth=1.0015)
    bh = buy_and_hold(price_df, ["T0", "T1", "T2"], 10_000.0, 0.0)
    trades = [{
        "ticker": "T0", "entry_date": "2020-01-02", "exit_date": "2020-03-02",
        "notional": 200.0, "pnl_dollars": 0.0,
    }]
    m = composite(bh, trades, 10_000.0)
    bh_mult = bh["curve"][-1][1] / bh["curve"][0][1]
    comp_mult = 1 + m["total_ret_pct"] / 100
    assert comp_mult == pytest.approx(bh_mult, rel=0.10)
    assert m["avg_deployed_pct"] < 5.0


def test_composite_with_strategic_pnl_adds_to_bh():
    """The strategy's own P&L must ADD to the composite, not be lost."""
    price_df = _price_df(n_days=400, growth=1.0015)
    bh = buy_and_hold(price_df, ["T0", "T1", "T2"], 10_000.0, 0.0)
    trades = [{
        "ticker": "T0", "entry_date": "2020-01-02", "exit_date": "2020-03-02",
        "notional": 200.0, "pnl_dollars": 0.0,
    }]
    base = composite(bh, trades, 10_000.0)
    trades2 = [dict(trades[0], pnl_dollars=1000.0)]
    with_pnl = composite(bh, trades2, 10_000.0)
    assert with_pnl["total_ret_pct"] > base["total_ret_pct"]
