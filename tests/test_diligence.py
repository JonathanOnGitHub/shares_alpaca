"""Tests for the diligence suite checks that were mis-specified.

- B&H benchmark: must be built from returns (equal weight), not from averaged price levels.
- Sharpe significance: an *annualised* Sharpe must be compared with 2/sqrt(years), not 2/sqrt(months).
- Permutation test: must test the strategy's own edge, not rank it against pooled single-stock returns.
"""
import numpy as np
import pandas as pd
import pytest

from src.validation.diligence import DiligenceSuite


def _suite(equity: pd.Series, prices: dict | None = None) -> DiligenceSuite:
    return DiligenceSuite(equity_curve=equity, trades=[], prices=prices or {}, strategy_name="test")


def _check(report, name_prefix: str):
    matches = [c for c in report.checks if c.name.startswith(name_prefix)]
    assert len(matches) == 1, f"expected one check starting with {name_prefix!r}, got {[c.name for c in report.checks]}"
    return matches[0]


def _equity_from_monthly(monthly: np.ndarray, start="2020-01-01") -> pd.Series:
    """Daily equity curve whose calendar month-end values compound exactly to the given monthly returns."""
    months = pd.period_range(start, periods=len(monthly), freq="M")
    days = pd.bdate_range(months[0].start_time, months[-1].end_time)
    month_of_day = days.to_period("M")
    values, level = [], 100.0
    for month, r in zip(months, monthly):
        n_days = int((month_of_day == month).sum())
        daily = (1 + r) ** (1 / n_days)
        for _ in range(n_days):
            level *= daily
            values.append(level)
    return pd.Series(values, index=days)


def _monthly_with_sharpe(annual_sharpe: float, n_months: int, seed: int) -> np.ndarray:
    z = np.random.default_rng(seed).normal(size=n_months)
    z = (z - z.mean()) / z.std(ddof=1)
    sigma = 0.04
    mu = annual_sharpe / np.sqrt(12) * sigma
    return mu + sigma * z


# ---------------------------------------------------------------- Sharpe significance

def test_sharpe_threshold_uses_years_not_months():
    # ~4 years of monthly data. The correct 5% threshold on an annualised Sharpe is about 2/sqrt(4) = 1.0.
    # The old code used 2/sqrt(48) = 0.29, which let a Sharpe of 0.5 through.
    weak = _suite(_equity_from_monthly(_monthly_with_sharpe(0.5, 48, seed=1))).run_all()
    strong = _suite(_equity_from_monthly(_monthly_with_sharpe(2.0, 48, seed=2))).run_all()

    assert not _check(weak, "Sharpe Significance").passed
    assert _check(strong, "Sharpe Significance").passed


def test_sharpe_threshold_value_matches_formula():
    report = _suite(_equity_from_monthly(_monthly_with_sharpe(1.5, 60, seed=3))).run_all()
    check = _check(report, "Sharpe Significance")
    n_months = int(check.detail.split("N=")[1].split(" ")[0])
    expected = 2 / np.sqrt(n_months / 12)
    assert float(check.metrics["Significance Threshold"]) == pytest.approx(expected, abs=0.005)


# ---------------------------------------------------------------- Buy & hold benchmark

def test_buy_hold_benchmark_is_equal_weight_not_price_weighted():
    days = pd.bdate_range("2021-01-04", periods=251)
    n = len(days) - 1
    # A: $1000 stock that goes nowhere. B: $10 stock that doubles smoothly.
    a = pd.Series(1000.0, index=days)
    b = pd.Series(10.0 * 2 ** (np.arange(len(days)) / n), index=days)
    prices = {"A": pd.DataFrame({"close": a}), "B": pd.DataFrame({"close": b})}

    # Strategy: smooth +30% over the same period.
    strat = pd.Series(100.0 * 1.30 ** (np.arange(len(days)) / n), index=days)

    check = _check(_suite(strat, prices).run_all(), "vs Equal-Weight")
    bh_return = float(check.metrics["B&H Return"].rstrip("%"))

    # Equal weight of a flat stock and a doubling stock returns roughly +41%, so +30% does not beat it.
    # (Averaging price levels would have given the benchmark only about +1%, and a false pass.)
    assert 35 < bh_return < 45
    assert not check.passed


def test_buy_hold_benchmark_handles_late_listing():
    days = pd.bdate_range("2021-01-04", periods=251)
    a = pd.Series(50.0 * 1.0004 ** np.arange(len(days)), index=days)
    b = pd.Series(np.nan, index=days)
    b.iloc[125:] = 5.0 * 1.0004 ** np.arange(len(days) - 125)  # lists half way through
    prices = {"A": pd.DataFrame({"close": a}), "B": pd.DataFrame({"close": b})}
    strat = pd.Series(100.0 * 1.0004 ** np.arange(len(days)), index=days)

    check = _check(_suite(strat, prices).run_all(), "vs Equal-Weight")
    # Both series grow at the same daily rate, so the benchmark must match the strategy and must not
    # show a spurious jump when B first appears.
    bh_return = float(check.metrics["B&H Return"].rstrip("%"))
    strat_return = float(check.metrics["Strategy Return"].rstrip("%"))
    assert bh_return == pytest.approx(strat_return, abs=0.5)


# ---------------------------------------------------------------- Permutation test

def _daily_equity(daily_returns: np.ndarray, start="2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(daily_returns))
    return pd.Series(100.0 * np.cumprod(1 + daily_returns), index=idx)


def test_permutation_passes_for_a_real_edge():
    rets = np.random.default_rng(10).normal(0.0015, 0.008, 1000)  # Sharpe around 3
    check = _check(_suite(_daily_equity(rets)).run_all(), "Permutation")
    assert check.passed


def test_permutation_fails_for_noise():
    for seed in range(5):
        rets = np.random.default_rng(seed).normal(0.0, 0.01, 1000)
        check = _check(_suite(_daily_equity(rets)).run_all(), "Permutation")
        assert not check.passed, f"zero-mean noise should not pass (seed {seed})"


def test_permutation_does_not_depend_on_unrelated_prices():
    # The old test ranked the strategy against pooled daily returns of whatever universe was passed in,
    # so the verdict changed with the universe. The strategy's own edge should not.
    rets = np.random.default_rng(11).normal(0.0006, 0.01, 1000)
    eq = _daily_equity(rets)
    idx = eq.index
    rng = np.random.default_rng(12)
    calm = {f"C{i}": pd.DataFrame({"close": 100 * np.cumprod(1 + rng.normal(0, 0.002, len(idx)))}, index=idx) for i in range(3)}
    wild = {f"W{i}": pd.DataFrame({"close": 100 * np.cumprod(1 + rng.normal(0, 0.05, len(idx)))}, index=idx) for i in range(3)}

    p_calm = _check(_suite(eq, calm).run_all(), "Permutation").metrics["p-value"]
    p_wild = _check(_suite(eq, wild).run_all(), "Permutation").metrics["p-value"]
    assert p_calm == p_wild


def test_permutation_is_deterministic():
    rets = np.random.default_rng(13).normal(0.0004, 0.01, 800)
    eq = _daily_equity(rets)
    a = _check(_suite(eq).run_all(), "Permutation").metrics["p-value"]
    b = _check(_suite(eq).run_all(), "Permutation").metrics["p-value"]
    assert a == b
