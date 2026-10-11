"""Tests for HighVolStrategy: history handling and annualisation of monthly returns."""
import numpy as np
import pandas as pd
import pytest

from src.trading.high_vol import HighVolStrategy


def _prices(seed: int, start: str, end: str, daily_vol: float, drift: float = 0.0004) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    close = 50 * np.cumprod(1 + rng.normal(drift, daily_vol, len(idx)))
    return pd.DataFrame({"close": close}, index=idx)


def _universe(late: dict[str, tuple[str, float]] | None = None, n_stable: int = 12) -> dict[str, pd.DataFrame]:
    """n_stable symbols trading throughout 2015-2020, plus optional late listings {name: (start, daily_vol)}."""
    data = {f"S{i}": _prices(i, "2015-01-01", "2020-12-31", daily_vol=0.01 + 0.001 * i) for i in range(n_stable)}
    for k, (name, (start, vol)) in enumerate((late or {}).items()):
        data[name] = _prices(100 + k, start, "2020-12-31", daily_vol=vol)
    return data


def _rebalance_dates(signals: pd.DataFrame) -> list[tuple[int, int]]:
    return [(d.year, d.month) for d in signals.index]


def test_late_listing_does_not_truncate_the_history_of_everything_else():
    # One symbol lists in mid-2019. Rebalances before that must still happen for the symbols that
    # existed (the old dropna() discarded every month before the youngest symbol appeared).
    data = _universe(late={"LATE": ("2019-06-03", 0.02)})
    signals = HighVolStrategy({"n_stocks": 10}).compute_signals(data)

    years = {y for y, m in _rebalance_dates(signals)}
    # Data starts Jan 2015; a 10-month history is first available for the October 2016 rebalance.
    assert {2016, 2017, 2018, 2019, 2020} <= years


def test_symbol_without_enough_history_is_not_ranked():
    # IPO lists June 2018 with extreme volatility. At the October 2018 rebalance it has only ~4
    # monthly returns, so it must not be selected on a 4-month volatility estimate. A year later it has
    # enough history and, being the most volatile, must be selected.
    data = _universe(late={"IPO": ("2018-06-01", 0.08)})
    signals = HighVolStrategy({"n_stocks": 10, "min_vol_months": 10}).compute_signals(data)

    oct_2018 = signals[(signals.index.year == 2018) & (signals.index.month == 10)].iloc[0]
    oct_2019 = signals[(signals.index.year == 2019) & (signals.index.month == 10)].iloc[0]

    assert pd.isna(oct_2018["IPO"]) or oct_2018["IPO"] == 0
    assert oct_2019["IPO"] == pytest.approx(0.1)
    assert oct_2018.sum() == pytest.approx(1.0)  # still fully invested, in names with enough history


def test_annual_return_and_sharpe_use_monthly_units():
    data = _universe()
    result = HighVolStrategy({"n_stocks": 10}).backtest(data)
    cum = result["equity_curve"]
    assert len(cum) > 12

    monthly = cum.pct_change()
    monthly.iloc[0] = cum.iloc[0] - 1
    n_months = len(monthly)

    expected_sharpe = monthly.mean() / monthly.std() * np.sqrt(12)
    expected_ann = cum.iloc[-1] ** (12 / n_months) - 1

    assert result["sharpe"] == pytest.approx(expected_sharpe, rel=1e-6)
    assert result["annual_return"] == pytest.approx(expected_ann, rel=1e-6)
    # Sanity: a monthly series annualised with the right factor gives a plausible CAGR, not thousands of percent.
    assert abs(result["annual_return"]) < 5
