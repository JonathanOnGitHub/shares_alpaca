"""Regression tests: backtest equity curves must reconcile with per-trade P&L.

The mean-reversion and swing backtests used to book short trades with long-style cash flow
in the equity curve (cash out at entry, cash back at the cover price), so a short that lost
money showed as a gain whenever the price rose. The per-trade ``pnl`` field was always right,
so the two disagreed. These tests check the invariant directly:

    on any date when no position is open (and with zero costs),
    equity - initial_capital == sum of realised trade pnl up to that date.
"""
import numpy as np
import pandas as pd
import pytest

from src.trading.mean_reversion import MeanReversion
from src.trading.swing import SwingTrading

INITIAL = 100_000.0
ZERO_COST = {"slippage_pct": 0.0, "commission_pct": 0.0, "initial_capital": INITIAL}

# RSI can never be >= 101 or <= -1, so these switch one side off.
NO_SHORTS = {"rsi_overbought": 101}
NO_LONGS = {"rsi_oversold": -1}


def make_series(seed: int, n: int = 1500, drift: float = 0.0006, vol: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift, vol, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.lognormal(mean=13.8, sigma=0.4, size=n)
    idx = pd.bdate_range("2018-01-02", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def make_universe(seed: int, drift: float, n_symbols: int = 10) -> dict[str, pd.DataFrame]:
    return {f"S{i}": make_series(seed * 100 + i, drift=drift) for i in range(n_symbols)}


def flat_date_residuals(result: dict) -> tuple[list[float], int]:
    """Return (equity - initial - cumulative realised pnl) on every date with no open position."""
    trades = result["trade_log"]
    equity = result["equity_curve"]
    by_date: dict = {}
    for t in trades:
        by_date.setdefault(t.date, []).append(t)

    buys = sells = 0
    cum_pnl = 0.0
    residuals = []
    for date in sorted(by_date):
        for t in by_date[date]:
            if t.side == "buy":
                buys += 1
            else:
                sells += 1
                cum_pnl += t.pnl
        if buys == sells and date in equity.index:
            residuals.append(float(equity.loc[date]) - INITIAL - cum_pnl)
    return residuals, sells


STRATEGIES = [
    pytest.param(MeanReversion, {}, id="mean_reversion"),
    # Looser thresholds than the defaults so the synthetic data produces plenty of trades.
    pytest.param(
        SwingTrading,
        {"volume_avg_multiplier": 0.5, "sma_trend": 20, "rsi_oversold": 40, "rsi_overbought": 60},
        id="swing",
    ),
]
MODES = [
    pytest.param({}, id="both-sides"),
    pytest.param(NO_SHORTS, id="long-only"),
    pytest.param(NO_LONGS, id="short-only"),
]


@pytest.mark.parametrize("drift", [0.0006, 0.0, -0.0006])
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("strategy_cls,extra", STRATEGIES)
def test_equity_reconciles_with_trade_pnl(strategy_cls, extra, mode, seed, drift):
    strategy = strategy_cls({**ZERO_COST, **extra, **mode})
    result = strategy.backtest(make_universe(seed, drift))

    residuals, n_sells = flat_date_residuals(result)
    assert n_sells >= 20, "synthetic data should produce enough closed trades to be meaningful"
    assert residuals, "expected at least one date with no open positions"
    assert max(abs(r) for r in residuals) < 1.0, (
        f"equity curve and trade pnl disagree by up to ${max(abs(r) for r in residuals):,.2f}"
    )


def test_position_value_matches_direction():
    from src.backtest.engine import position_value

    # Long: worth shares * price.
    assert position_value(shares=10, entry_price=100.0, price=90.0, direction=1) == pytest.approx(900.0)
    # Short: collateral (shares * entry) plus gain from the price falling.
    assert position_value(shares=10, entry_price=100.0, price=90.0, direction=-1) == pytest.approx(1100.0)
    assert position_value(shares=10, entry_price=100.0, price=110.0, direction=-1) == pytest.approx(900.0)
    # At the entry price both sides are worth the amount reserved.
    assert position_value(shares=10, entry_price=100.0, price=100.0, direction=-1) == pytest.approx(1000.0)
