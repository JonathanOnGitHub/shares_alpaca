"""The ``weights`` / ``asset_returns`` each backtest returns must reproduce its own returns.

The selection and timing skill tests (src/validation/skill_tests.py) are only as good as the weights
they are fed, so these tests check the weights against each strategy's equity curve:

* Rebalancing strategies (momentum, low-vol, high-vol) and MA timing multiply weights by returns
  directly, so ``sum(weights * asset_returns)`` must match the equity curve exactly.
* The event-driven strategies (mean reversion, swing) fill at the open and are charged costs, while the
  weights are valued at the previous close. They reconcile exactly when the open equals the previous
  close and there are no costs (this includes shorts), and closely otherwise.
"""
import numpy as np
import pandas as pd
import pytest

from src.trading.high_vol import HighVolStrategy
from src.trading.low_vol import LowVol
from src.trading.ma_timing import MATiming
from src.trading.mean_reversion import MeanReversion
from src.trading.momentum import CrossSectionalMomentum
from src.trading.swing import SwingTrading


def gross_returns(result: dict) -> pd.Series:
    """Per-period return implied by the weights and the asset returns (NaN return = no return)."""
    return (result["weights"] * result["asset_returns"].fillna(0.0)).sum(axis=1)


def ohlcv(seed: int, n: int = 1500, drift: float = 0.0006, vol: float = 0.02, gaps: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift, vol, n))
    open_ = np.empty(n)
    open_[0] = close[0]
    # With gaps=False the open equals the previous close exactly.
    open_[1:] = close[:-1] * ((1 + rng.normal(0, 0.002, n - 1)) if gaps else 1.0)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.lognormal(13.8, 0.4, n)
    idx = pd.bdate_range("2018-01-02", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def universe(n_symbols: int, seed: int, **kw) -> dict[str, pd.DataFrame]:
    return {f"S{i}": ohlcv(seed * 100 + i, **kw) for i in range(n_symbols)}


EVENT_STRATEGIES = [
    pytest.param(MeanReversion, {}, id="mean_reversion"),
    # Looser thresholds than the defaults so the synthetic data produces plenty of trades.
    pytest.param(
        SwingTrading,
        {"volume_avg_multiplier": 0.5, "sma_trend": 20, "rsi_oversold": 40, "rsi_overbought": 60},
        id="swing",
    ),
]


# ------------------------------------------------------------------ event-driven strategies

@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("strategy_cls,extra", EVENT_STRATEGIES)
def test_event_weights_reconcile_exactly_without_gaps_or_costs(strategy_cls, extra, seed):
    cfg = {"initial_capital": 100_000.0, "slippage_pct": 0.0, "commission_pct": 0.0, **extra}
    result = strategy_cls(cfg).backtest(universe(10, seed, gaps=False))
    W = result["weights"]

    assert (W > 0).any().any() and (W < 0).any().any(), "expected both long and short positions"
    assert len([t for t in result["trade_log"] if t.side == "sell"]) >= 20

    equity_returns = result["equity_curve"].pct_change().loc[W.index]
    np.testing.assert_allclose(gross_returns(result).to_numpy(), equity_returns.to_numpy(), atol=1e-10)


@pytest.mark.parametrize("strategy_cls,extra", EVENT_STRATEGIES)
def test_event_weights_track_equity_closely_with_gaps_and_costs(strategy_cls, extra):
    result = strategy_cls({"initial_capital": 100_000.0, **extra}).backtest(universe(10, 1, gaps=True))
    W = result["weights"]
    equity_returns = result["equity_curve"].pct_change().loc[W.index]
    gross = gross_returns(result)

    assert np.corrcoef(gross, equity_returns)[0, 1] > 0.99
    assert (gross - equity_returns).abs().max() < 5e-3


@pytest.mark.parametrize("strategy_cls,extra", EVENT_STRATEGIES)
def test_event_weights_shape_and_flat_days(strategy_cls, extra):
    data = universe(6, 4)
    result = strategy_cls({"initial_capital": 100_000.0, **extra}).backtest(data)
    W, R = result["weights"], result["asset_returns"]

    assert list(W.columns) == list(data) and list(R.columns) == list(data)
    assert W.index.equals(R.index)
    assert W.index.is_monotonic_increasing and not W.index.has_duplicates
    assert W.index[0] > min(df.index[0] for df in data.values())  # no weights on the first day
    flat = W.abs().sum(axis=1) == 0
    assert flat.any() and (~flat).any(), "expected both flat and invested days"


# ------------------------------------------------------------------ rebalancing strategies

def _assert_matches_equity_curve(result: dict) -> None:
    W, equity = result["weights"], result["equity_curve"]
    assert W.index.equals(equity.index)
    rebuilt = (1 + gross_returns(result)).cumprod()
    np.testing.assert_allclose(rebuilt.to_numpy(), equity.to_numpy(), rtol=1e-12)


def test_momentum_weights_reproduce_the_equity_curve():
    cfg = {"lookback_months": 3, "skip_months": 1, "holding_months": 1, "top_n": 3, "bottom_n": 3}
    result = CrossSectionalMomentum(cfg).backtest(universe(12, 5))
    _assert_matches_equity_curve(result)
    W = result["weights"]
    assert (W > 0).any().any() and (W < 0).any().any()
    # Long and short legs are each one unit of exposure per holding period.
    held = W[W.abs().sum(axis=1) > 0]
    np.testing.assert_allclose(held.clip(lower=0).sum(axis=1), 1.0)
    np.testing.assert_allclose(held.clip(upper=0).sum(axis=1), -1.0)


def test_low_vol_weights_reproduce_the_equity_curve():
    cfg = {"vol_window": 63, "n_stocks": 5, "weighting": "inverse_vol"}
    result = LowVol(cfg).backtest(universe(12, 6))
    _assert_matches_equity_curve(result)
    held = result["weights"][result["weights"].abs().sum(axis=1) > 0]
    assert len(held) > 100
    np.testing.assert_allclose(held.sum(axis=1), 1.0)
    assert ((held > 0).sum(axis=1) == 5).all()


def test_high_vol_weights_reproduce_the_equity_curve_in_monthly_units():
    def monthly_universe():
        out = {}
        for i in range(12):
            rng = np.random.default_rng(i)
            idx = pd.bdate_range("2015-01-01", "2020-12-31")
            close = 50 * np.cumprod(1 + rng.normal(0.0004, 0.01 + 0.001 * i, len(idx)))
            out[f"S{i}"] = pd.DataFrame({"close": close}, index=idx)
        return out

    result = HighVolStrategy({"n_stocks": 10}).backtest(monthly_universe())
    _assert_matches_equity_curve(result)
    W, R = result["weights"], result["asset_returns"]
    # Monthly: consecutive rows are about a month apart, and each month holds 10 equal weights.
    assert 25 <= np.median(np.diff(W.index.to_numpy()).astype("timedelta64[D]").astype(int)) <= 31
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    assert ((W > 0).sum(axis=1) == 10).all()
    assert W.index.equals(R.index)


# ------------------------------------------------------------------ MA timing

def _ma_data(seed: int = 3, n: int = 1200) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    return {"S0": pd.DataFrame({"close": close}, index=pd.bdate_range("2018-01-02", periods=n))}


@pytest.mark.parametrize("use_benchmark", [False, True])
def test_ma_timing_weights_are_the_lagged_signal_and_reproduce_returns(use_benchmark):
    data = _ma_data()
    cfg = {"ma_window": 50, "slippage_pct": 0.0, "rf_rate": 0.0}
    result = MATiming(cfg).backtest(data, benchmark="S0" if use_benchmark else None)
    W = result["weights"]

    close = data["S0"]["close"]
    expected = (close > close.rolling(50).mean()).astype(float).shift(2).fillna(0.0)
    np.testing.assert_array_equal(W.iloc[:, 0].to_numpy(), expected.to_numpy())
    assert set(np.unique(W.to_numpy())) == {0.0, 1.0}

    # With no cash interest and no costs the equity curve is exactly the weights times the returns.
    equity_returns = result["equity_curve"].pct_change().iloc[1:]
    np.testing.assert_allclose(gross_returns(result).iloc[1:].to_numpy(), equity_returns.to_numpy(), atol=1e-12)
