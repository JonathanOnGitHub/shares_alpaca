"""DiligenceSuite's permutation check when per-date weights are supplied.

With ``weights`` and ``asset_returns`` the check must test selection or timing skill. The key property is that a
strategy which is merely long a rising market passes the old test against zero edge but fails the skill test.
"""
import numpy as np
import pandas as pd
import pytest

from src.validation.diligence import DiligenceSuite

N_DAYS = 750
DAYS = pd.bdate_range("2019-01-01", periods=N_DAYS)


def _returns(seed: int, n_symbols: int = 20, common_drift: float = 0.001,
             good: int = 0, good_extra: float = 0.0, vol: float = 0.01) -> pd.DataFrame:
    """Daily returns: a shared drift, plus extra drift for the first ``good`` symbols."""
    rng = np.random.default_rng(seed)
    drift = np.full(n_symbols, common_drift)
    drift[:good] += good_extra
    data = drift + vol * rng.normal(size=(N_DAYS, n_symbols))
    return pd.DataFrame(data, index=DAYS, columns=[f"S{i}" for i in range(n_symbols)])


def _hold(returns: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    w[symbols] = 1.0 / len(symbols)
    return w


def _suite(weights: pd.DataFrame | None, returns: pd.DataFrame) -> DiligenceSuite:
    gross = ((weights if weights is not None else _hold(returns, list(returns.columns)[:5])) * returns).sum(axis=1)
    equity = 100_000 * (1 + gross).cumprod()
    kwargs = {} if weights is None else {"weights": weights, "asset_returns": returns}
    return DiligenceSuite(equity_curve=equity, trades=[], prices={}, strategy_name="t", **kwargs)


def _perm(suite: DiligenceSuite):
    return suite._check_permutation()


def test_planted_selection_skill_passes_and_says_so():
    returns = _returns(seed=1, good=5, good_extra=0.003)
    check = _perm(_suite(_hold(returns, ["S0", "S1", "S2", "S3", "S4"]), returns))

    assert check.passed, check.detail
    assert check.name == "Permutation Test"
    assert check.detail.startswith("Selection test")
    assert "selection" in check.metrics["Test"]
    assert float(check.metrics["p-value"]) < 0.05


def test_being_long_a_rising_market_passes_the_zero_edge_test_but_not_the_selection_test():
    # Every symbol has the same drift, so holding any five of them earns the market return and
    # nothing for choosing well. The zero-edge test (no weights) cannot tell the difference.
    passes_edge_only = 0
    passes_skill = 0
    seeds = range(8)
    for seed in seeds:
        returns = _returns(seed=seed, good=0)
        held = ["S0", "S1", "S2", "S3", "S4"]
        w = _hold(returns, held)

        edge_only = _perm(_suite(None, returns))  # built without weights
        skill = _perm(_suite(w, returns))
        passes_edge_only += bool(edge_only.passed)
        passes_skill += bool(skill.passed)

    assert passes_edge_only >= 6, "market drift should look like an edge when nothing asks for skill"
    # A calibrated test passes ~5% of the time by chance; allow some slack on 8 draws.
    assert passes_skill <= 2


def test_few_symbols_use_the_timing_test():
    returns = _returns(seed=3, n_symbols=2)
    w = _hold(returns, ["S0"])
    check = _perm(_suite(w, returns))

    assert check.detail.startswith("Timing test")
    assert "timing" in check.metrics["Test"]


def test_timing_skill_passes_and_constant_exposure_does_not():
    rng = np.random.default_rng(5)
    regime = np.repeat(rng.choice([0, 1], size=N_DAYS // 50 + 1), 50)[:N_DAYS]
    r = pd.DataFrame({"X": np.where(regime == 1, 0.003, -0.003) + 0.01 * rng.normal(size=N_DAYS)}, index=DAYS)

    in_when_up = pd.DataFrame({"X": regime.astype(float)}, index=DAYS)
    always_in = pd.DataFrame({"X": np.ones(N_DAYS)}, index=DAYS)

    skilled = _perm(_suite(in_when_up, r))
    constant = _perm(_suite(always_in, r))
    assert skilled.detail.startswith("Timing test")  # the weights were actually used
    assert skilled.passed
    assert constant.detail.startswith("Timing test")
    assert not constant.passed


def test_weights_and_asset_returns_must_be_given_together():
    returns = _returns(seed=2)
    equity = 100_000 * (1 + (_hold(returns, ["S0"]) * returns).sum(axis=1)).cumprod()
    with pytest.raises(ValueError, match="together"):
        DiligenceSuite(equity_curve=equity, trades=[], prices={}, weights=_hold(returns, ["S0"]))
    with pytest.raises(ValueError, match="together"):
        DiligenceSuite(equity_curve=equity, trades=[], prices={}, asset_returns=returns)


def test_a_strategy_with_no_positions_fails_cleanly_instead_of_raising():
    returns = _returns(seed=4)
    flat = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    equity = pd.Series(100_000.0 + np.arange(N_DAYS), index=returns.index)
    suite = DiligenceSuite(equity_curve=equity, trades=[], prices={}, weights=flat, asset_returns=returns)

    check = _perm(suite)
    assert not check.passed
    assert "not possible" in check.detail


def test_timezone_aware_indexes_are_aligned():
    returns = _returns(seed=6, good=5, good_extra=0.003)
    w = _hold(returns, ["S0", "S1", "S2", "S3", "S4"])
    equity = 100_000 * (1 + (w * returns).sum(axis=1)).cumprod()

    aware = lambda df: df.set_axis(df.index.tz_localize("America/New_York"), axis=0)
    suite = DiligenceSuite(
        equity_curve=equity.set_axis(equity.index.tz_localize("America/New_York")),
        trades=[], prices={}, weights=aware(w), asset_returns=aware(returns),
    )
    check = _perm(suite)
    assert check.detail.startswith("Selection test")  # weights and returns aligned, not silently ignored
    assert check.passed


def test_run_all_still_produces_nine_checks_with_weights():
    returns = _returns(seed=7, good=5, good_extra=0.003)
    report = _suite(_hold(returns, ["S0", "S1", "S2", "S3", "S4"]), returns).run_all()
    assert len(report.checks) == 9
    assert [c.name for c in report.checks].count("Permutation Test") == 1
