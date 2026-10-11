"""Tests for the selection and timing skill tests (src/validation/skill_tests.py)."""
import numpy as np
import pandas as pd
import pytest

from src.validation.skill_tests import (
    SkillTestResult,
    _episodes,
    selection_test,
    timing_test,
)


def _universe(n_days: int = 600, n_syms: int = 30, vol: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"S{i}" for i in range(n_syms)]
    return pd.DataFrame(rng.normal(0.0003, vol, (n_days, n_syms)), index=idx, columns=cols)


def _monthly_picks(returns: pd.DataFrame, picker, k: int = 3, block: int = 21) -> pd.DataFrame:
    """Equal-weight long k symbols per 21-day block, chosen by ``picker(block_returns, k)``."""
    W = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for start in range(0, len(returns), block):
        blk = returns.iloc[start:start + block]
        W.loc[blk.index, list(picker(blk, k))] = 1.0 / k
    return W


def _oracle(blk: pd.DataFrame, k: int):
    """Look-ahead picker: the k symbols that did best over the block. Planted skill."""
    return blk.sum().nlargest(k).index


def _random_picker(seed: int):
    rng = np.random.default_rng(seed)
    return lambda blk, k: rng.choice(blk.columns, size=k, replace=False)


# ------------------------------------------------------------------ plumbing

def test_episodes_split_on_gaps_and_sign_changes():
    W = np.zeros((8, 2))
    W[:, 0] = [0, 0.1, 0.1, 0, 0.1, -0.1, -0.1, 0]
    W[:, 1] = [0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0]
    eps = sorted(_episodes(W))
    # (column, start, end_exclusive)
    assert eps == [(0, 1, 3), (0, 4, 5), (0, 5, 7), (1, 0, 4)]


def test_observed_statistic_is_mean_gross_return_of_the_weights():
    rets = _universe(n_days=300, n_syms=10, seed=3)
    W = _monthly_picks(rets, _random_picker(1))
    rets_with_gaps = rets.copy()
    rets_with_gaps.iloc[:40, 9] = np.nan  # NaN returns are treated as zero by the real portfolio
    W.iloc[:40, 9] = 0.0  # ...and the strategy holds nothing there

    result = selection_test(W, rets_with_gaps, n_permutations=50)
    expected = float((W * rets_with_gaps.fillna(0.0)).sum(axis=1).mean())
    assert result.observed == pytest.approx(expected, rel=1e-12, abs=1e-15)

    timing = timing_test(W, rets_with_gaps, n_permutations=50)
    assert timing.observed == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_results_are_deterministic_for_a_seed():
    rets = _universe(n_days=300, n_syms=12, seed=4)
    W = _monthly_picks(rets, _random_picker(2))
    a = selection_test(W, rets, n_permutations=200, seed=7)
    b = selection_test(W, rets, n_permutations=200, seed=7)
    assert a == b
    c = selection_test(W, rets, n_permutations=200, seed=8)
    assert c.null_mean != a.null_mean


def test_input_validation():
    rets = _universe(n_days=100, n_syms=6)
    W = _monthly_picks(rets, _random_picker(0))
    with pytest.raises(ValueError, match="missing weights columns"):
        selection_test(W.assign(EXTRA=0.1), rets)
    with pytest.raises(ValueError, match="no positions"):
        selection_test(W * 0.0, rets)
    with pytest.raises(ValueError, match="no positions"):
        timing_test(W * 0.0, rets)
    with pytest.raises(ValueError, match="no rows"):
        selection_test(W, rets.iloc[:50])


# ------------------------------------------------------------------ selection: skill and calibration

def test_selection_test_detects_planted_skill():
    rets = _universe(seed=11)
    W = _monthly_picks(rets, _oracle)
    result = selection_test(W, rets, n_permutations=1000)
    assert isinstance(result, SkillTestResult)
    assert result.p_value < 0.01
    assert result.observed > result.null_mean + 3 * result.null_std


def test_selection_test_is_roughly_calibrated_for_random_pickers():
    # With no skill the p-value should be roughly uniform: about 5% of 30 strategies below 0.05.
    # Allow up to 6 (chance of 7+ under a true 5% rate is under 0.1%).
    pvals = []
    for seed in range(30):
        rets = _universe(n_days=420, n_syms=25, seed=100 + seed)
        W = _monthly_picks(rets, _random_picker(500 + seed))
        pvals.append(selection_test(W, rets, n_permutations=400, seed=1).p_value)
    assert sum(p < 0.05 for p in pvals) <= 6
    assert np.mean(pvals) > 0.3


def test_buy_and_hold_the_whole_universe_has_no_selection_or_timing_skill():
    rets = _universe(n_days=400, n_syms=20, seed=21)
    W = pd.DataFrame(1.0 / rets.shape[1], index=rets.index, columns=rets.columns)
    sel = selection_test(W, rets, n_permutations=500)
    tim = timing_test(W, rets, n_permutations=500)
    assert sel.p_value > 0.1
    # Shifting constant weights changes nothing, so no shift can be beaten: p is exactly 1.
    assert tim.p_value == pytest.approx(1.0)


def test_no_dispersion_means_no_selection_skill():
    # Every symbol has the same returns, so no choice of symbol can matter.
    base = np.random.default_rng(5).normal(0.0005, 0.01, 300)
    idx = pd.bdate_range("2020-01-01", periods=300)
    rets = pd.DataFrame({f"S{i}": base for i in range(8)}, index=idx)
    W = _monthly_picks(rets, _random_picker(9))
    assert selection_test(W, rets, n_permutations=300).p_value == pytest.approx(1.0)


def test_shorts_are_replaced_by_shorts():
    # One symbol rips upward; the others are flat. Shorting the one that rips is the worst possible
    # short. If the null flipped the side the strategy would look fine; it must not.
    idx = pd.bdate_range("2020-01-01", periods=300)
    cols = [f"S{i}" for i in range(10)]
    rets = pd.DataFrame(0.0, index=idx, columns=cols)
    rets["S0"] = 0.004
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    for start in range(0, 300, 30):
        W.iloc[start:start + 30, 0] = -0.5  # short S0 every block
    result = selection_test(W, rets, n_permutations=500)
    assert result.observed < result.null_mean
    assert result.p_value > 0.99


def test_symbols_without_returns_for_the_whole_episode_are_not_eligible():
    # A symbol that has not listed yet (NaN returns) must not be drawn for an episode it could not
    # have been held in. Everything else earns the same, so with correct eligibility the randomised
    # statistic equals the real one exactly; if the unlisted symbol were drawable (counted as zero),
    # many randomisations would fall short and the p-value would drop well below 1.
    idx = pd.bdate_range("2020-01-01", periods=300)
    cols = ["A", "B", "C", "D", "E", "NEW"]
    rets = pd.DataFrame(0.01, index=idx, columns=cols)
    rets["NEW"] = np.where(np.arange(300) < 250, np.nan, 0.01)
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    for start in range(0, 240, 30):  # eight episodes, all before NEW lists
        W.iloc[start:start + 30, 0] = 0.5
        W.iloc[start + 15:start + 30, 1] = 0.25  # a second, shorter position to vary the structure
    result = selection_test(W, rets, n_permutations=400)
    assert result.p_value == pytest.approx(1.0)
    assert result.n_candidates == 5  # NEW is never eligible


# ------------------------------------------------------------------ timing

def _regime_weights(index: pd.DatetimeIndex, seed: int, block: int = 21) -> pd.Series:
    rng = np.random.default_rng(seed)
    flags = rng.integers(0, 2, size=int(np.ceil(len(index) / block)))
    return pd.Series(np.repeat(flags, block)[: len(index)].astype(float), index=index)


def test_timing_test_detects_planted_skill():
    rets = _universe(n_days=600, n_syms=1, vol=0.01, seed=31)
    # Be long on the days the asset goes up (look-ahead: planted skill).
    W = (rets > 0).astype(float)
    result = timing_test(W, rets, n_permutations=500)
    assert result.p_value < 0.01


def test_timing_test_is_roughly_calibrated_for_random_regimes():
    pvals = []
    for seed in range(30):
        rets = _universe(n_days=500, n_syms=1, vol=0.01, seed=200 + seed)
        W = pd.DataFrame({"S0": _regime_weights(rets.index, seed=700 + seed)})
        pvals.append(timing_test(W, rets, n_permutations=300, seed=1).p_value)
    assert sum(p < 0.05 for p in pvals) <= 6
    assert np.mean(pvals) > 0.3


def test_timing_shift_pairs_weights_with_later_returns():
    # One position on day 10 in a series whose returns are 0, 1, 2, ... A shift of k pairs that
    # weight with the return k days later, so the randomised statistic must equal (10 + k) / n.
    n = 200
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.DataFrame({"S0": np.arange(n, dtype=float)}, index=idx)
    W = pd.DataFrame({"S0": np.zeros(n)}, index=idx)
    W.iloc[10, 0] = 1.0

    seed, min_shift = 3, 21
    k = int(np.random.default_rng(seed).integers(min_shift, n - min_shift + 1, size=1)[0])
    result = timing_test(W, rets, n_permutations=1, seed=seed, min_shift=min_shift)
    assert result.observed == pytest.approx(10 / n)
    assert result.null_mean == pytest.approx(((10 + k) % n) / n)
