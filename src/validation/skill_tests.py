"""Selection and timing skill tests for strategies that expose per-date weights.

The questions these answer are different from "is the mean return positive?":

* ``selection_test``: did the strategy pick better *symbols* than chance, given when it held
  positions, for how long, how large, and on which side?
* ``timing_test``: did the strategy hold its positions at better *times* than chance, given the
  same positions, durations and sizes?

Inputs (both are date x symbol frames on the same dates):

``weights``
    Signed exposure held during each period, as a fraction of equity. +0.10 means long 10% of
    equity in that symbol during the period; negative is short. Zero means no position.
``returns``
    The per-period return of every symbol in the *universe* the strategy could have chosen from
    (including symbols it never held). ``returns`` columns must contain every ``weights`` column.

Statistic: the mean per-period *gross* return of the weights, ``mean_t sum_i w[t,i] * r[t,i]``.
It is computed the same way for the real weights and for every randomised version, so costs,
cash interest and execution details cancel out of the comparison. It is not the strategy's
net return.

Selection null
    Split the weights into position *episodes*: runs of consecutive periods in which one symbol
    is held on one side. Replace the symbol of every episode with one drawn uniformly at random
    from the universe, keeping the episode's dates, weights and sign. This preserves timing,
    holding periods, position sizes, concurrency and long/short structure, so it removes only
    the choice of symbol. Draws are with replacement, so two episodes may land on the same
    symbol; a symbol is only eligible if it has returns for every period of the episode.
    Market-wide timing earns no credit here (a random symbol is exposed at the same times),
    so a strategy that is merely long the market when the market rises will not pass.

Timing null
    Circularly shift the whole weights matrix in time by a random offset, keeping the returns
    fixed. This preserves every position's size, duration and symbol and the cross-section at
    each date, and removes only the alignment with the returns. It is the natural test for
    single-asset or few-asset strategies where there is nothing to select among. A constant
    (buy-and-hold) exposure cannot beat it, since shifting leaves the weights unchanged.

Both are one-sided: the p-value is the share of randomisations whose statistic is at least as
large as the real one (with +1 on numerator and denominator, so it is never exactly zero).

Limits
    * Episodes inside one backtest are treated as independent draws. Where the real strategy's
      picks are persistent for a reason (e.g. volatility ranks) this is the usual random-pick
      null, not a model of that persistence.
    * With few episodes the p-value is coarse; it cannot be smaller than ``1/(n_permutations+1)``.
    * Neither test corrects for the number of strategies or parameter sets you tried.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Ties between the real and randomised statistic are real (e.g. every symbol has the same
# returns, or the weights are constant), so compare with a tolerance far above float rounding
# (statistics are mean per-period returns, typically 1e-4) and far below any real difference.
_TIE_TOL = 1e-12

# Selection needs something to select among.
MIN_UNIVERSE_FOR_SELECTION = 5


@dataclass(frozen=True)
class SkillTestResult:
    name: str  # "selection" or "timing"
    observed: float  # mean per-period gross return of the real weights
    null_mean: float
    null_std: float
    p_value: float
    n_permutations: int
    n_periods: int
    n_episodes: int | None = None  # selection only
    n_candidates: int | None = None  # selection only: symbols eligible for at least one episode

    @property
    def percentile(self) -> float:
        return (1.0 - self.p_value) * 100.0


def _prepare(weights: pd.DataFrame, returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """Align the two frames and return (W, R, columns) as float arrays.

    ``W`` is zero-filled. ``R`` keeps NaN (a symbol with no return that period) so the selection
    test can avoid drawing it; the real portfolio treats NaN as a zero return, as the backtests do.
    """
    if not isinstance(weights, pd.DataFrame) or not isinstance(returns, pd.DataFrame):
        raise TypeError("weights and returns must both be DataFrames")
    missing = [c for c in weights.columns if c not in returns.columns]
    if missing:
        raise ValueError(f"returns is missing weights columns: {missing[:5]}")
    if weights.index.has_duplicates or returns.index.has_duplicates:
        raise ValueError("weights and returns must have unique dates")
    not_in_returns = weights.index.difference(returns.index)
    if len(not_in_returns) > 0:
        raise ValueError(f"returns has no rows for {len(not_in_returns)} weights dates")

    columns = list(returns.columns)
    W = weights.reindex(columns=columns, fill_value=0.0).fillna(0.0).to_numpy(dtype=float)
    R = returns.reindex(weights.index).to_numpy(dtype=float)
    return W, R, columns


def _episodes(W: np.ndarray) -> list[tuple[int, int, int]]:
    """Runs of consecutive nonzero weights of one sign, as (column, start, end_exclusive)."""
    n_rows, n_cols = W.shape
    episodes: list[tuple[int, int, int]] = []
    for j in range(n_cols):
        col = W[:, j]
        if not col.any():
            continue
        sign = np.sign(col)
        cuts = np.flatnonzero(np.diff(sign) != 0) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [n_rows]))
        for s, e in zip(starts, ends):
            if sign[s] != 0:
                episodes.append((j, int(s), int(e)))
    return episodes


def selection_test(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    n_permutations: int = 2000,
    seed: int = 42,
) -> SkillTestResult:
    """Did the strategy pick better symbols than random picks held over the same periods?"""
    W, R, _ = _prepare(weights, returns)
    n_rows, n_cols = W.shape
    episodes = _episodes(W)
    if not episodes:
        raise ValueError("weights contain no positions")

    nan_mask = np.isnan(R)
    Rz = np.where(nan_mask, 0.0, R)
    nan_cum = np.vstack([np.zeros((1, n_cols)), np.cumsum(nan_mask, axis=0)])

    n_ep = len(episodes)
    contrib = np.empty((n_ep, n_cols))  # contrib[e, j]: episode e's weights applied to symbol j
    valid = np.empty((n_ep, n_cols), dtype=bool)  # symbol j has returns for the whole episode
    own = np.empty(n_ep, dtype=int)
    for e, (j, s, t) in enumerate(episodes):
        contrib[e] = W[s:t, j] @ Rz[s:t]
        valid[e] = (nan_cum[t] - nan_cum[s]) == 0
        own[e] = j
        if not valid[e].any():  # nothing eligible: the episode keeps its own symbol
            valid[e, j] = True

    observed_sum = float(contrib[np.arange(n_ep), own].sum())
    observed = observed_sum / n_rows

    n_valid = valid.sum(axis=1)
    # For each episode, column ids with the eligible ones first, so a uniform integer in
    # [0, n_valid) picks a uniform eligible symbol.
    eligible_first = np.argsort(~valid, axis=1, kind="stable")

    rng = np.random.default_rng(seed)
    null_sums = np.empty(n_permutations)
    rows = np.arange(n_ep)[None, :]
    chunk = max(1, min(n_permutations, 4_000_000 // max(n_ep, 1)))
    done = 0
    while done < n_permutations:
        m = min(chunk, n_permutations - done)
        k = np.minimum((rng.random((m, n_ep)) * n_valid).astype(int), n_valid - 1)
        picked = eligible_first[rows, k]
        null_sums[done:done + m] = contrib[rows, picked].sum(axis=1)
        done += m

    null = null_sums / n_rows
    p_value = (1 + int((null >= observed - _TIE_TOL).sum())) / (1 + n_permutations)
    return SkillTestResult(
        name="selection",
        observed=observed,
        null_mean=float(null.mean()),
        null_std=float(null.std()),
        p_value=float(p_value),
        n_permutations=n_permutations,
        n_periods=n_rows,
        n_episodes=n_ep,
        n_candidates=int(valid.any(axis=0).sum()),
    )


def timing_test(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    n_permutations: int = 1000,
    seed: int = 42,
    min_shift: int | None = None,
) -> SkillTestResult:
    """Did the strategy hold its positions at better times than the same positions at random times?"""
    W, R, _ = _prepare(weights, returns)
    n_rows = W.shape[0]
    if n_rows < 60:
        raise ValueError(f"need at least 60 periods for a timing test, got {n_rows}")

    active = W.any(axis=0)
    if not active.any():
        raise ValueError("weights contain no positions")
    Wa = W[:, active]
    Ra = np.nan_to_num(R[:, active], nan=0.0)

    if min_shift is None:
        min_shift = max(21, int(0.05 * n_rows))
    if 2 * min_shift >= n_rows:
        raise ValueError("series too short for the minimum shift")

    observed = float((Wa * Ra).sum()) / n_rows

    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, n_rows - min_shift + 1, size=n_permutations)
    null = np.empty(n_permutations)
    for i, k in enumerate(shifts):
        # weights at t' meet the returns at t' + k (wrapping around)
        null[i] = (Wa * np.roll(Ra, -int(k), axis=0)).sum() / n_rows

    p_value = (1 + int((null >= observed - _TIE_TOL).sum())) / (1 + n_permutations)
    return SkillTestResult(
        name="timing",
        observed=observed,
        null_mean=float(null.mean()),
        null_std=float(null.std()),
        p_value=float(p_value),
        n_permutations=n_permutations,
        n_periods=n_rows,
    )
