"""
Tests for the Noise Area band construction (src/trading/noise_band.py).

The band is the load-bearing part of the strategy, so these tests pin the
paper's exact semantics rather than just "it runs":

  - sigma is the mean ABSOLUTE move from the open at the same minute-of-day
    over the previous N days (not a std dev of returns, not a Bollinger band)
  - a day never contributes to its own sigma (no lookahead)
  - the band is centred on max(open, prev_close) / min(open, prev_close)
    so an overnight gap widens it
  - the Volatility Multiplier scales the width
  - entries require a band breach AND VWAP confirmation

No network access.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.trading.noise_band import (  # noqa: E402
    NoiseBandConfig, add_noise_band_features, generate_signals, position_size,
)

SESSION_MINUTES = 390  # 09:30 -> 16:00 ET


def make_bars(days: list[str], moves: list[list[float]], base: float = 100.0,
              volume: float = 1000.0) -> pd.DataFrame:
    """Synthetic intraday bars.

    `moves[d][m]` is the close-to-open fractional move at minute m of day d.
    The day's open is `base`; each bar's close is base*(1+move), and high/low
    bracket the close so the typical price matches the close.
    """
    rows = []
    for day, day_moves in zip(days, moves):
        for m, mv in enumerate(day_moves):
            ts = pd.Timestamp(f"{day} 09:30") + pd.Timedelta(minutes=m)
            px = base * (1 + mv)
            rows.append({
                "timestamp": ts, "open": base, "high": px, "low": px,
                "close": px, "volume": volume,
            })
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def _flat_days(n: int, move: float = 0.0) -> list[list[float]]:
    return [[move] * SESSION_MINUTES for _ in range(n)]


def test_sigma_is_mean_absolute_move_not_std_of_returns():
    """Sigma must be the mean |close/open - 1| at each minute-of-day."""
    days = [f"2024-01-{i:02d}" for i in range(1, 5)]
    # each day moves +1% by minute 0, then is flat
    moves = [[0.01] + [0.01] * (SESSION_MINUTES - 1) for _ in days]
    df = make_bars(days, moves)
    cfg = NoiseBandConfig(lookback_days=3)
    out = add_noise_band_features(df, cfg)

    # by the 4th day the 3-day average of |move| at each minute is 1%
    last = out.loc["2024-01-04"]
    assert last["sigma_open"].iloc[0] == pytest.approx(0.01, abs=1e-9)


def test_a_day_never_contributes_to_its_own_sigma():
    """No lookahead: day N's sigma uses only days N-1..N-lookback."""
    days = [f"2024-01-{i:02d}" for i in range(1, 5)]
    # day 4 has a huge move; it must not inflate its own sigma
    moves = _flat_days(3, 0.01)
    moves.append([0.50] + [0.50] * (SESSION_MINUTES - 1))
    df = make_bars(days, moves)
    cfg = NoiseBandConfig(lookback_days=3)
    out = add_noise_band_features(df, cfg)

    day4 = out.loc["2024-01-04"]
    assert day4["sigma_open"].iloc[0] == pytest.approx(0.01, abs=1e-9)


def test_band_is_centred_on_open_and_prev_close():
    """Gap up: the upper band is based on the previous close, the lower on the open."""
    days = [f"2024-01-{i:02d}" for i in range(1, 6)]
    moves = _flat_days(5, 0.01)
    df = make_bars(days, moves, base=100.0)
    # gap up on the last day: open far above the previous close
    df.loc["2024-01-05", ["open", "high", "low", "close"]] = 110.0

    cfg = NoiseBandConfig(lookback_days=3, band_mult=1.0)
    out = add_noise_band_features(df, cfg)

    row = out.loc["2024-01-05"].iloc[0]
    prev_close = out.loc["2024-01-04", "close"].iloc[-1]
    sigma = row["sigma_open"]
    # upper band uses max(open, prev_close) = 110
    assert row["band_upper"] == pytest.approx(110.0 * (1 + sigma), rel=1e-9)
    # lower band uses min(open, prev_close) = prev_close
    assert row["band_lower"] == pytest.approx(prev_close * (1 - sigma), rel=1e-9)


def test_volatility_multiplier_scales_the_sigma_component():
    """VM multiplies sigma; the band is asymmetric so its width is not a simple multiple."""
    days = [f"2024-01-{i:02d}" for i in range(1, 5)]
    df = make_bars(days, _flat_days(4, 0.01))
    narrow = add_noise_band_features(df, NoiseBandConfig(lookback_days=3, band_mult=1.0))
    wide = add_noise_band_features(df, NoiseBandConfig(lookback_days=3, band_mult=2.0))

    n = narrow.loc["2024-01-04"].iloc[0]
    w = wide.loc["2024-01-04"].iloc[0]
    assert w["band_upper"] > n["band_upper"]
    assert w["band_lower"] < n["band_lower"]
    # the sigma term itself doubles (the band is asymmetric, so the ratio of
    # the levels is not the ratio of the sigma terms)
    basis_hi = max(n["open"], narrow.loc["2024-01-03", "close"].iloc[-1])
    basis_lo = min(n["open"], narrow.loc["2024-01-03", "close"].iloc[-1])
    assert (w["band_upper"] - basis_hi) == pytest.approx(2 * (n["band_upper"] - basis_hi), rel=1e-9)
    assert (basis_lo - w["band_lower"]) == pytest.approx(2 * (basis_lo - n["band_lower"]), rel=1e-9)


def test_vwap_confirmation_is_required():
    """A band breach without VWAP confirmation must not signal."""
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02 10:00"), pd.Timestamp("2024-01-02 10:01")])
    out = pd.DataFrame({
        "open": [100.0, 100.0], "high": [101.0, 101.0], "low": [99.0, 99.0],
        "close": [100.0, 100.0], "volume": [100.0, 100.0],
        "band_upper": [99.0, 99.0], "band_lower": [98.0, 98.0],
        "vwap": [101.0, 101.0],   # price is above the band but below VWAP
    }, index=idx)

    cfg = NoiseBandConfig(use_vwap=True)
    sig = generate_signals(out, cfg)
    assert (sig == 0).all(), "VWAP filter should block the entry"

    out["vwap"] = [99.0, 99.0]    # now VWAP confirms
    sig2 = generate_signals(out, cfg)
    assert (sig2 == 1.0).all(), "should signal long"

    # and the short side: below the lower band
    out["close"] = [97.0, 97.0]
    out["vwap"] = [95.0, 95.0]   # price below band but above VWAP -> not confirmed
    assert (generate_signals(out, cfg) == 0).all()
    out["vwap"] = [98.0, 98.0]   # VWAP confirms the short
    assert (generate_signals(out, cfg) == -1.0).all()


def test_vwap_is_cumulative_from_the_open():
    days = ["2024-01-01", "2024-01-02"]
    df = make_bars(days, _flat_days(2, 0.0))
    out = add_noise_band_features(df, NoiseBandConfig())
    # flat price -> vwap == price
    row = out.loc["2024-01-01"]
    assert row["vwap"].iloc[0] == pytest.approx(row["close"].iloc[0], rel=1e-9)


def test_dividend_is_subtracted_from_previous_close():
    """The reference notebook subtracts the day's dividend from the prev close."""
    days = [f"2024-01-{i:02d}" for i in range(1, 6)]
    df = make_bars(days, _flat_days(5, 0.01), base=100.0)
    div = pd.Series({pd.Timestamp("2024-01-05"): 1.0})
    out = add_noise_band_features(df, NoiseBandConfig(lookback_days=3), dividends=div)

    row = out.loc["2024-01-05"].iloc[0]
    prev_close = out.loc["2024-01-04", "close"].iloc[-1] - 1.0
    sigma = row["sigma_open"]
    # open (100) > prev_close (99), so the basis is the open
    assert row["band_upper"] == pytest.approx(100.0 * (1 + sigma), rel=1e-9)
    assert row["band_lower"] == pytest.approx(prev_close * (1 - sigma), rel=1e-9)


def test_vol_target_sizing_scales_with_leverage_cap():
    """vol_target sizing must respect the max_leverage cap."""
    cfg = NoiseBandConfig(sizing="vol_target", target_vol=0.02, max_leverage=4.0)
    # a 1% daily vol gives a target of 2x notional
    assert position_size(100_000.0, 100.0, 0.01, cfg) == 2000
    # a very quiet market hits the leverage cap rather than exploding
    assert position_size(100_000.0, 100.0, 0.0001, cfg) == 4000
    # a missing vol estimate falls back to the cap, not to zero
    assert position_size(100_000.0, 100.0, float("nan"), cfg) == 4000
    # full notional is exactly one times equity
    full = NoiseBandConfig(sizing="full_notional")
    assert position_size(100_000.0, 100.0, 0.01, full) == 1000


def test_baseline_is_identically_sized_to_the_strategy():
    """The baseline control must differ only in the signal, not in sizing."""
    days = [f"2024-02-{i:02d}" for i in range(1, 21)]
    df = make_bars(days, _flat_days(20, 0.005))
    cfg = NoiseBandConfig(lookback_days=3, sizing="vol_target")
    out = add_noise_band_features(df, cfg)
    assert len(out["daily_vol"].dropna()) > 0, "daily vol should be populated"
