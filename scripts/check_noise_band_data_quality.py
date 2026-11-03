#!/usr/bin/env python3
"""
Data-quality sensitivity check for the Noise Area strategy.

Alpaca's IEX feed leaves gaps in the 1-minute series.  A session with only a
few dozen bars cannot produce a meaningful band, and a session with a hole in
the middle can produce a fake signal.  This re-runs the strategy keeping only
sessions with at least MIN_BARS bars, and reports whether the edge survives.

Run with --feed iex to reproduce the gap problem; the default SIP feed is
complete and this check then passes trivially.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.backtest_noise_band import (  # noqa: E402
    fetch, fetch_dividends, stats, yearly_excess,
)
from src.trading.noise_band import (  # noqa: E402
    NoiseBandConfig, add_noise_band_features, run_backtest,
)

MIN_BARS = 300   # a full session is 390 bars


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", default="sip", choices=["sip", "iex"])
    args = ap.parse_args()

    df = fetch("SPY", datetime(2022, 1, 1), datetime(2026, 1, 1), feed=args.feed)
    if df.empty:
        print("no data")
        return 1
    ic = 100_000.0
    days = pd.DatetimeIndex(df.index).normalize()
    counts = pd.Series(days).value_counts()
    good = set(counts[counts >= MIN_BARS].index)
    print(f"bars {len(df):,}   sessions {len(counts):,}   "
          f"kept (>= {MIN_BARS} bars) {len(good):,}")

    df = df[days.isin(good)]
    print(f"bars after filter {len(df):,}\n")

    div = fetch_dividends("SPY")
    cfg = NoiseBandConfig(lookback_days=14, band_mult=1.0)
    out = add_noise_band_features(df, cfg, dividends=div)
    res = run_backtest(out, cfg, initial_capital=ic)
    base = run_backtest(out, cfg, initial_capital=ic, baseline=True)

    daily = res["equity_curve"].pct_change().dropna()
    s = stats(daily, "s")
    bs = stats(base["equity_curve"].pct_change().dropna(), "b")
    print(f"{'':22}{'strategy':>12}{'always-long':>14}")
    print(f"  total return      {s['s_total']:+12.1%}{bs['b_total']:+14.1%}")
    print(f"  Sharpe            {s['s_sharpe']:+12.2f}{bs['b_sharpe']:+14.2f}")
    print(f"  excess            {s['s_total']-bs['b_total']:+12.1%}")

    t = res["trades"]
    if len(t):
        ti = t.set_index("day")
        print("\n  by year:")
        ye = yearly_excess(res, base, ic)
        for yr, g in ti.groupby(pd.to_datetime(ti.index).year):
            sy = g["pnl"].sum() / ic
            print(f"    {yr}  strategy {sy:+7.1%}   excess {ye.get(yr, 0.0):+7.1%}")

        # how concentrated is the return?
        pnl = t["pnl"]
        print(f"\n  best 5 days:  ${pnl.nlargest(5).sum():,.0f} of ${pnl.sum():,.0f} total")
        print(f"  worst 5 days: ${pnl.nsmallest(5).sum():,.0f}")
        print(f"  median day:   ${pnl.median():,.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
