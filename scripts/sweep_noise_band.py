#!/usr/bin/env python3
"""
Parameter sweep for the Noise Area strategy.

Sweeps the Volatility Multiplier and lookback on cached SPY 1-minute bars and
reports each against the unconditional intraday-long baseline and SPY buy & hold.

The baseline is long from the first mark to the close and flat overnight, so
it is NOT buy-and-hold -- see src/trading/noise_band.py.

The point is not to find the best parameter -- it is to see whether the edge
survives a range of parameters.  A strategy that only works at one setting is
a curve fit.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.backtest_noise_band import (  # noqa: E402
    fetch, fetch_dividends, stats, yearly_excess,
)
from src.trading.noise_band import (  # noqa: E402
    NoiseBandConfig, add_noise_band_features, run_backtest,
)


def main() -> int:
    df = fetch("SPY", datetime(2022, 1, 1), datetime(2026, 1, 1))
    if df.empty:
        print("no data")
        return 1
    print(f"bars {len(df):,}\n")
    div = fetch_dividends("SPY")
    ic = 100_000.0

    print(f"{'VM':>5} {'lookback':>9} {'total':>9} {'base':>9} {'excess':>9} "
          f"{'Sharpe':>7} {'trades':>7}   yearly excess")
    for lookback in (14, 90):
        for vm in (0.5, 1.0, 1.5, 2.0):
            cfg = NoiseBandConfig(lookback_days=lookback, band_mult=vm)
            out = add_noise_band_features(df, cfg, dividends=div)
            res = run_backtest(out, cfg, initial_capital=ic)
            base = run_backtest(out, cfg, initial_capital=ic, baseline=True)
            daily = res["equity_curve"].pct_change().dropna()
            s = stats(daily, "s")
            bs = stats(base["equity_curve"].pct_change().dropna(), "b")
            ye = yearly_excess(res, base, ic)
            ystr = "  ".join(f"{y}:{v:+.0%}" for y, v in sorted(ye.items()))
            print(f"{vm:5.1f} {lookback:9d} {s['s_total']:+9.1%} {bs['b_total']:+9.1%} "
                  f"{s['s_total']-bs['b_total']:+9.1%} {s['s_sharpe']:+7.2f} "
                  f"{res['n_trades']:7d}   {ystr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
