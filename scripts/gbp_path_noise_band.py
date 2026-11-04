#!/usr/bin/env python3
"""
What would GBP 10,000 in the Noise Area strategy have been worth each year?

Runs the backtest in USD from a given start date, then converts the equity
curve to GBP at the actual historical GBP/USD rate for each date, so the
answer is in the currency the money was actually in.

The strategy trades SPY and is flat at the close, so the GBP value of a
position is (USD equity) x (USD per GBP).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.backtest_noise_band import (  # noqa: E402
    fetch, fetch_dividends, stats, max_dd,
)
from src.trading.noise_band import (  # noqa: E402
    NoiseBandConfig, add_noise_band_features, run_backtest,
)


def gbpusd() -> pd.Series:
    """USD per GBP, daily close."""
    import yfinance as yf
    h = yf.Ticker("GBPUSD=X").history(start="2018-01-01", end="2027-01-01", interval="1d")
    s = pd.Series(h["Close"].to_numpy(), index=pd.DatetimeIndex(h.index).tz_localize(None).normalize())
    return s[~s.index.duplicated(keep="last")].dropna()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--gbp", type=float, default=10_000.0)
    ap.add_argument("--sizing", default="full_notional",
                    choices=["full_notional", "vol_target"])
    ap.add_argument("--band-mult", type=float, default=1.0)
    ap.add_argument("--lookback", type=int, default=14)
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    # Need 14 sessions of sigma history plus ~15 for the vol estimate before
    # the first traded day, so pull from a year earlier.
    df = fetch("SPY", start.replace(year=start.year - 1), end, feed="sip")
    if df.empty:
        print("no data")
        return 1
    div = fetch_dividends("SPY")
    fx = gbpusd()

    cfg = NoiseBandConfig(lookback_days=args.lookback, band_mult=args.band_mult,
                          sizing=args.sizing)
    # Features (sigma, VWAP, bands, daily vol) are computed on the FULL
    # history so the warm-up year feeds the band; the backtest then only
    # trades from the start date, so the capital base is the amount
    # actually invested rather than an accumulated 2019 balance.
    out = add_noise_band_features(df, cfg, dividends=div)
    out = out[out["day"] >= pd.Timestamp(args.start).normalize()]

    # GBP -> USD at the start date; the strategy runs in USD.
    fx_pre = gbpusd()
    fx_start = float(fx_pre.reindex([pd.Timestamp(args.start)]).ffill().bfill().iloc[0])
    usd_start = args.gbp / fx_start
    res = run_backtest(out, cfg, initial_capital=usd_start)
    base = run_backtest(out, cfg, initial_capital=usd_start, baseline=True)

    curve = res["equity_curve"].copy()
    bcurve = base["equity_curve"].copy()
    curve.index = pd.DatetimeIndex(curve.index)
    bcurve.index = pd.DatetimeIndex(bcurve.index)

    # The strategy trades SPY, so the capital must be in USD. A GBP amount is
    # converted to USD at the start date's rate, the strategy is run in USD,
    # and the resulting equity curve is converted back to GBP at each day's
    # rate. Everything reported is therefore genuinely GBP.
    first = pd.Timestamp(args.start)
    fx_series = gbpusd()
    fx = fx_series.reindex(curve.index).ffill().bfill()
    fx_start = float(fx_series.reindex([first]).ffill().bfill().iloc[0])
    usd_start = args.gbp / fx_start          # GBP -> USD at the start date
    print(f"\nGBP {args.gbp:,.0f} invested {first.date()}  "
          f"({args.sizing}, VM {args.band_mult}, lookback {args.lookback})")
    print(f"  = USD {usd_start:,.0f} at GBPUSD {fx_start:.4f}")
    n_sessions = int((curve.index >= first).sum())
    print(f"  sessions traded: {n_sessions}\n")

    def gbp(series: pd.Series) -> pd.Series:
        return series * fx

    gbp_full = gbp(curve)
    gbp_base_full = gbp(bcurve)

    print(f"{'year':>6} {'start GBP':>11} {'end GBP':>11} {'P&L':>10} {'return':>9} "
          f"{'intraday-long':>13} {'excess':>9}")
    for yr in range(first.year, pd.Timestamp(args.end).year + 1):
        g = gbp_full[gbp_full.index.year == yr]
        if not len(g):
            continue
        prev_g = gbp_full[gbp_full.index < g.index[0]]
        prev = float(prev_g.iloc[-1]) if len(prev_g) else args.gbp
        end_gbp = float(g.iloc[-1])

        bl = gbp_base_full[gbp_base_full.index.year == yr]
        bl_prev_g = gbp_base_full[gbp_base_full.index < bl.index[0]] if len(bl) else pd.Series(dtype=float)
        bl_prev = float(bl_prev_g.iloc[-1]) if len(bl_prev_g) else args.gbp
        bl_ret = (float(bl.iloc[-1]) / bl_prev - 1) if len(bl) else float("nan")

        ret = end_gbp / prev - 1
        print(f"{yr:>6} {prev:>11,.0f} {end_gbp:>11,.0f} {end_gbp-prev:>10,.0f} "
              f"{ret:>9.1%} {bl_ret:>12.1%} {ret-bl_ret:>9.1%}")

    gbp_curve = gbp_full[gbp_full.index >= first]
    gbp_base = gbp_base_full[gbp_base_full.index >= first]
    final_gbp = float(gbp_curve.iloc[-1])
    final_base = float(gbp_base.iloc[-1])
    usd_ret = float(curve.iloc[-1]) / usd_start - 1
    usd_base = float(bcurve.iloc[-1]) / usd_start - 1

    print(f"\n  final value         GBP {final_gbp:>12,.0f}   "
          f"(USD strategy {usd_ret:+.1%}, intraday-long {usd_base:+.1%})")
    print(f"  GBP return          {final_gbp/args.gbp-1:+.1%}")
    print(f"  GBP intraday-long   GBP {final_base:>12,.0f}   "
          f"({final_base/args.gbp-1:+.1%})")
    print(f"  GBP excess          GBP {final_gbp-final_base:>12,.0f}   "
          f"({final_gbp/args.gbp-final_base/args.gbp:+.1%})")
    fx_end = float(fx.iloc[-1])
    print(f"\n  GBPUSD {fx_start:.4f} -> {fx_end:.4f} "
          f"({fx_end/fx_start-1:+.1%} on the currency itself)")

    gbp_ret = gbp_curve.pct_change().dropna()
    s = stats(gbp_ret, "s")
    b = stats(gbp_base.pct_change().dropna(), "b")
    print(f"\n  Sharpe (GBP) {s['s_sharpe']:.2f}   intraday-long {b['b_sharpe']:.2f}")
    print(f"  worst day   {s['s_worst']:.1%}   intraday-long {b['b_worst']:.1%}")
    print(f"  max DD      {max_dd(gbp_curve):.1%}   intraday-long {max_dd(gbp_base):.1%}")
    print(f"  total cost  ${res['trades']['cost'].sum():,.0f} "
          f"({res['trades']['cost'].sum()/usd_start:.1%} of capital)")

    # True SPY buy-and-hold in GBP: hold from the first session's open to
    # the last close, converted at each date's rate. The intraday-long
    # baseline is flat overnight, so it misses the gap return; this is the
    # number a passive investor would recognise.
    in_range = df[df.index >= pd.Timestamp(args.start)]
    per_day = in_range.groupby(in_range.index.normalize())
    closes = per_day["close"].last()
    opens = per_day["open"].first()
    fx_d = fx_series.reindex(closes.index).ffill().bfill()
    bh_shares = usd_start / opens.iloc[0]
    bh_gbp_end = float(closes.iloc[-1] * bh_shares * fx_d.iloc[-1])
    print(f"\n  SPY buy-and-hold   GBP {bh_gbp_end:>12,.0f}   "
          f"({bh_gbp_end/args.gbp-1:+.1%})   <- holds overnight")
    print(f"  intraday-long       GBP {final_base:>12,.0f}   "
          f"({final_base/args.gbp-1:+.1%})   <- flat at each close")
    print(f"  strategy            GBP {final_gbp:>12,.0f}   "
          f"({final_gbp/args.gbp-1:+.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
