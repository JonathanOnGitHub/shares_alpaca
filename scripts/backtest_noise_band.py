#!/usr/bin/env python3
"""
Backtest the Noise Area intraday momentum strategy on SPY 1-minute bars.

Zarattini, Aziz & Barbon (2024), "Beat the Market", SSRN 4824172.

Data comes from the Alpaca API (1-minute bars) and is cached to pickle so
repeat runs do not re-hit the API.

    python scripts/backtest_noise_band.py --start 2022-01-01 --end 2026-01-01

Reports the strategy against SPY buy & hold on the same days, with the
unconditional baseline (same exposure schedule and sizing, no signal) as the
control.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.trading.noise_band import (  # noqa: E402
    NoiseBandConfig, add_noise_band_features, run_backtest,
)

CACHE = Path(os.environ.get("NOISE_BAND_CACHE", ROOT / "data" / "cache_noise_band"))
SESSION_BARS = 390


def _regular_session_index(idx: pd.DatetimeIndex) -> pd.Index:
    """Boolean mask of bars inside the 09:30-15:59 ET regular session."""
    t = idx.time
    return (t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("15:59").time())


def fetch(symbol: str, start: datetime, end: datetime,
          feed: str = "sip") -> pd.DataFrame:
    """1-minute bars for `symbol`, cached per month.

    The SIP feed is used by default because IEX leaves gaps in the intraday
    series; a session missing bars cannot produce a meaningful band.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur < end:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        stop = min(nxt, end)
        key = CACHE / f"{symbol}_{feed}_{cur:%Y_%m}.pkl"
        if key.exists():
            df = pickle.load(open(key, "rb"))
        else:
            from src.data.alpaca_client import AlpacaClient
            client = AlpacaClient(paper=True)
            bars = client.get_bars([symbol], "Minute", start=cur, end=stop, feed=feed)
            df = bars.get(symbol, pd.DataFrame())
            if not df.empty:
                df = df.copy()
                df.index = pd.DatetimeIndex(df.index).tz_convert("America/New_York")
                df = df[_regular_session_index(df.index)]
                df.index = df.index.tz_localize(None)
            pickle.dump(df, open(key, "wb"))
        if df is not None and not df.empty:
            frames.append(df)
        cur = nxt
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return pd.DataFrame(out)


def fetch_dividends(symbol: str) -> pd.Series:
    """Cash dividends keyed by ex-date (ET midnight).

    Failure here is not fatal: the band just centres on the unadjusted
    previous close, which only matters on the four ex-dividend days a year.
    """
    key = CACHE / f"{symbol}_dividends.pkl"
    if key.exists():
        try:
            return pickle.load(open(key, "rb"))
        except Exception:
            pass
    try:
        import yfinance as yf
        div = yf.Ticker(symbol).dividends
    except Exception as e:
        print(f"  (dividend fetch failed: {e})")
        return pd.Series(dtype=float)
    if div is None or len(div) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(div.to_numpy(), index=pd.DatetimeIndex(div.index).tz_convert(
        "America/New_York").normalize().tz_localize(None))
    s = s.groupby(level=0).sum().astype(float)
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(s, open(key, "wb"))
    return s


def stats(daily_ret: pd.Series, label: str) -> dict:
    r = daily_ret.dropna()
    if r.empty:
        return {f"{label}_sharpe": 0.0, f"{label}_total": 0.0}
    total = float((1 + r).prod() - 1)
    vol = float(r.std() * np.sqrt(252))
    ann = float((1 + total) ** (252 / len(r)) - 1)
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    return {
        f"{label}_total": total,
        f"{label}_ann": ann,
        f"{label}_vol": vol,
        f"{label}_sharpe": sharpe,
        f"{label}_worst": float(r.min()),
    }


def max_dd(equity: pd.Series) -> float:
    """Peak-to-trough drawdown of an equity curve."""
    if equity is None or not len(equity):
        return 0.0
    return float((equity / equity.cummax() - 1).min())


def yearly_excess(res: dict, base_res: dict, initial_capital: float) -> dict:
    """Excess over the intraday-long baseline, by calendar year."""
    t = res["trades"]
    if not len(t):
        return {}
    ti = t.set_index("day")
    base_eq = base_res["equity_curve"]
    out = {}
    for yr, g in ti.groupby(pd.to_datetime(ti.index).year):
        be = base_eq.loc[[d for d in g.index if d in base_eq.index]]
        base_yr = (be.iloc[-1] / be.iloc[0] - 1) if len(be) > 1 else 0.0
        out[yr] = g["pnl"].sum() / initial_capital - base_yr
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--lookback", type=int, default=14)
    ap.add_argument("--band-mult", type=float, default=1.0)
    ap.add_argument("--trade-freq", type=int, default=30)
    ap.add_argument("--feed", default="sip", choices=["sip", "iex"])
    ap.add_argument("--sizing", default="full_notional",
                    choices=["full_notional", "vol_target"])
    ap.add_argument("--min-bars", type=int, default=0,
                    help="drop sessions with fewer bars than this (0 = keep all)")
    ap.add_argument("--slippage", type=float, default=0.0,
                    help="per-share slippage; the paper models none")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    print(f"fetching {args.symbol} 1-min bars ({args.feed}) {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
    df = fetch(args.symbol, start, end, feed=args.feed)
    if df.empty:
        print("no data")
        return 1
    print(f"bars: {len(df):,}  {df.index[0]} -> {df.index[-1]}")

    if args.min_bars:
        days = pd.DatetimeIndex(df.index).normalize()
        counts = pd.Series(days).value_counts()
        good = set(counts[counts >= args.min_bars].index)
        df = df[days.isin(good)]
        print(f"  kept {len(good)}/{len(counts)} sessions (>= {args.min_bars} bars)")

    div = fetch_dividends(args.symbol)

    cfg = NoiseBandConfig(
        lookback_days=args.lookback,
        band_mult=args.band_mult,
        trade_freq=args.trade_freq,
        sizing=args.sizing,
        slippage=args.slippage,
    )
    out = add_noise_band_features(df, cfg, dividends=div)
    initial_capital = 100_000.0
    res = run_backtest(out, cfg, initial_capital=initial_capital)
    base_res = run_backtest(out, cfg, initial_capital=initial_capital, baseline=True)

    eq = res["equity_curve"]
    daily = eq.pct_change().dropna()
    base_daily = base_res["equity_curve"].pct_change().dropna()
    trades = res["trades"]
    spy_daily = trades["spy_ret"] if len(trades) else pd.Series(dtype=float)
    spy_curve = (1 + spy_daily.fillna(0.0)).cumprod()

    print(f"\nsessions: {res['n_days']}   trades: {res['n_trades']:,}")
    if len(trades):
        print(f"total cost: ${trades['cost'].sum():,.0f}")

    s = {}
    s.update(stats(daily, "strat"))
    s.update(stats(base_daily, "base"))
    s.update(stats(spy_daily, "spy"))
    print("\n                     strategy   intraday-long   SPY B&H")
    print(f"  total return   {s['strat_total']:+12.1%}  {s['base_total']:+12.1%}  {s['spy_total']:+12.1%}")
    print(f"  annualised     {s['strat_ann']:+12.1%}  {s['base_ann']:+12.1%}  {s['spy_ann']:+12.1%}")
    print(f"  volatility     {s['strat_vol']:+12.1%}  {s['base_vol']:+12.1%}  {s['spy_vol']:+12.1%}")
    print(f"  Sharpe         {s['strat_sharpe']:+12.2f}  {s['base_sharpe']:+12.2f}  {s['spy_sharpe']:+12.2f}")
    print(f"  worst day      {s['strat_worst']:+12.1%}  {s['base_worst']:+12.1%}  {s['spy_worst']:+12.1%}")
    print(f"  max drawdown   {max_dd(eq):+12.1%}  {max_dd(base_res['equity_curve']):+12.1%}"
          f"  {max_dd(spy_curve):+12.1%}")
    print(f"\n  EXCESS over intraday-long: {s['strat_total']-s['base_total']:+.1%}")
    if len(trades):
        print(f"  hit rate (days>0): {100*(daily>0).mean():.1f}%")
        print(f"  mean gross/day:  ${trades['gross'].mean():,.1f}")
        print(f"  mean cost/day:   ${trades['cost'].mean():,.1f}")
        # Concentration: a return carried by a handful of days is not a
        # repeatable edge, it is a few lucky fills.
        pnl = trades["pnl"]
        print(f"  best 5 days:    ${pnl.nlargest(5).sum():,.0f} of ${pnl.sum():,.0f} total")
        print(f"  worst 5 days:   ${pnl.nsmallest(5).sum():,.0f}")
        print(f"  median day:     ${pnl.median():,.1f}")

    # Sign stability by year. The repo's own fade study found a real edge that
    # flipped sign across years; a strategy that only works in some regimes is
    # not deployable.
    if len(trades):
        print("\n  by year:   strategy    intraday-long      excess")
        ye = yearly_excess(res, base_res, initial_capital)
        t = trades.set_index("day")
        for yr, g in t.groupby(pd.to_datetime(t.index).year):
            strat_yr = g["pnl"].sum() / initial_capital
            print(f"    {yr}   {strat_yr:+10.1%}  {'':12}  {ye.get(yr, 0.0):+10.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
