"""
Intraday FVG Strategy — Backtest Module

Target: 0–10 trades/day, flat by 14:00 EST
Universe: Top 20 liquid US stocks/ETFs
Bar frequency: 5-minute (yfinance ~60-day lookback)

Entry logic (Continuation):
  Bull FVG forms on bar i-1: low[i-1] > high[i-2]  → zone [high[i-2], low[i-1]]
  Entry: bar i closes ABOVE zone top (displacement confirmed)
  Bear FVG: high[i-1] < low[i-2] → zone [low[i-2], high[i-1]]
  Entry: bar i closes BELOW zone bottom

Stop:  0.5 × ATR(14) from entry
Target: 2× risk (2:1 R:R)
Risk:   $100 per trade flat; max 3 concurrent positions
Filter: FVG must form within first 90 min of RTH (09:30–11:00 EST)
Volume: confirmation bar volume > median of prior 20 bars
Exit:   stop / target / 14:00 EST hard close
Friction: 1.0 bps round-trip (realistic: 0.5 bps spread + 0.25 bps slippage + 0.25 bps commission)
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────────────

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder-smoothed ATR."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    if n > period:
        atr[period] = np.mean(tr[1:period + 1])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _atr_scalar(high: float, low: float, close: float,
                 prev_close: float, period: int = 14,
                 prev_atr: Optional[float] = None) -> float:
    """Compute a single ATR value given current and prior close."""
    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
    if prev_atr is None:
        return tr
    return (prev_atr * (period - 1) + tr) / period


def _fvg_zones_5m(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    timestamps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect FVGs on 5-min bars. Returns bull_top, bull_bot, bear_top, bear_bot arrays.

    FVG forms on bar i-1 (gap bar); zone belongs to bar i-1.
    Only FVGs forming between 09:30 and 11:00 EST are flagged.
    """
    n = len(close)
    bull_top = np.full(n, np.nan)
    bull_bot = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bot = np.full(n, np.nan)

    # Pre-compute 09:30 and 11:00 EST timestamps for each trading day
    # to filter FVG formation window
    for i in range(2, n):
        ts = timestamps[i]
        # Extract hour and minute from timestamp (assumes tz-aware or UTC with US conversion)
        # timestamps are timezone-aware pandas timestamps
        hour   = ts.hour if hasattr(ts, 'hour') else int(str(ts)[11:13])
        minute = ts.minute if hasattr(ts, 'minute') else int(str(ts)[14:16])

        # Skip if outside 09:30–11:00 EST formation window
        total_min = hour * 60 + minute if isinstance(hour, int) else 0
        if not (570 <= total_min <= 660):   # 09:30 = 570 min, 11:00 = 660 min
            continue

        p_o, p_h, p_l = open_[i - 2], high[i - 2], low[i - 2]
        g_o, g_h, g_l = open_[i - 1], high[i - 1], low[i - 1]

        # Bullish FVG: gap candle (i-1) body entirely above prior candle (i-2) body
        if g_l > p_h:
            bull_top[i - 1] = p_h
            bull_bot[i - 1] = g_l

        # Bearish FVG
        if g_h < p_l:
            bear_top[i - 1] = g_h
            bear_bot[i - 1] = p_l

    return bull_top, bull_bot, bear_top, bear_bot


def _max_drawdown(rets: List[float]) -> Tuple[float, float]:
    """Worst peak-to-trough drawdown and peak equity index."""
    if not rets:
        return 0.0, 0
    cum = np.cumprod([1 + r for r in rets])
    running_max = np.maximum.accumulate(cum)
    dd = (cum / running_max) - 1
    worst_idx = int(np.argmin(dd))
    return float(np.min(dd) * 100), worst_idx


def _apply_friction(pnl: float, entry_px: float, shares: int, direction: int) -> float:
    """Apply 1 bps round-trip friction."""
    if shares <= 0 or entry_px <= 0:
        return pnl
    notional = entry_px * shares
    cost = notional * 0.0001   # 1 bps
    return pnl - cost


# ── core backtest ─────────────────────────────────────────────────────────────

def _run_ticker(
    ticker: str,
    df: pd.DataFrame,
    risk_dollars: float = 100.0,
    max_risk_mult: float = 2.0,
    cost_bps: float = 1.0,
    min_volume_ratio: float = 1.0,
) -> Dict:
    """
    Run intraday FVG backtest on a single ticker.

    Returns dict with summary stats and trade list.
    """
    df = df.dropna().copy()
    df = df.sort_index()
    if len(df) < 50:
        return {"error": f"Insufficient data ({len(df)} rows)", "trades": []}

    o = np.asarray(df["Open"].values, dtype=float)
    h = np.asarray(df["High"].values, dtype=float)
    l = np.asarray(df["Low"].values, dtype=float)
    c = np.asarray(df["Close"].values, dtype=float)
    v = np.asarray(df["Volume"].values, dtype=float)
    # Keep DatetimeIndex for .hour/.minute access
    ts_idx = df.index

    n = len(c)
    atr14 = _atr(h, l, c, period=14)

    # Compute median volume over prior 20 bars (rolling)
    med_vol = np.full(n, np.nan)
    for i in range(20, n):
        med_vol[i] = np.median(v[i - 20:i])

    bull_top, bull_bot, bear_top, bear_bot = _fvg_zones_5m(o, h, l, c, ts_idx)
    bull_valid = ~(np.isnan(bull_top) | np.isnan(bull_bot))
    bear_valid = ~(np.isnan(bear_top) | np.isnan(bear_bot))

    trades: List[Dict] = []
    open_trade: Optional[Dict] = None   # {ticker, direction, entry_px, stop_px, target_px, shares, entry_bar}

    for i in range(2, n):
        date = ts_idx[i]

        # ── Time-of-day filter: 09:30–14:00 EST hard window ─────────────────
        hour   = date.hour if hasattr(date, 'hour') else int(str(date)[11:13])
        minute = date.minute if hasattr(date, 'minute') else int(str(date)[14:16])
        total_min = hour * 60 + minute
        # Trading window: 09:30 (570) to 16:00 (960) EST; we trade 09:30–14:00
        if not (570 <= total_min <= 840):
            # Hard exit any open position at/before 14:00 EST close
            if open_trade is not None:
                exit_px = float(c[i])
                exit_price = max(open_trade["stop_px"], exit_px) if open_trade["direction"] == 1 else min(open_trade["stop_px"], exit_px)
                pnl = (exit_price - open_trade["entry_px"]) * open_trade["shares"] * open_trade["direction"]
                pnl = _apply_friction(pnl, open_trade["entry_px"], open_trade["shares"], open_trade["direction"])
                trades.append({
                    "ticker": ticker, "direction": "long" if open_trade["direction"] == 1 else "short",
                    "entry": open_trade["entry_px"], "exit": exit_price,
                    "shares": open_trade["shares"], "pnl": round(pnl, 2),
                    "exit_reason": "time_close",
                    "entry_time": str(open_trade["entry_time"]), "exit_time": str(date),
                    "bars_held": i - open_trade["entry_bar"],
                })
                open_trade = None
            continue

        bar_high  = float(h[i])
        bar_low   = float(l[i])
        bar_close = float(c[i])
        bar_vol   = float(v[i])

        # ── Volume filter on confirmation bar ───────────────────────────────
        vol_ok = (med_vol[i] > 0) and (bar_vol >= min_volume_ratio * med_vol[i])

        # ── Check if any FVG fires at this bar ─────────────────────────────
        for direction in [1, -1]:   # 1 = long, -1 = short
            if direction == 1:
                zone_active = bull_valid[i - 1]
                if not zone_active:
                    continue
                zone_top = bull_top[i - 1]
                zone_bot = bull_bot[i - 1]
            else:
                zone_active = bear_valid[i - 1]
                if not zone_active:
                    continue
                zone_top = bear_top[i - 1]
                zone_bot = bear_bot[i - 1]

            if open_trade is not None:
                continue   # already in a position in this ticker

            # Entry: bar closes beyond zone boundary (displacement)
            if direction == 1:
                entry_triggered = bar_close > zone_top
            else:
                entry_triggered = bar_close < zone_bot

            if not entry_triggered:
                continue

            # Volume filter
            if not vol_ok:
                continue

            # ATR at entry bar
            entry_atr = float(atr14[i]) if not np.isnan(atr14[i]) else (bar_high - bar_low) * 0.7
            if entry_atr <= 0:
                continue

            # Stop: 0.5 ATR from entry
            stop_dist = entry_atr * 0.5
            if direction == 1:
                stop_px = round(bar_close - stop_dist, 4)
                target_px = round(bar_close + stop_dist * 2.0, 4)
            else:
                stop_px = round(bar_close + stop_dist, 4)
                target_px = round(bar_close - stop_dist * 2.0, 4)

            # Risk-dollar sizing: shares = risk_dollars / stop_dist
            shares = int(risk_dollars / stop_dist)
            if shares < 1:
                continue

            # Max notional: 2× risk_dollars (position cap)
            notional = bar_close * shares
            if notional > risk_dollars * max_risk_mult:
                shares = int((risk_dollars * max_risk_mult) / bar_close)
                if shares < 1:
                    continue

            open_trade = {
                "direction": direction,
                "entry_px": bar_close,
                "stop_px": stop_px,
                "target_px": target_px,
                "shares": shares,
                "entry_bar": i,
                "entry_time": date,
            }

        # ── Manage open trade ───────────────────────────────────────────────
        if open_trade is not None:
            d = open_trade["direction"]
            hit_stop    = (d ==  1 and bar_low  <= open_trade["stop_px"]) or \
                          (d == -1 and bar_high >= open_trade["stop_px"])
            hit_target  = (d ==  1 and bar_high >= open_trade["target_px"]) or \
                          (d == -1 and bar_low  <= open_trade["target_px"])

            if hit_stop or hit_target:
                if hit_stop:
                    exit_px = open_trade["stop_px"]
                    reason  = "stop"
                else:
                    exit_px = open_trade["target_px"]
                    reason  = "target"

                pnl = (exit_px - open_trade["entry_px"]) * open_trade["shares"] * d
                pnl = _apply_friction(pnl, open_trade["entry_px"], open_trade["shares"], d)
                trades.append({
                    "ticker": ticker, "direction": "long" if d == 1 else "short",
                    "entry": open_trade["entry_px"], "exit": round(exit_px, 4),
                    "shares": open_trade["shares"], "pnl": round(pnl, 2),
                    "exit_reason": reason,
                    "entry_time": str(open_trade["entry_time"]), "exit_time": str(date),
                    "bars_held": i - open_trade["entry_bar"],
                    "atr_at_entry": round(entry_atr, 4),
                    "zone_top": round(zone_top, 4),
                    "zone_bot": round(zone_bot, 4),
                })
                open_trade = None

    return {"ticker": ticker, "trades": trades}


def _summarize(results: List[Dict], risk_dollars: float) -> Dict:
    """Aggregate per-ticker results into a combined summary."""
    all_trades = []
    for r in results:
        if "error" not in r:
            all_trades.extend(r.get("trades", []))

    if not all_trades:
        return {"n_trades": 0, "error": "No trades generated"}

    pnls = [t["pnl"] for t in all_trades]
    n_wins = sum(1 for p in pnls if p > 0)
    n_losses = len(pnls) - n_wins

    rets = [p / (risk_dollars) for p in pnls]   # normalize to risk units
    cum_rets = np.cumprod([1 + r for r in rets])

    ann_factor = 252 * 78   # 78 five-min bars per trading day
    ann_ret = (np.prod([1 + r for r in rets]) ** (ann_factor / len(rets)) - 1) * 100
    std_ret = np.std(rets) * np.sqrt(ann_factor)
    sharpe = (np.mean(rets) / np.std(rets) * np.sqrt(ann_factor)) if np.std(rets) > 0 else 0.0
    max_dd, _ = _max_drawdown(rets)

    n_days = len(set(t["entry_time"][:10] for t in all_trades))
    trades_per_day = len(all_trades) / max(n_days, 1)

    return {
        "n_trades":       len(all_trades),
        "n_tickers":      len([r for r in results if "error" not in r]),
        "n_days":         n_days,
        "trades_per_day": round(trades_per_day, 2),
        "win_rate":       round(n_wins / len(pnls) * 100, 1),
        "ann_return_pct": round(float(ann_ret), 1),
        "sharpe":         round(float(sharpe), 2),
        "max_dd_pct":     round(float(max_dd), 1),
        "avg_pnl":        round(float(np.mean(pnls)), 2),
        "total_pnl":      round(float(sum(pnls)), 2),
        "best_trade":     round(float(max(pnls)), 2),
        "worst_trade":    round(float(min(pnls)), 2),
        "n_wins":         n_wins,
        "n_losses":       n_losses,
    }


# ── universe + orchestration ────────────────────────────────────────────────────

UNIVERSE_20 = [
    "SPY", "QQQ", "IWM",           # ETFs
    "AAPL", "MSFT", "NVDA", "AMZN",  # Mega-cap tech
    "META", "GOOGL", "TSLA", "AMD",
    "NFLX", "CRM", "ORCL", "INTC",
    "JPM", "BAC", "GS",             # Financials
    "XOM", "CVX",                   # Energy
    "GLD", "TLT",                   # Alternatives
]


def fetch_5min_bars(ticker: str, days: int = 60) -> pd.DataFrame:
    """Fetch 5-minute bars from yfinance for the last N days."""
    import yfinance as yf
    import datetime

    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)

    df = yf.download(
        ticker, start=start, end=end,
        interval="5m",   # 5-minute bars
        prepost=True,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        return df

    # Flatten multi-level columns from yfinance (single-ticker download)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Drop zero-volume bars (pre/after-hours noise) and NaN rows
    df = df[df["Volume"] > 0].dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Keep only RTH-equivalent window (09:30–16:00 ET)
    df = df.between_time("09:30", "16:00")

    return df


def run_backtest(
    universe: List[str] = None,
    days: int = 60,
    risk_dollars: float = 100.0,
    cost_bps: float = 1.0,
    min_volume_ratio: float = 1.0,
    output_path: str = "reports/intraday_fvg_results.json",
) -> Tuple[Dict, Dict]:
    """
    Run the full intraday FVG backtest across the universe.

    Returns summary dict and writes results JSON to output_path.
    """
    universe = universe or UNIVERSE_20
    results = []
    fetched = {}

    print(f"[iFVG] Fetching 5-min data for {len(universe)} tickers ({days} days)...")
    for ticker in universe:
        df = fetch_5min_bars(ticker, days=days)
        if df.empty:
            print(f"  {ticker}: no data")
            continue
        fetched[ticker] = df
        print(f"  {ticker}: {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")

    print(f"\n[iFVG] Running backtest on {len(fetched)} tickers...")
    for ticker, df in fetched.items():
        r = _run_ticker(ticker, df, risk_dollars=risk_dollars, cost_bps=cost_bps,
                        min_volume_ratio=min_volume_ratio)
        results.append(r)
        n = len(r.get("trades", []))
        print(f"  {ticker}: {n} trades")

    summary = _summarize(results, risk_dollars)
    print(f"\n[iFVG] SUMMARY:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Write results (strip trade arrays for summary JSON)
    out = {"summary": summary, "by_ticker": {}}
    for r in results:
        ticker = r.get("ticker", "unknown")
        if "error" not in r:
            out["by_ticker"][ticker] = {
                "n_trades": len(r.get("trades", [])),
                "error": None,
            }
        else:
            out["by_ticker"][ticker] = {"n_trades": 0, "error": r["error"]}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n[iFVG] Results written to {output_path}")
    return summary, out


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--risk", type=float, default=100.0)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--vol-ratio", type=float, default=1.0)
    ap.add_argument("--output", default="reports/intraday_fvg_results.json")
    args = ap.parse_args()

    universe = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA",
                "AMD", "NFLX", "JPM", "GS", "XOM", "IWM", "GLD"]
    run_backtest(
        universe=universe,
        days=args.days,
        risk_dollars=args.risk,
        cost_bps=args.cost_bps,
        min_volume_ratio=args.vol_ratio,
        output_path=args.output,
    )
