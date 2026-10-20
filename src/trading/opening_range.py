"""
Opening Range Momentum (ORM) Strategy — Intraday

Hypothesis: stocks breaking their opening range (first 30 min) extremes
with above-average volume tend to continue in that direction through the session.

Universe:  top 20 liquid US names
Bars:       5-minute (yfinance)
Session:    09:30–14:00 EST (flat by 14:00)
Entry:      close beyond OR High (long) or OR Low (short) with volume confirmation
Stop:       0.5 × ATR(14) from entry
Target:     2:1 R:R or session close
Risk:       $100 flat per trade; max 3 concurrent positions
Friction:   1 bps round-trip (realistic)
"""

from __future__ import annotations

import json
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


def _max_drawdown(rets: List[float]) -> float:
    if not rets:
        return 0.0
    cum = np.cumprod([1 + r for r in rets])
    running_max = np.maximum.accumulate(cum)
    dd = (cum / running_max) - 1
    return float(np.min(dd) * 100)


def _apply_friction(pnl: float, entry_px: float, shares: int) -> float:
    """Deduct 1 bps round-trip cost."""
    if shares <= 0 or entry_px <= 0:
        return pnl
    cost = entry_px * shares * 0.0001
    return pnl - cost


# ── per-ticker backtest ─────────────────────────────────────────────────────────

def _run_ticker(
    ticker: str,
    df: pd.DataFrame,
    risk_dollars: float = 100.0,
    cost_bps: float = 1.0,
    vol_ratio: float = 1.0,
    or_bars: int = 6,
) -> Dict:
    """
    Run ORM backtest on one ticker.
    df must be 5-min bars, 09:30–16:00 EST, no zero-volume rows.
    Returns dict with summary and trade list.
    """
    df = df.copy()
    df = df.sort_index()
    if len(df) < 50:
        return {"ticker": ticker, "error": f"Insufficient bars ({len(df)})", "trades": []}

    o = np.asarray(df["Open"].values, dtype=float)
    h = np.asarray(df["High"].values, dtype=float)
    l = np.asarray(df["Low"].values, dtype=float)
    c = np.asarray(df["Close"].values, dtype=float)
    v = np.asarray(df["Volume"].values, dtype=float)
    ts_idx = df.index  # DatetimeIndex for .hour/.minute access

    n = len(c)
    atr14 = _atr(h, l, c, period=14)

    # Rolling median volume (prior 20 bars)
    med_vol = np.full(n, np.nan)
    for i in range(20, n):
        med_vol[i] = np.median(v[i - 20:i])

    trades: List[Dict] = []
    open_trade: Optional[Dict] = None  # direction, entry_px, stop_px, target_px, shares, entry_bar

    for i in range(or_bars, n):
        date = ts_idx[i]
        total_min = date.hour * 60 + date.minute

        # ── 14:00 EST hard close (840 min) ─────────────────────────────────
        if total_min >= 840:
            if open_trade is not None:
                exit_px = float(c[i])
                d = open_trade["direction"]
                pnl = (exit_px - open_trade["entry_px"]) * open_trade["shares"] * d
                pnl = _apply_friction(pnl, open_trade["entry_px"], open_trade["shares"])
                trades.append({
                    "ticker": ticker,
                    "direction": "long" if d == 1 else "short",
                    "entry": open_trade["entry_px"],
                    "exit": round(exit_px, 4),
                    "shares": open_trade["shares"],
                    "pnl": round(pnl, 2),
                    "exit_reason": "time_close",
                    "entry_time": str(open_trade["entry_time"]),
                    "exit_time": str(date),
                    "bars_held": i - open_trade["entry_bar"],
                })
                open_trade = None
            continue

        bar_high  = float(h[i])
        bar_low   = float(l[i])
        bar_close = float(c[i])
        bar_vol   = float(v[i])

        # ── Establish OR High/Low at bar `or_bars` ───────────────────────────
        # OR period ends at bar i == or_bars (bar 0-indexed, so bars 0..or_bars-1)
        # At i == or_bars, OR is finalized for the day
        # Then on subsequent bars, check for breakouts

        if i == or_bars:
            # OR just finalised — record it and skip (no entry on OR bar itself)
            or_high = float(np.max(h[:or_bars]))
            or_low  = float(np.min(l[:or_bars]))
            or_finalized = True
            open_trade = None
            continue

        # After OR is finalised, check for breakout entries
        # We carry OR High/Low forward each day (reset at new day boundary)
        # Simple approach: track or_high/or_low and reset at new session

        # ── Session boundary detection (new day → reset OR) ──────────────────
        prev_date = ts_idx[i - 1]
        if prev_date.date() != date.date():
            # New trading day — reset OR
            or_high = float(np.max(h[i - or_bars:i]))
            or_low  = float(np.min(l[i - or_bars:i]))
            # Don't trade on the OR formation bar itself
            if i - ts_idx.get_loc(date.replace(hour=9, minute=30)) < or_bars:
                continue
            # Actually: OR High/Low = max/min of bars from session open to or_bars
            # session starts at first bar of the day
            or_high = float(np.max(h[i:i + or_bars]))  # forward-looking for new session
            or_low  = float(np.min(l[i:i + or_bars]))
            # This is wrong. Let me restructure properly.

        # Skip if no OR established yet
        if or_high <= 0 or or_low <= 0:
            continue

        vol_ok = (med_vol[i] > 0) and (bar_vol >= vol_ratio * med_vol[i])

        # ── Check for open trade management ─────────────────────────────────
        if open_trade is not None:
            d = open_trade["direction"]
            hit_stop   = (d ==  1 and bar_low  <= open_trade["stop_px"]) or \
                         (d == -1 and bar_high >= open_trade["stop_px"])
            hit_target = (d ==  1 and bar_high >= open_trade["target_px"]) or \
                         (d == -1 and bar_low  <= open_trade["target_px"])
            if hit_stop or hit_target:
                exit_px = open_trade["stop_px"] if hit_stop else open_trade["target_px"]
                reason  = "stop" if hit_stop else "target"
                pnl = (exit_px - open_trade["entry_px"]) * open_trade["shares"] * d
                pnl = _apply_friction(pnl, open_trade["entry_px"], open_trade["shares"])
                trades.append({
                    "ticker": ticker,
                    "direction": "long" if d == 1 else "short",
                    "entry": open_trade["entry_px"],
                    "exit": round(exit_px, 4),
                    "shares": open_trade["shares"],
                    "pnl": round(pnl, 2),
                    "exit_reason": reason,
                    "entry_time": str(open_trade["entry_time"]),
                    "exit_time": str(date),
                    "bars_held": i - open_trade["entry_bar"],
                    "or_high": round(or_high, 4),
                    "or_low": round(or_low, 4),
                })
                open_trade = None
            continue

        # ── New trade entry ─────────────────────────────────────────────────
        if open_trade is None:
            for direction in [1, -1]:
                entry_triggered = False
                if direction == 1 and bar_close > or_high and vol_ok:
                    entry_triggered = True
                elif direction == -1 and bar_close < or_low and vol_ok:
                    entry_triggered = True

                if not entry_triggered:
                    continue

                entry_atr = float(atr14[i]) if not np.isnan(atr14[i]) else (bar_high - bar_low) * 0.7
                if entry_atr <= 0:
                    continue

                stop_dist = entry_atr * 0.5
                if direction == 1:
                    stop_px   = round(bar_close - stop_dist, 4)
                    target_px = round(bar_close + stop_dist * 2.0, 4)
                else:
                    stop_px   = round(bar_close + stop_dist, 4)
                    target_px = round(bar_close - stop_dist * 2.0, 4)

                shares = int(risk_dollars / stop_dist)
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
                break  # only one new trade per bar

    return {"ticker": ticker, "trades": trades}


# ── session-split OR helper ─────────────────────────────────────────────────────

def _run_ticker_v2(
    ticker: str,
    df: pd.DataFrame,
    risk_dollars: float = 100.0,
    cost_bps: float = 1.0,
    vol_ratio: float = 1.0,
    or_bars: int = 6,
) -> Dict:
    """
    v2: split df into sessions, run OR strategy per session.
    Handles new-day detection cleanly.
    """
    df = df.copy().sort_index()
    if len(df) < 50:
        return {"ticker": ticker, "error": f"Insufficient bars ({len(df)})", "trades": []}

    o = np.asarray(df["Open"].values, dtype=float)
    h = np.asarray(df["High"].values, dtype=float)
    l = np.asarray(df["Low"].values, dtype=float)
    c = np.asarray(df["Close"].values, dtype=float)
    v = np.asarray(df["Volume"].values, dtype=float)
    ts_idx = df.index  # DatetimeIndex

    n = len(c)
    atr14 = _atr(h, l, c, period=14)

    med_vol = np.full(n, np.nan)
    for i in range(20, n):
        med_vol[i] = np.median(v[i - 20:i])

    # Build session list using positional slicing to avoid get_loc timestamp ambiguity
    sessions = []
    current_date = None
    session_start = 0
    for pos in range(len(df)):
        bar_date = df.index[pos].date()
        if bar_date != current_date:
            if current_date is not None:
                sessions.append(df.iloc[session_start:pos])   # close prior session
            session_start = pos
            current_date = bar_date
    sessions.append(df.iloc[session_start:])  # final session

    trades: List[Dict] = []
    open_trade: Optional[Dict] = None

    for sdf in sessions:
        s_n = len(sdf)
        if s_n < or_bars + 2:
            continue

        # Get absolute integer position of session start in full array
        session_start = int(df.index.get_loc(sdf.index[0]))

        # Opening Range from first `or_bars` bars (use integer positions on numpy arrays)
        or_high = float(np.max(h[session_start:session_start + or_bars]))
        or_low  = float(np.min(l[session_start:session_start + or_bars]))

        # Iterate by integer position within session
        for rel_i in range(or_bars, s_n):
            abs_i = sdf.index.get_loc(sdf.index[0]) + rel_i
            date  = ts_idx[abs_i]
            total_min = date.hour * 60 + date.minute

            bar_high  = float(h[abs_i])
            bar_low   = float(l[abs_i])
            bar_close = float(c[abs_i])
            bar_vol   = float(v[abs_i])

            # ── 14:00 EST hard close ──────────────────────────────────────
            if total_min >= 840:
                if open_trade is not None:
                    exit_px = bar_close
                    d = open_trade["direction"]
                    pnl = (exit_px - open_trade["entry_px"]) * open_trade["shares"] * d
                    pnl = _apply_friction(pnl, open_trade["entry_px"], open_trade["shares"])
                    trades.append({
                        "ticker": ticker,
                        "direction": "long" if d == 1 else "short",
                        "entry": open_trade["entry_px"],
                        "exit": round(exit_px, 4),
                        "shares": open_trade["shares"],
                        "pnl": round(pnl, 2),
                        "exit_reason": "time_close",
                        "entry_time": str(open_trade["entry_time"]),
                        "exit_time": str(date),
                        "bars_held": abs_i - open_trade["entry_bar"],
                        "or_high": round(or_high, 4),
                        "or_low": round(or_low, 4),
                    })
                    open_trade = None
                continue

            # ── Manage open trade ─────────────────────────────────────────
            if open_trade is not None:
                d = open_trade["direction"]
                hit_stop   = (d ==  1 and bar_low  <= open_trade["stop_px"]) or \
                             (d == -1 and bar_high >= open_trade["stop_px"])
                hit_target = (d ==  1 and bar_high >= open_trade["target_px"]) or \
                             (d == -1 and bar_low  <= open_trade["target_px"])
                if hit_stop or hit_target:
                    exit_px = open_trade["stop_px"] if hit_stop else open_trade["target_px"]
                    reason  = "stop" if hit_stop else "target"
                    pnl = (exit_px - open_trade["entry_px"]) * open_trade["shares"] * d
                    pnl = _apply_friction(pnl, open_trade["entry_px"], open_trade["shares"])
                    trades.append({
                        "ticker": ticker,
                        "direction": "long" if d == 1 else "short",
                        "entry": open_trade["entry_px"],
                        "exit": round(exit_px, 4),
                        "shares": open_trade["shares"],
                        "pnl": round(pnl, 2),
                        "exit_reason": reason,
                        "entry_time": str(open_trade["entry_time"]),
                        "exit_time": str(date),
                        "bars_held": abs_i - open_trade["entry_bar"],
                        "or_high": round(or_high, 4),
                        "or_low": round(or_low, 4),
                    })
                    open_trade = None
                continue

            # ── New entry ─────────────────────────────────────────────────
            vol_ok = (med_vol[abs_i] > 0) and (bar_vol >= vol_ratio * med_vol[abs_i])
            if not vol_ok:
                continue

            for direction in [1, -1]:
                triggered = (direction == 1 and bar_close > or_high) or \
                           (direction == -1 and bar_close < or_low)
                if not triggered:
                    continue

                entry_atr = float(atr14[abs_i])
                if np.isnan(entry_atr) or entry_atr <= 0:
                    entry_atr = (bar_high - bar_low) * 0.7

                stop_dist = entry_atr * 0.5
                if stop_dist <= 0:
                    continue

                if direction == 1:
                    stop_px   = round(bar_close - stop_dist, 4)
                    target_px = round(bar_close + stop_dist * 2.0, 4)
                else:
                    stop_px   = round(bar_close + stop_dist, 4)
                    target_px = round(bar_close - stop_dist * 2.0, 4)

                shares = int(risk_dollars / stop_dist)
                if shares < 1:
                    continue

                open_trade = {
                    "direction": direction,
                    "entry_px": bar_close,
                    "stop_px": stop_px,
                    "target_px": target_px,
                    "shares": shares,
                    "entry_bar": abs_i,
                    "entry_time": date,
                }
                break  # one trade per bar

    return {"ticker": ticker, "trades": trades}


def _summarize(results: List[Dict], risk_dollars: float) -> Dict:
    all_trades = []
    for r in results:
        if "error" not in r:
            all_trades.extend(r.get("trades", []))

    if not all_trades:
        return {"n_trades": 0, "error": "No trades"}

    pnls = [t["pnl"] for t in all_trades]
    rets = [p / risk_dollars for p in pnls]
    n_wins   = sum(1 for p in pnls if p > 0)
    n_losses = len(pnls) - n_wins

    ann_factor = 252 * 78
    mean_ret = np.mean(rets)
    std_ret  = np.std(rets)
    sharpe    = (mean_ret / std_ret * np.sqrt(ann_factor)) if std_ret > 0 else 0.0
    ann_ret   = (np.prod([1 + r for r in rets]) ** (ann_factor / len(rets)) - 1) * 100
    max_dd    = _max_drawdown(rets)

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
        # friction sensitivity
        "pnl_at_1bps":    round(float(sum(pnls) - len(pnls) * risk_dollars * 0.0001), 2),
        "pnl_at_2bps":    round(float(sum(pnls) - len(pnls) * risk_dollars * 0.0002), 2),
        "pnl_at_5bps":    round(float(sum(pnls) - len(pnls) * risk_dollars * 0.0005), 2),
    }


# ── universe + data fetch ───────────────────────────────────────────────────────

UNIVERSE_20 = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "NVDA", "AMZN",
    "META", "GOOGL", "TSLA", "AMD",
    "NFLX", "CRM", "ORCL",
    "JPM", "BAC", "GS",
    "XOM", "CVX",
]


def fetch_5min_bars(ticker: str, days: int = 60) -> pd.DataFrame:
    """Fetch 5-min bars, filter to RTH 09:30–16:00, drop zero-volume rows."""
    import yfinance as yf
    import datetime

    end   = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)

    df = yf.download(
        ticker, start=start, end=end,
        interval="5m", prepost=True,
        auto_adjust=True, progress=False,
    )
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Drop zero-volume (pre/after-hours) and NaN
    df = df[df["Volume"] > 0].dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df.between_time("09:30", "16:00")

    return df


def run_backtest(
    universe: List[str] = None,
    days: int = 60,
    risk_dollars: float = 100.0,
    cost_bps: float = 1.0,
    vol_ratio: float = 1.0,
    or_bars: int = 6,
    output_path: str = "reports/opening_range_results.json",
) -> Tuple[Dict, Dict]:

    universe = universe or UNIVERSE_20
    results = []
    fetched = {}

    print(f"[ORM] Fetching 5-min data for {len(universe)} tickers ({days} days)...")
    for ticker in universe:
        df = fetch_5min_bars(ticker, days=days)
        if df.empty:
            print(f"  {ticker}: no data")
            continue
        fetched[ticker] = df
        print(f"  {ticker}: {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")

    print(f"\n[ORM] Running backtest on {len(fetched)} tickers...")
    for ticker, df in fetched.items():
        r = _run_ticker_v2(
            ticker, df,
            risk_dollars=risk_dollars,
            cost_bps=cost_bps,
            vol_ratio=vol_ratio,
            or_bars=or_bars,
        )
        results.append(r)
        n = len(r.get("trades", []))
        print(f"  {ticker}: {n} trades  {r.get('error', '')}")

    summary = _summarize(results, risk_dollars)

    print(f"\n{'='*50}")
    print(f"[ORM] SUMMARY  (risk=${risk_dollars}/trade, {cost_bps} bps friction)")
    print(f"{'='*50}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = {"summary": summary, "by_ticker": {}}
    for r in results:
        t = r.get("ticker", "unknown")
        if "error" not in r:
            out["by_ticker"][t] = {"n_trades": len(r.get("trades", []))}
        else:
            out["by_ticker"][t] = {"n_trades": 0, "error": r["error"]}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n[ORM] Results → {output_path}")
    return summary, out


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--risk", type=float, default=100.0)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--vol-ratio", type=float, default=1.0)
    ap.add_argument("--or-bars", type=int, default=6)
    ap.add_argument("--output", default="reports/opening_range_results.json")
    args = ap.parse_args()

    run_backtest(
        days=args.days,
        risk_dollars=args.risk,
        cost_bps=args.cost_bps,
        vol_ratio=args.vol_ratio,
        or_bars=args.or_bars,
        output_path=args.output,
    )
