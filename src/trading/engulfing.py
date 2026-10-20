"""
Engulfing Bar Strategy — Backtest Module

Pattern:   Daily OHLCV candle N-1 vs N
             Bullish engulfing:  O[N-1] > C[N-1], C[N] > O[N-1], C[N] > H[N-1]
             Bearish engulfing:  O[N-1] < C[N-1], C[N] < O[N-1], C[N] < L[N-1]

Signal:    Long  on bullish engulfing  (after N-day downtrend)
           Short on bearish engulfing  (after N-day uptrend)

Exits:     intraday  → sell at close of day N+1 (same session)
           overnight  → sell at open of day N+2  (carry position overnight)

Variants tested:
  - Bullish-only, intraday
  - Bullish-only, overnight
  - Bearish-only, intraday
  - Bearish-only, overnight
  - Both combined,   intraday
  - Both combined,   overnight
  - With trend filter (require prior N days in opposite direction)
  - Without trend filter

Entry:     Open of day N+1
Stop:      H[N-1] - 1*ATR(14)  (long)  /  L[N-1] + 1*ATR(14)  (short)
Target:    2× risk  (R:R = 1:2)  — evaluated at each bar until hit
"""

from __future__ import annotations

import json
import math
import pickle
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────────────

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder-smoothed ATR.  Returns array of same length as close."""
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


def _engulfing_flags(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (bullish_flag, bearish_flag)  — boolean arrays, same length.
    Engulfing candle is the CURRENT bar (index i); prior bar is i-1.
    Index 0 is always False (no prior bar for comparison).
    """
    n = len(close)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)

    for i in range(1, n):
        p_o, p_c = open_[i - 1], close[i - 1]   # prior candle
        c_o, c_c = open_[i],     close[i]       # current candle
        c_h, c_l = high[i],      low[i]

        # Bullish: prior close < open (red/empty), current body engulfs prior body
        if p_c < p_o and c_c > p_o and c_c > p_c:
            bull[i] = True

        # Bearish: prior close > open (green/filled), current body engulfs prior body
        if p_c > p_o and c_c < p_o and c_c < p_c:
            bear[i] = True

    return bull, bear


def _trend_filter(
    close: np.ndarray,
    lookback: int,
    direction: str,          # "bull" or "bear"
) -> np.ndarray:
    """
    Return boolean array — True where the prior `lookback` days were
    consistently down (for bull) or up (for bear).
    Index 0..lookback-1 are always False.
    """
    n = len(close)
    out = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        window = close[i - lookback:i]
        if direction == "bull":
            out[i] = window[-1] < window[0]          # price lower than lookback days ago
        else:
            out[i] = window[-1] > window[0]          # price higher than lookback days ago
    return out


def _run_single(
    price_df: Dict[str, pd.DataFrame],
    ticker: str,
    direction: str,          # "bull", "bear", or "both"
    exit_mode: str,           # "intraday" or "overnight"
    trend_lookback: int,     # 0 = no trend filter
    min_body_ratio: float,    # 0 = no minimum
    start_date: str,
    end_date: str,
    cost_bps: float = 0.0,    # round-trip commission + slippage in bps
) -> Dict:
    """
    Run a single strategy variant on one ticker.
    Returns a dict of per-trade results and summary metrics.
    """
    df = price_df[ticker].dropna().copy()
    df = df.sort_index()
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    if len(df) < 30:
        return {"error": f"Insufficient data ({len(df)} rows)", "trades": []}

    o = np.asarray(df["Open"].values, dtype=float)
    h = np.asarray(df["High"].values, dtype=float)
    l = np.asarray(df["Low"].values, dtype=float)
    c = np.asarray(df["Close"].values, dtype=float)
    idx = np.asarray(df.index.values)

    atr14 = _atr(h, l, c, period=14)

    bull_flag, bear_flag = _engulfing_flags(o, h, l, c)

    if trend_lookback > 0:
        bull_trend = _trend_filter(c, trend_lookback, "bull")
        bear_trend = _trend_filter(c, trend_lookback, "bear")
    else:
        bull_trend = np.ones(len(c), dtype=bool)
        bear_trend = np.ones(len(c), dtype=bool)

    trades: List[Dict] = []

    for i in range(1, len(c) - 2):
        date = idx[i]

        # ── signal detection (engulfing is on bar i) ────────────────────────
        if direction in ("bull", "both"):
            long_signal = bull_flag[i] and bull_trend[i]
        else:
            long_signal = False

        if direction in ("bear", "both"):
            short_signal = bear_flag[i] and bear_trend[i]
        else:
            short_signal = False

        if not (long_signal or short_signal):
            continue

        is_long = long_signal

        # ── entry: open of next bar (i+1) ───────────────────────────────────
        entry_price = o[i + 1]
        if entry_price <= 0:
            continue

        # ── stop: prior bar's extreme ± 1 ATR ───────────────────────────────
        if is_long:
            stop_price = h[i] - 1.0 * atr14[i + 1]
        else:
            stop_price = l[i] + 1.0 * atr14[i + 1]

        # ── target: 2× risk ─────────────────────────────────────────────────
        risk = abs(entry_price - stop_price)
        target_price = entry_price + 2 * risk if is_long else entry_price - 2 * risk

        # ── walk forward to find exit ───────────────────────────────────────
        exited = False
        exit_price: float = math.nan
        exit_bar = -1
        exit_type = "unknown"

        for j in range(i + 2, len(c)):
            if exit_mode == "intraday":
                # Exit at close of same session (bar j = bar i+1)
                exit_price = float(c[j])
                exit_bar = j
                exit_type = "intraday"
                exited = True
                break
            else:
                # Overnight: walk bar by bar — stop or target can fire first
                bar_high = float(h[j])
                bar_low  = float(l[j])

                if is_long:
                    if bar_low <= stop_price:
                        exit_price  = stop_price
                        exit_bar    = j
                        exit_type   = "stop"
                        exited      = True
                        break
                    if bar_high >= target_price:
                        exit_price  = target_price
                        exit_bar    = j
                        exit_type   = "target"
                        exited      = True
                        break
                else:
                    if bar_high >= stop_price:
                        exit_price  = stop_price
                        exit_bar    = j
                        exit_type   = "stop"
                        exited      = True
                        break
                    if bar_low <= target_price:
                        exit_price  = target_price
                        exit_bar    = j
                        exit_type   = "target"
                        exited      = True
                        break

        if not exited or exit_bar < 0:
            # No valid exit found within data — skip
            continue

        pnl_r = (exit_price - entry_price) / entry_price if is_long \
                else (entry_price - exit_price) / entry_price
        pnl_r -= cost_bps / 10_000          # deduct round-trip cost
        pnl_bps = pnl_r * 10_000
        ret = pnl_r * 100

        trades.append({
            "ticker":       ticker,
            "entry_date":   str(date)[:10],
            "exit_date":    str(idx[exit_bar])[:10],
            "direction":    "long" if is_long else "short",
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "stop_price":   stop_price,
            "target_price": target_price,
            "pnl_r":        pnl_r,
            "pnl_bps":      pnl_bps,
            "ret_pct":      ret,
            "exit_type":    exit_type,
            "atr_entry":    atr14[i + 1],
        })

    # ── summary metrics ────────────────────────────────────────────────────
    if not trades:
        return {"ticker": ticker, "direction": direction, "exit_mode": exit_mode,
                "trend_lookback": trend_lookback, "trades": [],
                "n": 0, "error": None}

    rets   = [t["pnl_r"] for t in trades]
    bps    = [t["pnl_bps"] for t in trades]
    wins   = [t for t in trades if t["pnl_r"] > 0]
    stops  = [t for t in trades if t["exit_type"] == "stop"]
    targets= [t for t in trades if t["exit_type"] == "target"]
    intra  = [t for t in trades if t["exit_type"] == "intraday"]

    total_ret   = np.sum(rets)
    ann_ret     = (1 + total_ret) ** (252 / max(len(rets), 1)) - 1 if rets else 0.0
    ann_vol     = np.std(rets) * math.sqrt(252) if len(rets) > 1 else 0.0
    sharpe      = (ann_ret / ann_vol) if ann_vol > 0 else 0.0
    max_dd      = _max_drawdown(rets)
    win_rate    = len(wins) / len(trades) * 100 if trades else 0.0
    avg_bps     = np.mean(bps)
    avg_win_bps = np.mean([t["pnl_bps"] for t in wins]) if wins else 0.0
    avg_loss_bps= np.mean([t["pnl_bps"] for t in trades if t["pnl_r"] < 0]) if any(t["pnl_r"] < 0 for t in trades) else 0.0
    stop_rate   = len(stops)  / len(trades) * 100 if trades else 0.0
    target_rate = len(targets) / len(trades) * 100 if trades else 0.0
    intra_rate  = len(intra)  / len(trades) * 100 if trades else 0.0

    return {
        "ticker":        ticker,
        "direction":     direction,
        "exit_mode":     exit_mode,
        "trend_lookback":trend_lookback,
        "trades":        trades,
        "n":             len(trades),
        "error":         None,
        # summary
        "total_ret":     total_ret,
        "ann_ret":       ann_ret,
        "ann_vol":       ann_vol,
        "sharpe":        sharpe,
        "max_dd":        max_dd,
        "win_rate":      win_rate,
        "avg_bps":       avg_bps,
        "avg_win_bps":   avg_win_bps,
        "avg_loss_bps":  avg_loss_bps,
        "stop_rate":     stop_rate,
        "target_rate":   target_rate,
        "intraday_rate": intra_rate,
    }


def _max_drawdown(rets: List[float]) -> float:
    """Worst peak-to-trough drawdown from a list of period returns."""
    if not rets:
        return 0.0
    cum = np.cumprod([1 + r for r in rets])
    running_max = np.maximum.accumulate(cum)
    dd = (cum / running_max) - 1
    return float(np.min(dd) * 100)


def run_all_variants(
    price_df: Dict[str, pd.DataFrame],
    tickers: List[str],
    start_date: str = "2018-01-01",
    end_date:   str = "2025-12-31",
    min_body_ratio: float = 0.0,
    cost_bps: float = 0.0,
) -> Dict:
    """
    Run every combination of:
      direction  = bull / bear / both
      exit_mode  = intraday / overnight
      trend_lookback = 0 / 5 / 10
    Returns dict keyed by (direction, exit_mode, trend_lookback).
    """
    directions   = ["bull", "bear", "both"]
    exit_modes   = ["intraday", "overnight"]
    trend_looks  = [0, 5, 10]

    results: Dict[str, Dict] = {}
    all_trades: List[Dict]   = []

    for direction in directions:
        for exit_mode in exit_modes:
            for tlook in trend_looks:
                key = f"{direction}_{exit_mode}_tl{tlook}"
                _combined: Dict = {
                    "n": 0,
                    "win_rate": 0.0, "avg_bps": 0.0,
                    "avg_win_bps": 0.0, "avg_loss_bps": 0.0,
                    "stop_rate": 0.0, "target_rate": 0.0, "intraday_rate": 0.0,
                    "direction": direction, "exit_mode": exit_mode,
                    "trend_lookback": tlook,
                }
                trades_pool: List[Dict] = []

                for ticker in tickers:
                    r = _run_single(
                        price_df, ticker, direction, exit_mode,
                        tlook, min_body_ratio, start_date, end_date,
                        cost_bps=cost_bps,
                    )
                    n_r = r.get("n", 0)
                    if n_r > 0:
                        trades_pool.extend(r["trades"])
                        _combined["n"]          += n_r
                        _combined["avg_bps"]   += r.get("avg_bps",   0.0) * n_r
                        _combined["win_rate"]  += r.get("win_rate",  0.0) * n_r
                        _combined["stop_rate"]  += r.get("stop_rate", 0.0) * n_r
                        _combined["target_rate"]+= r.get("target_rate", 0.0) * n_r
                        _combined["intraday_rate"]+= r.get("intraday_rate", 0.0) * n_r
                        _combined["avg_win_bps"]   += r.get("avg_win_bps",   0.0) * n_r
                        _combined["avg_loss_bps"]  += r.get("avg_loss_bps",  0.0) * n_r

                if _combined["n"] > 0:
                    n_all = _combined["n"]

                    # ── true pooled-trade metrics ──────────────────────────────
                    rets      = [t["pnl_r"]  for t in trades_pool]
                    bps_list  = [t["pnl_bps"] for t in trades_pool]
                    wins_list = [t for t in trades_pool if t["pnl_r"] > 0]
                    loss_list = [t for t in trades_pool if t["pnl_r"] < 0]

                    cum   = np.cumprod([1 + r for r in rets])
                    total_return = float(cum[-1]) - 1
                    ann_vol     = float(np.std(rets) * math.sqrt(252))
                    ann_ret_agg = (1 + total_return) ** (252 / max(len(rets), 1)) - 1
                    _combined["total_ret"] = total_return
                    _combined["ann_ret"]    = ann_ret_agg
                    _combined["ann_vol"]    = ann_vol
                    _combined["sharpe"]     = (ann_ret_agg / ann_vol) if ann_vol > 0 else 0.0
                    _combined["max_dd"]     = _max_drawdown(rets)

                    # ── pooled-rate metrics ─────────────────────────────────
                    _combined["avg_bps"]       = float(np.mean(bps_list))
                    _combined["win_rate"]       = len(wins_list) / n_all * 100
                    _combined["avg_win_bps"]    = float(np.mean([t["pnl_bps"] for t in wins_list])) if wins_list else 0.0
                    _combined["avg_loss_bps"]  = float(np.mean([t["pnl_bps"] for t in loss_list])) if loss_list else 0.0
                    _combined["stop_rate"]      = len([t for t in trades_pool if t["exit_type"] == "stop"]) / n_all * 100
                    _combined["target_rate"]    = len([t for t in trades_pool if t["exit_type"] == "target"]) / n_all * 100
                    _combined["intraday_rate"]  = len([t for t in trades_pool if t["exit_type"] == "intraday"]) / n_all * 100

                    _combined["all_trades"] = trades_pool

                results[key] = _combined
                all_trades.extend(trades_pool)

    results["_meta"] = {"tickers": tickers, "start": start_date, "end": end_date}
    return results


def print_results(results: Dict) -> None:
    meta = results.get("_meta", {})
    print(f"\n{'═'*72}")
    print(f"  ENGULFING BAR BACKTEST  |  {meta.get('start','?')} – {meta.get('end','?')}")
    print(f"  Universe: {len(meta.get('tickers',[]))} tickers")
    print(f"{'═'*72}")

    variants = [(k, v) for k, v in results.items() if k != "_meta"]
    # Sort by sharpe descending
    variants.sort(key=lambda x: x[1].get("sharpe", -999), reverse=True)

    header = (
        f"{'Variant':<38} {'N':>5} {'Ret%':>7} {'Ann%':>7} "
        f"{'Sharpe':>7} {'MaxDD%':>7} {'WR%':>5} "
        f"{'AvgBps':>7} {'Stop%':>6} {'Tgt%':>6}"
    )
    print(header)
    print("─" * 90)

    for name, r in variants:
        n = r.get("n", 0)
        if n == 0:
            continue
        ret   = r.get("total_ret", 0)   * 100
        ann   = r.get("ann_ret",   0.0)  * 100
        shp   = r.get("sharpe",    0.0)
        dd    = r.get("max_dd",     0.0)
        wr    = r.get("win_rate",  0.0)
        bps   = r.get("avg_bps",   0.0)
        stp   = r.get("stop_rate", 0.0)
        tgt   = r.get("target_rate", 0.0)

        print(
            f"{name:<38} {n:>5} {ret:>7.1f} {ann:>7.1f} "
            f"{shp:>7.2f} {dd:>7.1f} {wr:>5.1f} "
            f"{bps:>7.1f} {stp:>6.1f} {tgt:>6.1f}"
        )

    print("─" * 90)

    # ── exit-type breakdown across all combined (both, both modes) ───────────
    combined = results.get("both_overnight_tl0") or results.get("both_intraday_tl0") or {}
    if combined and combined.get("n", 0) > 0:
        trades = combined.get("all_trades", [])
        if trades:
            by_type: Dict[str, List] = {}
            for t in trades:
                by_type.setdefault(t["exit_type"], []).append(t["pnl_bps"])
            print("\nExit-type breakdown (both, both modes):")
            print(f"  {'Type':<12} {'N':>5} {'Share%':>7} {'AvgBps':>8}")
            total = len(trades)
            for et in ["intraday", "stop", "target", "overnight_roll"]:
                if et in by_type:
                    sub = by_type[et]
                    print(
                        f"  {et:<12} {len(sub):>5} "
                        f"{len(sub)/total*100:>7.1f} "
                        f"{np.mean(sub):>8.1f}"
                    )


def save_results(results: Dict, path: str) -> None:
    out: Dict = {}
    for k, v in results.items():
        if k == "_meta":
            out[k] = v
        elif isinstance(v, dict) and "all_trades" in v:
            trades_only = [{kk: vv for kk, vv in t.items()
                           if kk not in ("all_trades",)}
                          for t in v.get("all_trades", [])]
            summary = {kk: vv for kk, vv in v.items() if kk != "all_trades"}
            out[k] = {"summary": summary, "trades": trades_only}
        else:
            out[k] = v
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)


# ── Walk-Forward Validation ──────────────────────────────────────────────────

def walk_forward_validate(
    price_df: Dict[str, pd.DataFrame],
    tickers: List[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    cost_bps: float = 0.0,
    top_n: int = 5,
) -> Dict:
    """
    Walk-forward validation with in-sample training and out-of-sample testing.

    1. Train:  find the best trend_lookback for each direction/exit variant
       using in-sample Sharpe.
    2. Test:   apply the trained config out-of-sample and record metrics.

    Returns a dict of results per variant (direction/exit), each with
    train_metrics, test_metrics, and selected_tlook.
    """
    directions  = ["bull", "bear", "both"]
    exit_modes  = ["intraday", "overnight"]

    # ── Step 1: In-sample training ────────────────────────────────────────────
    train_results = run_all_variants(
        price_df, tickers,
        start_date=train_start, end_date=train_end,
        cost_bps=cost_bps,
    )

    # Select best tl for each variant by Sharpe on train
    best_tl: Dict[str, int] = {}
    for direction in directions:
        for exit_mode in exit_modes:
            key_prefix = f"{direction}_{exit_mode}_tl"
            candidates = {
                k: v for k, v in train_results.items()
                if k.startswith(key_prefix) and v.get("n", 0) > 0
            }
            if not candidates:
                continue
            best = max(candidates.keys(), key=lambda k: candidates[k].get("sharpe", -999))
            best_tl[f"{direction}_{exit_mode}"] = int(best.split("_tl")[1])

    # ── Step 2: Out-of-sample test ────────────────────────────────────────────
    test_results = run_all_variants(
        price_df, tickers,
        start_date=test_start, end_date=test_end,
        cost_bps=cost_bps,
    )

    summary: Dict = {}
    for direction in directions:
        for exit_mode in exit_modes:
            combo_key = f"{direction}_{exit_mode}"
            if combo_key not in best_tl:
                continue
            tlook = best_tl[combo_key]
            variant_key = f"{combo_key}_tl{tlook}"

            train_r = train_results.get(variant_key, {})
            test_r  = test_results.get(variant_key,  {})

            def _summarize(r: Dict) -> Dict:
                return {
                    "n":           r.get("n", 0),
                    "ann_ret":     r.get("ann_ret", 0.0) * 100,
                    "sharpe":      r.get("sharpe", 0.0),
                    "max_dd":      r.get("max_dd", 0.0),
                    "win_rate":    r.get("win_rate", 0.0),
                    "avg_bps":     r.get("avg_bps", 0.0),
                    "stop_rate":   r.get("stop_rate", 0.0),
                    "target_rate": r.get("target_rate", 0.0),
                }

            summary[combo_key] = {
                "train": _summarize(train_r),
                "test":  _summarize(test_r),
                "selected_tl":  tlook,
                "variant_key":  variant_key,
            }

    return {
        "summary": summary,
        "best_tl": best_tl,
        "train_period": {"start": train_start, "end": train_end},
        "test_period":  {"start": test_start,  "end": test_end},
        "cost_bps": cost_bps,
        "n_tickers": len(tickers),
    }


def print_wfo_results(wfo: Dict) -> None:
    """Pretty-print walk-forward validation results."""
    meta = wfo
    print(f"\n{'═'*72}")
    print(f"  WALK-FORWARD VALIDATION  |  train: {meta['train_period']['start']} – {meta['train_period']['end']}")
    print(f"                            |  test:  {meta['test_period']['start']} – {meta['test_period']['end']}")
    print(f"                            |  cost:  {meta['cost_bps']} bps round-trip")
    print(f"  Universe: {meta['n_tickers']} tickers")
    print(f"{'═'*72}")

    header = (f"{'Variant':<28} {'TLook':>6} "
              f"{'Train Sharpe':>12} {'Test Sharpe':>12} "
              f"{'Train Ann%':>10} {'Test Ann%':>10} "
              f"{'Test MaxDD%':>11} {'Test WR%':>8}")
    print(header)
    print("─" * 100)

    summary = wfo.get("summary", {})
    for key, v in sorted(summary.items()):
        tr, te = v["train"], v["test"]
        if tr["n"] == 0 and te["n"] == 0:
            continue
        print(
            f"{key:<28} {v['selected_tl']:>6} "
            f"{tr['sharpe']:>12.2f} {te['sharpe']:>12.2f} "
            f"{tr['ann_ret']:>10.1f} {te['ann_ret']:>10.1f} "
            f"{te['max_dd']:>11.1f} {te['win_rate']:>8.1f}"
        )
    print("─" * 100)
    print(f"\nBest trend-lookback selection (in-sample):")
    for k, v in sorted(wfo.get("best_tl", {}).items()):
        print(f"  {k:<28}  tl={v}")
