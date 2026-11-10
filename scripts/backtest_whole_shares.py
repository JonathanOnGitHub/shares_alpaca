#!/usr/bin/env python3
"""
backtest_whole_shares.py
========================
Backtest the ENGULFING bull_overnight_tl10 strategy the way the LIVE
executor actually trades it, so that the fractional -> whole-share sizing
change can be measured rather than assumed.

WHY THIS SCRIPT EXISTS
----------------------
The published backtest (reports/engulfing_summary.md) is a PER-TRADE,
RETURN-BASED simulation: no capital, no position size, no concurrency
constraint, and the position is held until the stop or target is hit
however many sessions that takes.

The live executor is materially different:

  * capital is PAPER_CAPITAL_US = $10,000 (NOT the account's $103k)
  * risk per trade = 1% of capital ($100), notional capped at 2x ($200)
  * entry is a MARKET order at the open of the session after the signal bar
  * an overnight position is force-closed at the NEXT session's OPEN
    (close_deadline = open of the next weekday after the entry session),
    so a position is held for exactly ONE session
  * the signal generator computes ATR as a SIMPLE MEAN of the last 14
    true ranges over a 15-bar window, whereas the backtest uses Wilder's
    recursive smoothing seeded once at bar 14
  * the live trend filter compares c[i] with c[i-lookback]; the backtest
    compares c[i-1] with c[i-lookback]
  * the live signal universe is the top 20 names from
    data/universe_rankings.pkl, not all 60

Those divergences matter, so this script models the LIVE path directly
and reports the sizing question on top of it:

    fractional sizing   (what the backtest implicitly assumes)
    whole-share sizing  (what the executor now does, so that a
                        broker-side GTC stop can be placed)

Outputs reports/engulfing_whole_shares.json + a markdown summary.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CACHE = ROOT / "data" / "engulfing_universe.pkl"
RANKINGS = ROOT / "data" / "universe_rankings.pkl"
OUT_JSON = ROOT / "reports" / "engulfing_whole_shares.json"
OUT_MD = ROOT / "reports" / "engulfing_whole_shares_summary.md"

# ── live executor constants (src/trading/executor_live.py) ──────────────────
PAPER_CAPITAL_US = 10_000.0
RISK_PCT = 0.01
MAX_RISK_MULT = 2.0
TREND_LOOKBACK = 10
LIVE_TOP_N = 20

# ── reporting knobs ─────────────────────────────────────────────────────────
CAPITALS = [10_000.0, 50_000.0, 103_000.0, 250_000.0]
COST_BPS = [0.0, 10.0]


# ══════════════════════════════════════════════════════════════════════════════
# Live-path signal detection
# ══════════════════════════════════════════════════════════════════════════════

def live_atr(high, low, close, i: int) -> float:
    """Exactly the live generator's ATR (src/trading/engulfing_live.py::_atr14).

    The live call passes the 15-bar window h[i-14:i+1], so n == 15 and the
    Wilder recursion never runs — the result is a SIMPLE MEAN of 14 true
    ranges. This is NOT the same as the backtest's Wilder-smoothed ATR.
    """
    lo = max(0, i - 14)
    hh = high[lo:i + 1]
    ll = low[lo:i + 1]
    cc = close[lo:i + 1]
    n = len(cc)
    tr = np.zeros(n)
    tr[0] = hh[0] - ll[0]
    for k in range(1, n):
        tr[k] = max(hh[k] - ll[k], abs(hh[k] - cc[k - 1]), abs(ll[k] - cc[k - 1]))
    if n < 15:
        return float(np.mean(tr))
    return float(np.mean(tr[1:15]))


def detect_signals(df, trend_lookback: int = TREND_LOOKBACK) -> list[dict]:
    """Bullish engulfing + downtrend, evaluated on every bar.

    Mirrors the live detector's per-bar logic (the live generator simply
    restricts the match to one target date; backtesting evaluates every
    bar as if it were "today").
    """
    df = df.dropna().sort_index()
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    dates = df.index

    out = []
    for i in range(1, len(c)):
        p_o, p_c = o[i - 1], c[i - 1]
        c_o, c_c = o[i], c[i]
        # bullish engulfing: prior red, current body engulfs prior body
        if not (p_c < p_o and c_c > p_o and c_c > p_c):
            continue
        # live trend filter: c[i] must be below c[i-lookback]
        if i >= trend_lookback and c[i] >= c[i - trend_lookback]:
            continue

        atr_val = live_atr(h, l, c, i)
        entry_proxy = float(c[i])          # live signal's entry_price proxy
        stop = float(h[i]) - atr_val
        if stop <= 0 or stop >= entry_proxy:
            continue                        # nonsense geometry — live skips
        risk = abs(entry_proxy - stop)
        target = entry_proxy + 2.0 * risk

        out.append({
            "bar_i": i,
            "bar_date": str(dates[i].date()),
            "proxy": entry_proxy,
            "stop": stop,
            "target": target,
            "atr": atr_val,
        })
    return out


def build_trades(price_df, tickers) -> list[dict]:
    """Turn signals into concrete entry/exit pairs on the LIVE path.

    Entry : market order at the OPEN of the session after the signal bar.
    Exit  : stop / target hit during that session, otherwise the OPEN of
            the following session (the live close deadline).
    """
    trades = []
    for tk in tickers:
        df = price_df[tk].dropna().sort_index()
        if len(df) < 30:
            continue
        o = df["Open"].values
        h = df["High"].values
        l = df["Low"].values
        c = df["Close"].values
        dates = df.index

        for sig in detect_signals(df):
            i = sig["bar_i"]
            if i + 1 >= len(c):
                continue

            entry = float(o[i + 1])            # market order at the open
            stop = sig["stop"]
            target = sig["target"]
            if entry <= 0:
                continue
            # live refuses to enter when the price has already gapped
            # through the stop (would be an instant stop-out)
            if entry <= stop:
                continue

            # ── exit ─────────────────────────────────────────────────────
            exit_price = None
            exit_reason = None
            if i + 2 >= len(c):
                # no following session in the data — cannot close, skip
                continue
            if l[i + 1] <= stop:
                exit_price, exit_reason = stop, "stop"
            elif h[i + 1] >= target:
                exit_price, exit_reason = target, "target"
            else:
                exit_price = float(o[i + 2])    # close deadline: next open
                exit_reason = "overnight_exit"

            trades.append({
                "ticker": tk,
                "signal_date": sig["bar_date"],
                "entry_date": str(dates[i + 1].date()),
                "exit_date": str(dates[i + 2].date()),
                "proxy": sig["proxy"],
                "entry": entry,
                "stop": stop,
                "target": target,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "risk_per_share": abs(sig["proxy"] - stop),
            })
    return trades


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio simulation
# ══════════════════════════════════════════════════════════════════════════════

def size_position(qty_raw: float, entry_proxy: float, risk_per_share: float,
                  risk_amt: float, whole_shares: bool, respect_notional: bool = False):
    """Apply the executor's sizing rules.

    Fractional path:
        qty = risk_amt / stop_dist, then capped so notional <= 2x risk.

    Whole-share path (what the executor now does):
        long positions round DOWN to whole shares, because a fractional
        position cannot carry a broker-side GTC stop ("stop/stop_limit
        fractional GTC orders are not enabled"). If the capped size is
        under one share the setup is skipped UNLESS one share's risk fits
        the trade's risk budget.

    `respect_notional` additionally skips the setup when taking one share
        would push the notional past the 2x cap — the executor's current
        behaviour takes that share anyway (see the cap-breach note in the
        report).
    """
    qty = qty_raw
    max_notional = risk_amt * MAX_RISK_MULT
    capped = False
    if entry_proxy * qty > max_notional:
        qty = max_notional / entry_proxy
        capped = True
    if not whole_shares:
        return qty, capped, "ok"

    whole = math.floor(qty)
    if whole < 1:
        # The capped size is under one share. Taking one share
        # risks (stop distance) dollars, which is only acceptable
        # if it stays within the trade's risk budget — otherwise
        # the position would be riskier than the strategy
        # intends and the setup must be skipped.
        one_share_risk = risk_per_share
        if one_share_risk > risk_amt:
            return 0.0, capped, "skipped_risk"
        if respect_notional and entry_proxy > max_notional:
            # One share of this stock alone exceeds the notional cap.
            return 0.0, capped, "skipped_notional"
        whole = 1
    return float(whole), capped, "ok"


def simulate(trades, capital: float, whole_shares: bool, cost_bps: float,
             respect_notional: bool = False):
    """Event-driven portfolio sim on the live path.

    Positions entered at a session's open are closed at the NEXT
    session's open (or earlier on a stop/target hit), so cohorts are
    processed exit-first, then entry.
    """
    # ── event ordering ────────────────────────────────────────────────────
    # A position entered at the open of session D is closed at the open of
    # session D+1. So on any given session date: FIRST the previous
    # cohort's exits realise their P&L, THEN the new cohort's entries are
    # sized against the capital available at that moment. Getting this
    # order wrong lets a cohort size itself against capital that has not
    # been realised yet.
    by_entry: dict[str, list] = {}
    for t in trades:
        by_entry.setdefault(t["entry_date"], []).append(t)
    all_dates = sorted(set(by_entry) | {t["exit_date"] for t in trades})

    realized_capital = capital      # live tracks realized capital only
    n_skipped = 0
    n_trades = 0
    gross_notional = 0.0
    total_risk = 0.0
    pnls: list[float] = []
    bps_list: list[float] = []
    wins = 0
    stops = 0
    targets = 0
    time_exits = 0
    n_cap_breach = 0          # positions whose notional exceeded the 2x cap
    n_risk_breach = 0         # positions whose risk exceeded the 1% budget
    per_trade: list[dict] = []
    qty_hist: dict[str, int] = {}

    for d in all_dates:
        # ── 1. exits ──────────────────────────────────────────────────────
        for t in trades:
            if t["exit_date"] != d:
                continue
            # the trade's sizing must be resolved consistently at entry
            # time; recompute it here from the SAME realized-capital rule
            # is not possible (capital changes), so instead exits are
            # realised from a per-trade record built during the entry pass.
            rec = t.get("_rec")
            if rec is None:
                continue
            gross = rec["pnl_dollars"]
            realized_capital += gross
            n_trades += 1
            gross_notional += rec["notional"]
            total_risk += rec["risk_dollars"]
            pnls.append(rec["ret"])
            bps_list.append(rec["pnl_bps"])
            if rec["pnl_bps"] > 0:
                wins += 1
            if rec["exit_reason"] == "stop":
                stops += 1
            elif rec["exit_reason"] == "target":
                targets += 1
            else:
                time_exits += 1
            if rec["notional"] > rec["max_notional"] * 1.000001:
                n_cap_breach += 1
            if rec["risk_dollars"] > rec["risk_amt"] * 1.000001:
                n_risk_breach += 1
            per_trade.append({
                "ticker": t["ticker"], "entry_date": t["entry_date"],
                "exit_date": t["exit_date"], "qty": rec["qty"],
                "entry": t["entry"], "exit": t["exit_price"],
                "exit_reason": rec["exit_reason"],
                "pnl_dollars": gross, "pnl_bps": rec["pnl_bps"],
                "risk_dollars": rec["risk_dollars"],
                "notional": rec["notional"],
            })

        # ── 2. entries ────────────────────────────────────────────────────
        for t in by_entry.get(d, []):
            risk_amt = realized_capital * RISK_PCT
            stop_dist = abs(t["proxy"] - t["stop"])
            if stop_dist == 0:
                n_skipped += 1
                continue
            qty_raw = risk_amt / stop_dist
            qty, capped, status = size_position(
                qty_raw, t["proxy"], t["risk_per_share"], risk_amt, whole_shares,
                respect_notional=respect_notional)
            if status != "ok" or qty <= 0:
                n_skipped += 1
                continue

            # P&L on the ACTUAL fill prices
            gross = (t["exit_price"] - t["entry"]) * qty
            gross -= t["entry"] * qty * cost_bps / 10_000.0
            ret = (t["exit_price"] - t["entry"]) / t["entry"]
            pnl_bps = ret * 10_000 - cost_bps
            notional = t["entry"] * qty
            t["_rec"] = {
                "qty": qty,
                "pnl_dollars": gross,
                "pnl_bps": pnl_bps,
                "ret": ret,
                "risk_dollars": t["risk_per_share"] * qty,
                "notional": notional,
                "exit_reason": t["exit_reason"],
                "risk_amt": risk_amt,
                "max_notional": risk_amt * MAX_RISK_MULT,
            }
            key = f"{qty:.4f}"
            qty_hist[key] = qty_hist.get(key, 0) + 1

    # ── metrics ──────────────────────────────────────────────────────────
    # IMPORTANT: the per-trade RETURN stream is independent of position
    # size, so a Sharpe computed from it cannot distinguish fractional
    # from whole-share sizing — it would report the same number for both.
    # The portfolio metric has to be built from DOLLAR P&L realised over
    # time, which is what the account actually experiences.
    rets = np.array(pnls) if pnls else np.array([0.0])
    n = len(pnls)

    # ── dollar equity curve, bucketed by exit date ──────────────────────
    equity = capital
    curve: list[tuple[str, float]] = []
    by_exit: dict[str, float] = {}
    for p in per_trade:
        by_exit[p["exit_date"]] = by_exit.get(p["exit_date"], 0.0) + p["pnl_dollars"]
    for d in sorted(by_exit):
        equity += by_exit[d]
        curve.append((d, equity))

    if len(curve) > 1:
        # period returns of the dollar equity curve
        eq = np.array([c[1] for c in curve])
        prev = np.concatenate(([capital], eq[:-1]))
        per = (eq - prev) / prev
        per = per[prev > 0]
        # annualise: mean per-period return over the calendar span
        n_days = len(curve)
        span_years = max((pd.Timestamp(curve[-1][0]) - pd.Timestamp(curve[0][0])).days / 365.25, 1e-9)
        per_year = n_days / span_years
        ann_ret_dollar = float(np.prod(1 + per) ** (1 / span_years) - 1) if len(per) else 0.0
        ann_vol_dollar = float(np.std(per) * math.sqrt(per_year)) if len(per) > 1 else 0.0
        sharpe_dollar = (ann_ret_dollar / ann_vol_dollar) if ann_vol_dollar > 0 else 0.0
        peak = np.maximum.accumulate(eq)
        max_dd_dollar = float(np.min(eq / peak - 1)) if len(eq) else 0.0
    else:
        ann_ret_dollar = ann_vol_dollar = sharpe_dollar = 0.0
        max_dd_dollar = 0.0

    total_ret = float(np.prod(1 + rets) - 1) if n else 0.0
    ann_ret = ((1 + total_ret) ** (252 / max(n, 1)) - 1) if n else 0.0
    ann_vol = float(np.std(rets) * math.sqrt(252)) if n > 1 else 0.0
    sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0.0
    # max drawdown on the per-trade return curve
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.min(eq / peak - 1)) if n else 0.0

    return {
        "n_trades": n,
        "n_skipped": n_skipped,
        # ── return-stream metrics (size-independent; for reference only) ──
        "total_return_pct": total_ret * 100,
        "ann_ret_pct": ann_ret * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "win_rate_pct": (wins / n * 100) if n else 0.0,
        "avg_bps": float(np.mean(bps_list)) if bps_list else 0.0,
        # ── DOLLAR portfolio metrics (size-dependent — the ones that
        #    actually answer the whole-share question) ───────────────────
        "total_pnl_dollars": float(sum(p["pnl_dollars"] for p in per_trade)),
        "ann_ret_dollar_pct": ann_ret_dollar * 100,
        "sharpe_dollar": sharpe_dollar,
        "max_dd_dollar_pct": max_dd_dollar * 100,
        "avg_risk_dollars": (total_risk / n) if n else 0.0,
        "total_risk_dollars": total_risk,
        "avg_gross_notional": (gross_notional / n) if n else 0.0,
        "max_gross_notional": max((p["notional"] for p in per_trade), default=0.0),
        "stop_rate_pct": (stops / n * 100) if n else 0.0,
        "target_rate_pct": (targets / n * 100) if n else 0.0,
        "time_exit_rate_pct": (time_exits / n * 100) if n else 0.0,
        "n_cap_breach": n_cap_breach,
        "n_risk_breach": n_risk_breach,
        "qty_hist": dict(sorted(qty_hist.items(), key=lambda kv: -kv[1])[:12]),
        "final_capital": realized_capital,
        # Per-trade rows are intentionally NOT serialised: 24 configurations
        # x thousands of trades makes a >100MB file that the remote rejects.
        # Re-run with --verbose to dump them.
    }


def build_unbounded_trades(price_df, tickers) -> list[dict]:
    """Published-style trades: hold until stop/target, unbounded sessions.

    The published backtest (reports/engulfing_summary.md) uses this rule,
    while the live executor force-closes at the NEXT session's open. The
    two are different strategies and the published Sharpe does not
    describe live behaviour — so both are reported side by side.
    """
    trades = []
    for tk in tickers:
        df = price_df[tk].dropna().sort_index()
        if len(df) < 30:
            continue
        o = df["Open"].values
        h = df["High"].values
        l = df["Low"].values
        c = df["Close"].values
        dates = df.index
        for sig in detect_signals(df):
            i = sig["bar_i"]
            if i + 1 >= len(c):
                continue
            entry = float(o[i + 1])
            stop, target = sig["stop"], sig["target"]
            if entry <= 0 or entry <= stop:
                continue
            exit_price = exit_reason = None
            exit_bar = -1
            for j in range(i + 2, len(c)):
                if l[j] <= stop:
                    exit_price, exit_reason, exit_bar = stop, "stop", j
                    break
                if h[j] >= target:
                    exit_price, exit_reason, exit_bar = target, "target", j
                    break
            if exit_price is None:
                continue
            trades.append({
                "ticker": tk, "signal_date": sig["bar_date"],
                "entry_date": str(dates[i + 1].date()),
                "exit_date": str(dates[exit_bar].date()),
                "proxy": sig["proxy"], "entry": entry, "stop": stop,
                "target": target, "exit_price": exit_price,
                "exit_reason": exit_reason,
                "risk_per_share": abs(sig["proxy"] - stop),
            })
    return trades


def sizing_diagnostics(trades, capital: float) -> dict:
    """How the sizing rule behaves: skips, notional-cap breaches, risk."""
    risk_amt = capital * RISK_PCT
    max_notional = risk_amt * MAX_RISK_MULT
    n = len(trades)
    n_risk_skip = n_notional_breach = 0
    max_notional_seen = 0.0
    risk_sum = 0.0
    for t in trades:
        sd = abs(t["proxy"] - t["stop"])
        if sd == 0:
            continue
        q = risk_amt / sd
        if t["proxy"] * q > max_notional:
            q = max_notional / t["proxy"]
        whole = math.floor(q)
        if whole < 1:
            if sd > risk_amt:
                n_risk_skip += 1
                continue
            whole = 1
            if t["proxy"] > max_notional:
                n_notional_breach += 1
        notional = t["entry"] * whole
        max_notional_seen = max(max_notional_seen, notional)
        risk_sum += sd * whole
    return {
        "capital": capital,
        "risk_amt": risk_amt,
        "max_notional": max_notional,
        "n_signals": n,
        "n_risk_skipped": n_risk_skip,
        "n_notional_breach": n_notional_breach,
        "pct_notional_breach": (n_notional_breach / n * 100) if n else 0.0,
        "max_notional_seen": max_notional_seen,
        "avg_risk_dollars": (risk_sum / n) if n else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading cached universe …")
    with open(CACHE, "rb") as fh:
        price_df = pickle.load(fh)
    with open(RANKINGS, "rb") as fh:
        ranked = pickle.load(fh)
    live_universe = [t for t in ranked[:LIVE_TOP_N] if t in price_df]
    all_universe = [t for t in price_df.keys()]

    print(f"  cached tickers : {len(all_universe)}")
    print(f"  live universe  : {len(live_universe)} -> {live_universe}")

    results = {}
    diagnostics = {}
    comparisons = {}

    for uni_name, universe in (("live_top20", live_universe),
                               ("all60", all_universe)):
        print(f"\nBuilding live-path trades for '{uni_name}' …")
        trades = build_trades(price_df, universe)
        print(f"  {len(trades)} signal/trade candidates")
        if not trades:
            continue

        for cap in CAPITALS:
            diagnostics[f"{uni_name}|cap{int(cap)}"] = sizing_diagnostics(trades, cap)

        for cap in CAPITALS:
            for cost in COST_BPS:
                for mode in ("frac", "whole", "whole_capped"):
                    whole = mode != "frac"
                    respect = (mode == "whole_capped")
                    key = f"{uni_name}|cap{int(cap)}|cost{cost:g}|{mode}"
                    r = simulate(trades, cap, whole, cost, respect_notional=respect)
                    results[key] = r
                    print(f"  {key:56s} n={r['n_trades']:5d} skip={r['n_skipped']:4d} "
                          f"sharpe$={r['sharpe_dollar']:6.2f} "
                          f"pnl$={r['total_pnl_dollars']:10,.2f} "
                          f"annRet%={r['ann_ret_dollar_pct']:7.2f} "
                          f"dd%={r['max_dd_dollar_pct']:7.2f} "
                          f"avgrisk=${r['avg_risk_dollars']:7.2f} "
                          f"maxNot${r['max_gross_notional']:8,.2f}")

    # ── live exit rule vs published (unbounded) exit rule ────────────────
    print(f"\n{'═'*72}\nEXIT-RULE COMPARISON (live top-20, $10k capital)\n{'═'*72}")
    live_tr = build_trades(price_df, live_universe)
    unb_tr = build_unbounded_trades(price_df, live_universe)
    for label, tr in (("live 1-session hold", live_tr),
                       ("published unbounded hold", unb_tr)):
        reasons = {}
        for t in tr:
            reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
        r = simulate(tr, 10_000.0, True, 0.0)
        comparisons[label] = {
            "n": len(tr),
            "exit_reasons": reasons,
            "sharpe_dollar": r["sharpe_dollar"],
            "total_pnl_dollars": r["total_pnl_dollars"],
        }
        print(f"  {label:26s} n={len(tr):5d}  sharpe$={r['sharpe_dollar']:6.2f}  "
              f"P&L ${r['total_pnl_dollars']:10,.2f}  {reasons}")

    with open(OUT_JSON, "w") as fh:
        json.dump({"results": results, "sizing_diagnostics": diagnostics,
                   "exit_rule_comparison": comparisons}, fh, indent=2, default=str)
    print(f"\nSaved → {OUT_JSON}")
    _write_md(results, diagnostics, comparisons)


def _write_md(results: dict, diagnostics: dict, comparisons: dict) -> None:
    lines = [
        "# Engulfing `bull_overnight_tl10` — Whole-Share vs Fractional Sizing",
        "",
        "Simulated on the **live execution path**: market entry at the next",
        "session's open, one-session overnight hold, exit at the next open,",
        "live simple-mean ATR14, live trend filter, live top-20 universe.",
        "",
        "Capital base, risk per trade (1%) and the 2x notional cap match the",
        "live executor (`PAPER_CAPITAL_US = $10,000`).",
        "",
        "Metrics are built from the **dollar** equity curve, because the",
        "per-trade return stream is independent of position size and cannot",
        "distinguish the two sizing rules.",
        "",
        "## Sizing modes",
        "",
        "- `frac` — fractional shares (what the published backtest assumes)",
        "- `whole` — round DOWN to whole shares (what the executor does now, so a",
        "  broker-side GTC stop can be placed)",
        "- `whole_capped` — as `whole`, but additionally SKIP a setup when one",
        "  share alone would breach the 2x notional cap",
        "",
        "## Results",
        "",
        "| Universe | Capital | Cost bps | Sizing | N | Skipped | Sharpe$ | "
        "Total P&L $ | Ann Ret% | Max DD% | Avg risk $ | Max notional $ |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(results, key=lambda k: (k.split("|")[0], k.split("|")[1], k.split("|")[2], k.split("|")[3])):
        uni, cap, cost, sizing = key.split("|")
        r = results[key]
        label = {"frac": "fractional", "whole": "whole",
                 "whole_capped": "whole (capped)"}[sizing]
        lines.append(
            f"| {uni} | ${int(cap[3:])} | {cost.replace('cost','')} | {label} | "
            f"{r['n_trades']} | {r['n_skipped']} | {r['sharpe_dollar']:.2f} | "
            f"{r['total_pnl_dollars']:,.2f} | {r['ann_ret_dollar_pct']:.2f} | "
            f"{r['max_dd_dollar_pct']:.2f} | {r['avg_risk_dollars']:.2f} | "
            f"{r['max_gross_notional']:,.2f} |"
        )

    lines += [
        "",
        "## Sizing diagnostics — notional-cap breach by capital base",
        "",
        "| Universe | Capital | Risk $ | Max notional $ | Risk-skipped | "
        "Notional breaches | Breach % | Max notional seen $ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(diagnostics, key=lambda k: (k.split("|")[0], int(k.split("|")[1][3:]))):
        d = diagnostics[key]
        uni, cap = key.split("|")
        lines.append(
            f"| {uni} | ${int(cap[3:]):,} | {d['risk_amt']:,.2f} | "
            f"{d['max_notional']:,.2f} | {d['n_risk_skipped']} | "
            f"{d['n_notional_breach']} | {d['pct_notional_breach']:.1f}% | "
            f"{d['max_notional_seen']:,.2f} |"
        )

    lines += [
        "",
        "## Exit rule — live vs published",
        "",
        "| Exit rule | N | Sharpe$ | Total P&L $ | Exit reasons |",
        "|---|---:|---:|---:|---|",
    ]
    for label, c in comparisons.items():
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(c["exit_reasons"].items()))
        lines.append(f"| {label} | {c['n']} | {c['sharpe_dollar']:.2f} | "
                     f"{c['total_pnl_dollars']:,.2f} | {reasons} |")
    lines += [
        "",
        "The published backtest holds until the stop or target with NO time",
        "limit. The live executor force-closes at the NEXT session's open.",
        "They are different strategies, so the published Sharpe does not",
        "describe live behaviour.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Summary saved → {OUT_MD}")


if __name__ == "__main__":
    main()
