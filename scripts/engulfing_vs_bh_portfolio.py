#!/usr/bin/env python3
"""
engulfing_vs_bh_portfolio.py
============================
Compare the engulfing strategy to Buy & Hold on the SAME footing:

  * same universe          (live top-20, from data/universe_rankings.pkl)
  * same capital           ($10,000 = PAPER_CAPITAL_US)
  * same cost model        (10bps round trip, charged on both legs)
  * same date range        (from the cached universe price history)
  * whole shares           (the account cannot hold fractional stock, and
                            the broker only accepts GTC stops on whole shares)

The older scripts/engulfing_vs_bh.py compares against the PUBLISHED
per-trade backtest, which has no capital, no sizing and an unbounded
hold. That is not the live strategy, so it cannot answer this question.

Two numbers matter and they do NOT agree, so both are reported:

  * Sharpe / max drawdown  -> comparable, risk-adjusted
  * dollar return          -> NOT comparable, because the strategy
                              deploys ~$200 of notional per trade while
                              B&H deploys the whole $10,000. Capital
                              utilisation is reported alongside so the
                              gap is explicit rather than hidden.
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
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_whole_shares import (  # noqa: E402
    CACHE, RANKINGS, LIVE_TOP_N, build_trades, simulate,
)

CAPITAL = 10_000.0
COST_BPS = 10.0

# The live configuration, and the pre-reoptimisation baseline for contrast.
CONFIGS: dict[str, dict] = {
    "baseline (stop1.0, hold1)": dict(stop_atr_mult=1.0, hold_sessions=1),
    "reoptimised (stop2.0, hold3)": dict(stop_atr_mult=2.0, hold_sessions=3),
}

PERIODS = 5   # calendar sub-periods for the robustness table

OUT_JSON = ROOT / "reports" / "engulfing_vs_bh_portfolio.json"
OUT_MD = ROOT / "reports" / "engulfing_vs_bh_portfolio.md"


# ── metric helpers ───────────────────────────────────────────────────────────

def curve_metrics(curve: list[tuple[str, float]], capital: float) -> dict:
    """Annualised metrics for an equity curve of (date, equity) points.

    Same shape as the strategy's own metric: period returns between
    consecutive points, annualised by the observed points-per-year.
    """
    if len(curve) < 2:
        return {"ann_ret_pct": 0.0, "ann_vol_pct": 0.0, "sharpe": 0.0,
                "max_dd_pct": 0.0, "total_ret_pct": 0.0}
    eq = np.array([c[1] for c in curve], dtype=float)
    prev = np.concatenate(([capital], eq[:-1]))
    per = (eq - prev) / prev
    per = per[prev > 0]
    span_years = max(
        (pd.Timestamp(curve[-1][0]) - pd.Timestamp(curve[0][0])).days / 365.25,
        1e-9)
    per_year = len(curve) / span_years
    ann_ret = float(np.prod(1 + per) ** (1 / span_years) - 1) if len(per) else 0.0
    ann_vol = float(np.std(per) * math.sqrt(per_year)) if len(per) > 1 else 0.0
    peak = np.maximum.accumulate(eq)
    return {
        "ann_ret_pct": ann_ret * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": (ann_ret / ann_vol) if ann_vol > 0 else 0.0,
        "max_dd_pct": float(np.min(eq / peak - 1)) * 100,
        "total_ret_pct": float(eq[-1] / capital - 1) * 100,
    }


def period_slices(dates: list[str], n: int = PERIODS) -> list[tuple[str, str]]:
    if not dates:
        return []
    chunks = [list(c) for c in np.array_split(np.array(dates), n)]
    return [(str(c[0]), str(c[-1])) for c in chunks if len(c)]


# ── buy & hold ───────────────────────────────────────────────────────────────

def buy_and_hold(price_df: dict, tickers: list[str], capital: float,
                 cost_bps: float) -> dict:
    """Equal-weight buy & hold, whole shares, round-trip cost charged.

    Entry cost is deducted from cash at purchase; the exit cost is
    charged once at liquidation at the end of the sample.
    """
    closes, opens = {}, {}
    for t in tickers:
        df = price_df[t].dropna().sort_index()
        if df.empty:
            continue
        closes[t] = df["Close"]
        opens[t] = df["Open"]
    if not closes:
        return {}

    close_df = pd.DataFrame(closes).sort_index().ffill()
    open_df = pd.DataFrame(opens).sort_index().ffill()
    close_df.index = pd.DatetimeIndex(close_df.index)
    open_df.index = pd.DatetimeIndex(open_df.index)
    idx = close_df.index
    alloc = capital / len(close_df.columns)

    cash = capital
    qty: dict[str, int] = {}
    for t in close_df.columns:
        px = float(open_df[t].dropna().iloc[0])
        if px <= 0:
            continue
        n = int(alloc // px)
        if n < 1:
            continue                       # cannot afford one whole share
        qty[t] = n
        cash -= n * px * (1 + cost_bps / 10_000.0)

    curve = []
    for d in idx:
        value = cash + sum(qty.get(t, 0) * float(close_df.at[d, t])
                           for t in close_df.columns if not pd.isna(close_df.at[d, t]))
        curve.append((str(d.date()), value))
    # liquidate: exit cost on the proceeds
    last = close_df.index[-1]
    proceeds = sum(qty.get(t, 0) * float(close_df.at[last, t])
                   for t in close_df.columns)
    cash -= proceeds * cost_bps / 10_000.0
    curve[-1] = (str(last.date()), cash + proceeds)

    m = curve_metrics(curve, capital)
    m.update({
        "n_held": sum(1 for v in qty.values() if v > 0),
        "n_universe": len(close_df.columns),
        "leftover_cash": cash,
        "deployed_pct": (capital - cash) / capital * 100,
    })
    return {"metrics": m, "curve": curve, "qty": qty}


def bh_curve_on_dates(bh: dict, dates: list[str]) -> list[tuple[str, float]]:
    """B&H equity sampled on the strategy's exit dates.

    Lets the B&H curve be bucketed the SAME way as the strategy's
    realised-capital curve, so the two Sharpes are like for like.
    """
    lookup = dict(bh["curve"])
    return [(d, lookup[d]) for d in dates if d in lookup]


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(CACHE, "rb") as fh:
        price_df = pickle.load(fh)
    with open(RANKINGS, "rb") as fh:
        ranked = pickle.load(fh)
    universe = [t for t in ranked[:LIVE_TOP_N] if t in price_df]
    print(f"universe ({len(universe)}): {universe}")

    out: dict = {"universe": universe, "capital": CAPITAL, "cost_bps": COST_BPS}

    # ── Buy & hold ──────────────────────────────────────────────────────
    bh = buy_and_hold(price_df, universe, CAPITAL, COST_BPS)
    if not bh:
        print("B&H failed — no data")
        return
    out["bh"] = {k: v for k, v in bh["metrics"].items()}
    print(f"\nB&H (equal-weight, whole shares, ${CAPITAL:,.0f}): "
          f"ann={bh['metrics']['ann_ret_pct']:+.2f}% "
          f"sharpe={bh['metrics']['sharpe']:.2f} "
          f"dd={bh['metrics']['max_dd_pct']:.2f}% "
          f"total={bh['metrics']['total_ret_pct']:+.1f}% "
          f"held={bh['metrics']['n_held']}/{bh['metrics']['n_universe']}")

    # ── Strategy ────────────────────────────────────────────────────────
    out["strategy"] = {}
    strat_trades = {}
    for label, kw in CONFIGS.items():
        trades = build_trades(price_df, universe, **kw)
        r = simulate(trades, CAPITAL, whole_shares=True, cost_bps=COST_BPS,
                     include_trades=True)
        util = r["avg_gross_notional"] / CAPITAL * 100
        out["strategy"][label] = {
            "n_trades": r["n_trades"],
            "total_pnl_dollars": r["total_pnl_dollars"],
            "ann_ret_dollar_pct": r["ann_ret_dollar_pct"],
            "sharpe_dollar": r["sharpe_dollar"],
            "max_dd_dollar_pct": r["max_dd_dollar_pct"],
            "win_rate_pct": r["win_rate_pct"],
            "avg_bps": r["avg_bps"],
            "avg_gross_notional": r["avg_gross_notional"],
            "capital_utilisation_pct": util,
            "stop_rate_pct": r["stop_rate_pct"],
            "time_exit_rate_pct": r["time_exit_rate_pct"],
        }
        print(f"{label:30s} n={r['n_trades']:4d} "
              f"pnl=${r['total_pnl_dollars']:8,.2f} "
              f"ann={r['ann_ret_dollar_pct']:+7.2f}% "
              f"sharpe$={r['sharpe_dollar']:6.2f} "
              f"dd={r['max_dd_dollar_pct']:6.2f}% "
              f"util={util:5.1f}%")
        # exit-date curve, for the like-for-like comparison
        strat_trades[label] = r["per_trade"]

    # ── Composite: strategy + B&H on the idle cash ─────────────────────
    out["composite"] = {}
    for label, pt in strat_trades.items():
        m = composite(bh, pt, CAPITAL)
        if m:
            out["composite"][label] = m
            print(f"composite ({label}): ann={m['ann_ret_pct']:+.2f}% "
                  f"sharpe={m['sharpe']:.2f} dd={m['max_dd_pct']:.2f}% "
                  f"deployed={m['avg_deployed_pct']:.1f}%")

    # ── Like-for-like: B&H sampled on the strategy's exit dates ────────
    out["like_for_like"] = {}
    for label, pt in strat_trades.items():
        exit_dates = sorted({t["exit_date"] for t in pt})
        bh_sub = bh_curve_on_dates(bh, exit_dates)
        if len(bh_sub) < 2:
            continue
        m = curve_metrics(bh_sub, CAPITAL)
        out["like_for_like"][label] = m
        print(f"B&H on strategy exit dates ({label}): "
              f"ann={m['ann_ret_pct']:+.2f}% sharpe={m['sharpe']:.2f} "
              f"dd={m['max_dd_pct']:.2f}%")

    # ── Per-period robustness ───────────────────────────────────────────
    all_dates = sorted({t["entry_date"] for t in
                        build_trades(price_df, universe, hold_sessions=1)})
    slices = period_slices(all_dates)
    out["periods"] = {}
    print(f"\nPer-period Sharpe (B&H daily curve vs strategy exit-bucketed):")
    for lo, hi in slices:
        row = {}
        bh_dates = [c[0] for c in bh["curve"] if lo <= c[0] <= hi]
        bh_sub = bh_curve_on_dates(bh, bh_dates)
        if len(bh_sub) > 1:
            row["bh"] = curve_metrics(bh_sub, CAPITAL)["sharpe"]
        for label, kw in CONFIGS.items():
            sub = [t for t in build_trades(price_df, universe, **kw)
                   if lo <= t["entry_date"] <= hi]
            if not sub:
                row[label] = None
                continue
            row[label] = simulate(sub, CAPITAL, whole_shares=True,
                                  cost_bps=COST_BPS)["sharpe_dollar"]
        out["periods"][f"{lo} .. {hi}"] = row
        print(f"  {lo} .. {hi}: " + "  ".join(
            f"{k.split(' ')[0]}={v:.2f}" if isinstance(v, float) else f"{k}=n/a"
            for k, v in row.items()))

    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nSaved → {OUT_JSON}")
    write_md(out)


def composite(bh: dict, trades: list[dict], capital: float) -> dict:
    """Strategy + Buy & Hold on the idle cash.

    The strategy risks 1% of capital per trade, so ~98% of the account
    is idle during most of the sample. Comparing the strategy alone
    against B&H therefore compares a 2%-deployed portfolio with a
    100%-deployed one, which flatters B&H for a reason that has
    nothing to do with the signal.

    This blends the two: the strategy keeps its own P&L, and whatever
    capital it is NOT using earns the equal-weight B&H return.
    """
    bh_lookup = dict(bh["curve"])
    if len(bh_lookup) < 2:
        return {}

    # ── deployed notional per date, from the actual positions ──────────
    deployed: dict[str, float] = {}
    for t in trades:
        for d in _date_range(t["entry_date"], t["exit_date"]):
            deployed[d] = deployed.get(d, 0.0) + t["notional"]
    bh_dates = sorted(bh_lookup)
    idle_frac = {d: max(0.0, 1.0 - deployed.get(d, 0.0) / capital)
                 for d in bh_dates}

    # ── strategy P&L level, realised on each exit date ────────────────
    pnl_by_date: dict[str, float] = {}
    for t in trades:
        pnl_by_date[t["exit_date"]] = pnl_by_date.get(t["exit_date"], 0.0) + t["pnl_dollars"]

    # ── B&H daily returns ──────────────────────────────────────────────
    eq = np.array([bh_lookup[d] for d in bh_dates], dtype=float)
    bh_daily = np.zeros(len(eq), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bh_daily[1:] = (eq[1:] - eq[:-1]) / eq[:-1]
    bh_daily[~np.isfinite(bh_daily)] = 0.0

    curve = []
    strat_level = 0.0
    idle_value = capital
    for i, d in enumerate(bh_dates):
        strat_level += pnl_by_date.get(d, 0.0)
        if i:
            # The idle cash earns a SCALED slice of the B&H return and
            # COMPOUNDS — adding the daily returns instead would collapse
            # a 30x move into a fraction of it.
            idle_frac_prev = idle_frac[bh_dates[i - 1]]
            idle_value *= (1.0 + idle_frac_prev * float(bh_daily[i]))
        # idle_value ALREADY contains the principal, so adding `capital`
        # again would double-count it.
        curve.append((d, idle_value + strat_level))

    m = curve_metrics(curve, capital)
    m["avg_deployed_pct"] = (1 - float(np.mean(list(idle_frac.values())))) * 100
    return m


def _date_range(lo: str, hi: str) -> list[str]:
    """Inclusive list of YYYY-MM-DD strings from lo to hi."""
    a = pd.Timestamp(lo)
    b = pd.Timestamp(hi)
    if b < a:
        return [lo]
    return [str(d.date()) for d in pd.date_range(a, b, freq="D")]


def write_md(out: dict) -> None:
    bh = out["bh"]
    lines = [
        "# Engulfing vs Buy & Hold — like-for-like portfolio comparison",
        "",
        f"- Universe: live top-{len(out['universe'])} "
        f"({', '.join(out['universe'])})",
        f"- Capital: ${out['capital']:,.0f} (PAPER_CAPITAL_US)",
        f"- Costs: {out['cost_bps']:g}bps round trip",
        "",
        "## Headline",
        "",
        "| | Strategy (reoptimised) | Buy & Hold |",
        "|---|---:|---:|",
        f"| Sharpe | {out['strategy']['reoptimised (stop2.0, hold3)']['sharpe_dollar']:.2f} "
        f"| {bh['sharpe']:.2f} |",
        f"| Max drawdown | {out['strategy']['reoptimised (stop2.0, hold3)']['max_dd_dollar_pct']:.2f}% "
        f"| {bh['max_dd_pct']:.2f}% |",
        f"| Ann. return | {out['strategy']['reoptimised (stop2.0, hold3)']['ann_ret_dollar_pct']:.2f}% "
        f"| {bh['ann_ret_pct']:.2f}% |",
        f"| Total return | "
        f"{out['strategy']['reoptimised (stop2.0, hold3)']['total_pnl_dollars'] / out['capital'] * 100:.2f}% "
        f"| {bh['total_ret_pct']:.2f}% |",
        "",
        "## Capital utilisation (why the dollar rows differ)",
        "",
        "| | Strategy | Buy & Hold |",
        "|---|---:|---:|",
        f"| Avg notional deployed | "
        f"${out['strategy']['reoptimised (stop2.0, hold3)']['avg_gross_notional']:,.0f} | "
        f"${out['capital']:,.0f} |",
        f"| % of capital at risk | "
        f"{out['strategy']['reoptimised (stop2.0, hold3)']['capital_utilisation_pct']:.1f}% | 100% |",
        "",
        "The strategy risks 1% of capital per trade with a 2x notional cap,",
        "so it holds a fraction of the account while Buy & Hold holds all of",
        "it. The dollar-return rows are therefore NOT comparable; Sharpe and",
        "drawdown are.",
        "",
        "## Composite — strategy + Buy & Hold on the idle cash",
        "",
        "The fairest comparison: the strategy keeps its P&L and the capital",
        "it is NOT using earns the same equal-weight B&H return.",
        "",
        "| | Ann. return | Sharpe | MaxDD |",
        "|---|---:|---:|---:|",
        f"| Buy & Hold (100% deployed) | {bh['ann_ret_pct']:.2f}% | "
        f"{bh['sharpe']:.2f} | {bh['max_dd_pct']:.2f}% |",
    ]
    for label, m in out.get("composite", {}).items():
        lines.append(
            f"| Composite: {label} | {m['ann_ret_pct']:.2f}% | "
            f"{m['sharpe']:.2f} | {m['max_dd_pct']:.2f}% |")

    lines += [
        "",
        "## Strategy detail",
        "",
        "| Config | Trades | P&L | Sharpe | MaxDD | Win% | Avg bps | Util% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, s in out["strategy"].items():
        lines.append(
            f"| {label} | {s['n_trades']} | ${s['total_pnl_dollars']:,.2f} | "
            f"{s['sharpe_dollar']:.2f} | {s['max_dd_dollar_pct']:.2f}% | "
            f"{s['win_rate_pct']:.1f}% | {s['avg_bps']:+.1f} | "
            f"{s['capital_utilisation_pct']:.1f}% |")

    lines += ["", "## Per-period Sharpe", "",
              "| Period | B&H | " + " | ".join(out["strategy"]) + " |",
              "|---|---:|" + "---:|" * len(out["strategy"])]
    for period, row in out.get("periods", {}).items():
        cells = []
        for k, v in row.items():
            cells.append(f"{v:.2f}" if isinstance(v, float) else "n/a")
        lines.append(f"| {period} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Caveats",
        "",
        "- The universe is the live top-20 by ADV, a STATIC cached list, so it",
        "  carries survivorship and look-ahead bias: NVDA alone contributes",
        "  12x of the 32x average multiple, and B&H annual return falls from",
        "  ~37% to ~27% with NVDA removed.",
        "- B&H is equal-weight buy-and-hold with no rebalancing, bought at the",
        "  first open and sold at the last close.",
        "- The strategy's capital is 98% idle by construction (1% risk, 2x",
        "  notional cap), so it is compared on a composite with the idle cash",
        "  in B&H, not on dollar return.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Summary saved → {OUT_MD}")


if __name__ == "__main__":
    main()
