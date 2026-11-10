#!/usr/bin/env python3
"""
Backtest target/stop reoptimisation for the engulfing live path.

Sweeps the reward-to-risk multiple and the ATR stop multiple across hold
periods, and selects parameters the way you actually should:

  * parameters are chosen on a TRAIN window (first 60% of the data)
  * the chosen parameters are then scored on a TEST window (last 40%)
  * the OOS score is also broken out per calendar period, because a
    parameter set that only works in one regime is not an edge

Reporting the best config over ALL the data is in-sample and will look
much better than anything that survives out of sample. Both are printed.

Note on the sweep grid: the target multiple has an interior optimum only
while the target is still reached. Beyond ~3R the target hit rate falls
to ~0% and the target parameter stops doing anything, so the grid is
bounded at 6R and the winner is chosen from configs where the target is
still actually hit (>=1% of trades).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_whole_shares import (  # noqa: E402
    CACHE, RANKINGS, LIVE_TOP_N, build_trades, simulate,
)

TARGET_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
STOP_ATR_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
HOLD_PERIODS = [1, 2, 3, 5, 10]
COST_BPS = 10.0
CAPITAL = 10_000.0
TRAIN_FRAC = 0.6
MIN_TARGET_HIT_PCT = 1.0     # a config whose target is never hit is degenerate

OUT_JSON = ROOT / "reports" / "engulfing_target_sweep.json"
OUT_MD = ROOT / "reports" / "engulfing_target_sweep_summary.md"


def score(trades, capital=CAPITAL, cost=COST_BPS):
    if not trades:
        return None
    return simulate(trades, capital, whole_shares=True, cost_bps=cost)


def window(trades, lo=None, hi=None):
    return [t for t in trades
            if (lo is None or t["entry_date"] >= lo)
            and (hi is None or t["entry_date"] < hi)]


def main():
    price_df = pickle.load(open(CACHE, "rb"))
    ranked = pickle.load(open(RANKINGS, "rb"))
    uni = [t for t in ranked[:LIVE_TOP_N] if t in price_df]
    print(f"live universe: {len(uni)} tickers")

    cache: dict[tuple, list] = {}
    for hold in HOLD_PERIODS:
        for tm in TARGET_MULTS:
            for sm in STOP_ATR_MULTS:
                cache[(hold, tm, sm)] = build_trades(
                    price_df, uni, hold_sessions=hold,
                    target_mult=tm, stop_atr_mult=sm)
    print(f"built {len(cache)} trade sets")

    all_dates = sorted({t["entry_date"] for t in cache[(1, 2.0, 1.0)]})
    split = all_dates[int(len(all_dates) * TRAIN_FRAC)]
    print(f"train: < {split}   test: >= {split}")

    results = []
    for (hold, tm, sm), trades in cache.items():
        train = score(window(trades, None, split))
        test = score(window(trades, split, None))
        full = score(trades)
        if not train or not test or not full:
            continue
        results.append({
            "hold": hold, "target_mult": tm, "stop_atr_mult": sm,
            "n_trades": full["n_trades"],
            "train_sharpe": train["sharpe_dollar"],
            "test_sharpe": test["sharpe_dollar"],
            "train_pnl": train["total_pnl_dollars"],
            "test_pnl": test["total_pnl_dollars"],
            "full_sharpe": full["sharpe_dollar"],
            "full_pnl": full["total_pnl_dollars"],
            "target_rate": full["target_rate_pct"],
            "stop_rate": full["stop_rate_pct"],
            "time_exit_rate": full["time_exit_rate_pct"],
            "avg_risk_dollars": full["avg_risk_dollars"],
            "avg_notional_dollars": full["avg_notional_dollars"],
            "cap_breach_rate_pct": full["cap_breach_rate_pct"],
        })

    def eligible(r):
        """A config whose target is essentially never reached is degenerate —
        the target parameter is inert, so it is not a real optimum."""
        return r["target_rate"] >= MIN_TARGET_HIT_PCT

    # ── selection on TRAIN, scored on TEST ──────────────────────────────
    print(f"\n{'='*104}\nSELECTED ON TRAIN -> SCORED ON TEST (out of sample)\n{'='*104}")
    print(f"{'hold':>5} {'tgt':>5} {'stop':>5} | {'trainS':>7} {'TEST S':>7} "
          f"{'test P&L$':>11} | {'insampleS':>9} {'n':>6} {'tgtHit%':>7} "
          f"{'stopHit%':>8} {'timeEx%':>8}")
    selected = []
    for hold in HOLD_PERIODS:
        cands = [r for r in results if r["hold"] == hold and eligible(r)]
        if not cands:
            continue
        best = max(cands, key=lambda r: r["train_sharpe"])
        selected.append(best)
        print(f"{best['hold']:>5} {best['target_mult']:>5} {best['stop_atr_mult']:>5} | "
              f"{best['train_sharpe']:>7.2f} {best['test_sharpe']:>7.2f} "
              f"{best['test_pnl']:>11,.2f} | {best['full_sharpe']:>9.2f} "
              f"{best['n_trades']:>6} {best['target_rate']:>7.1f} "
              f"{best['stop_rate']:>8.1f} {best['time_exit_rate']:>8.1f}")

    # ── best in-sample per hold (the optimistic number) ────────────────
    print(f"\n{'='*104}\nBEST IN-SAMPLE (optimistic — NOT tradable)\n{'='*104}")
    insample = []
    for hold in HOLD_PERIODS:
        cands = [r for r in results if r["hold"] == hold]
        if not cands:
            continue
        best = max(cands, key=lambda r: r["full_sharpe"])
        insample.append(best)
        print(f"  hold={best['hold']:>2} tgt={best['target_mult']:>4} "
              f"stop={best['stop_atr_mult']:>4}  fullS={best['full_sharpe']:>6.2f}  "
              f"testS={best['test_sharpe']:>6.2f}  n={best['n_trades']}  "
              f"tgtHit%={best['target_rate']:.1f}")

    # ── the stop-width sweep, which is the real knob ───────────────────
    print(f"\n{'='*104}\nSTOP-WIDTH SWEEP (the real knob) — target fixed at 2R (live value)\n{'='*104}")
    stop_sweep = {}
    for hold in HOLD_PERIODS:
        rows = [r for r in results if r["hold"] == hold and r["target_mult"] == 2.0]
        if not rows:
            continue
        rows.sort(key=lambda r: r["stop_atr_mult"])
        stop_sweep[hold] = rows
        print(f"  hold={hold}")
        for r in rows:
            print(f"    stop={r['stop_atr_mult']:>5}  trainS={r['train_sharpe']:>6.2f} "
                  f"testS={r['test_sharpe']:>6.2f} fullS={r['full_sharpe']:>6.2f}  "
                  f"tgtHit%={r['target_rate']:>5.1f} stopHit%={r['stop_rate']:>5.1f} "
                  f"timeEx%={r['time_exit_rate']:>5.1f} avgRisk$={r['avg_risk_dollars']:>6.2f}")

    # ── robustness of the train-selected configs across periods ─────────
    print(f"\n{'='*104}\nPER-PERIOD ROBUSTNESS of train-selected configs\n{'='*104}")
    all_dates2 = sorted({t["entry_date"] for t in cache[(1, 2.0, 1.0)]})
    chunks = [list(c) for c in np.array_split(np.array(all_dates2), 5)]
    period_rows = {}
    for best in selected:
        trades = cache[(best["hold"], best["target_mult"], best["stop_atr_mult"])]
        row = {}
        for chunk in chunks:
            if not len(chunk):
                continue
            lo, hi = str(chunk[0]), str(chunk[-1])
            sub = [t for t in trades if lo <= t["entry_date"] <= hi]
            s = score(sub)
            row[f"{lo}..{hi}"] = s["sharpe_dollar"] if s else None
        period_rows[f"hold{best['hold']}_t{best['target_mult']}_s{best['stop_atr_mult']}"] = row
        vals = [v for v in row.values() if v is not None]
        npos = sum(1 for v in vals if v > 0)
        print(f"  hold={best['hold']:>2} tgt={best['target_mult']:>4} "
              f"stop={best['stop_atr_mult']:>4}  "
              + "  ".join(f"{v:>6.2f}" for v in vals)
              + f"   ({npos}/{len(vals)} periods positive)")

    base = next((r for r in results if r["hold"] == 1 and r["target_mult"] == 2.0
                 and r["stop_atr_mult"] == 1.0), None)
    if base:
        print(f"\n  BASELINE (live: hold=1, tgt=2.0, stop=1.0): "
              f"trainS={base['train_sharpe']:.2f} testS={base['test_sharpe']:.2f} "
              f"fullS={base['full_sharpe']:.2f}")

    with open(OUT_JSON, "w") as fh:
        json.dump({"results": results, "selected": selected,
                   "insample_best": insample, "period_robustness": period_rows,
                   "stop_sweep": stop_sweep, "train_end": split},
                  fh, indent=2, default=str)
    print(f"\nSaved → {OUT_JSON}")
    _write_md(results, selected, insample, period_rows, stop_sweep, split)


def _write_md(results, selected, insample, period_rows, stop_sweep, split):
    lines = [
        "# Engulfing — target/stop reoptimisation",
        "",
        f"Live path, live top-20 universe, ${CAPITAL:,.0f} capital, whole",
        f"shares, {COST_BPS:g}bps round-trip cost.",
        "",
        f"Parameters are selected on a TRAIN window (entries before {split})",
        "and scored on a TEST window (entries from that date on). The",
        "in-sample column is the optimistic number — do not trade on it.",
        "",
        f"A config whose target is reached in under {MIN_TARGET_HIT_PCT:g}% of",
        "trades is excluded from selection: once the target is never hit,",
        "the target parameter is inert and the 'optimum' is just noise.",
        "",
        "## Selected on train → scored out of sample",
        "",
        "| Hold | Target (R) | Stop (ATR) | Train Sharpe$ | **TEST Sharpe$** | "
        "Test P&L $ | In-sample Sharpe$ | N trades | Target% | Stop% | Time-exit% |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in selected:
        lines.append(
            f"| {b['hold']} | {b['target_mult']} | {b['stop_atr_mult']} | "
            f"{b['train_sharpe']:.2f} | **{b['test_sharpe']:.2f}** | "
            f"{b['test_pnl']:,.2f} | {b['full_sharpe']:.2f} | {b['n_trades']} | "
            f"{b['target_rate']:.0f}% | {b['stop_rate']:.0f}% | {b['time_exit_rate']:.0f}% |"
        )

    lines += [
        "",
        "## The real knob — stop width (target fixed at 2R, the live value)",
        "",
        "| Hold | Stop (ATR) | Train Sharpe$ | Test Sharpe$ | In-sample Sharpe$ | "
        "Target% | Stop% | Time-exit% | Avg risk $ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hold, rows in stop_sweep.items():
        for r in rows:
            lines.append(
                f"| {hold} | {r['stop_atr_mult']} | {r['train_sharpe']:.2f} | "
                f"{r['test_sharpe']:.2f} | {r['full_sharpe']:.2f} | "
                f"{r['target_rate']:.0f}% | {r['stop_rate']:.0f}% | "
                f"{r['time_exit_rate']:.0f}% | {r['avg_risk_dollars']:.2f} |"
            )

    lines += [
        "",
        "## Best in-sample (optimistic, not tradable)",
        "",
        "| Hold | Target (R) | Stop (ATR) | In-sample Sharpe$ | Test Sharpe$ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for b in insample:
        lines.append(f"| {b['hold']} | {b['target_mult']} | {b['stop_atr_mult']} | "
                     f"{b['full_sharpe']:.2f} | {b['test_sharpe']:.2f} |")

    first = next(iter(period_rows))
    cols = sorted(period_rows[first])
    lines += [
        "",
        "## Per-period robustness of the train-selected configs",
        "",
        "Sharpe$ per calendar sub-period. A config that only works in one",
        "regime is not an edge.",
        "",
        "| Config | " + " | ".join(cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for cfg, row in period_rows.items():
        cells = " | ".join(f"{row[k]:.2f}" if row[k] is not None else "n/a"
                           for k in cols)
        lines.append(f"| {cfg} | {cells} |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Summary saved → {OUT_MD}")


if __name__ == "__main__":
    main()
