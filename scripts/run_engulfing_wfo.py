"""
Engulfing Bar — Cost Sensitivity + Walk-Forward Validation

Runs:
  1. Fixed-split WFO: train 2015-01-01→2020-12-31, test 2021-01-01→2025-12-31
  2. Cost sensitivity  : 0, 5, 10, 15, 20, 25 bps round-trip
  3. Rolling WFO       : 3yr train / 1yr test, stepping 1yr at a time
"""

import json
import pickle
import sys
import textwrap
from pathlib import Path

# ── ensure src/ on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading.engulfing import walk_forward_validate, print_wfo_results

CACHE = ROOT / "data" / "engulfing_universe.pkl"
OUT_JSON  = ROOT / "reports" / "engulfing_wfo_results.json"
OUT_MD    = ROOT / "reports" / "engulfing_wfo_summary.md"

# ── Load cached universe ───────────────────────────────────────────────────────
print("Loading cached universe …")
with open(CACHE, "rb") as f:
    price_df = pickle.load(f)
tickers = list(price_df.keys())
print(f"  {len(tickers)} tickers, {next(iter(price_df.values())).index[0].date()} → {next(iter(price_df.values())).index[-1].date()}")


# ════════════════════════════════════════════════════════════════════════════════
# 1.  FIXED-SPLIT WFO — cost sensitivity
# ════════════════════════════════════════════════════════════════════════════════

COSTS = [0, 5, 10, 15, 20, 25]          # bps round-trip

print(f"\n{'═'*72}")
print("FIXED-SPLIT WFO  |  train: 2015 → 2020  |  test: 2021 → 2025")
print(f"{'═'*72}")

fixed_split: dict = {}
for cost in COSTS:
    print(f"\n── cost = {cost} bps ──")
    wfo = walk_forward_validate(
        price_df, tickers,
        train_start="2015-01-01", train_end="2020-12-31",
        test_start="2021-01-01",  test_end="2025-12-31",
        cost_bps=cost,
    )
    print_wfo_results(wfo)
    fixed_split[cost] = wfo

    # save per-cost so we don't lose partial results if something breaks later
    _snap = json.dumps(wfo, indent=2, default=str)
    # (snapshot path not saved — re-run is cheap)


# ════════════════════════════════════════════════════════════════════════════════
# 2.  ROLLING WFO  (3yr train / 1yr test, stepping 1yr)
# ════════════════════════════════════════════════════════════════════════════════

ROLL_WINDOWS = [
    (("2015-01-01", "2017-12-31"), ("2018-01-01", "2018-12-31")),
    (("2016-01-01", "2018-12-31"), ("2019-01-01", "2019-12-31")),
    (("2017-01-01", "2019-12-31"), ("2020-01-01", "2020-12-31")),
    (("2018-01-01", "2020-12-31"), ("2021-01-01", "2021-12-31")),
    (("2019-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31")),
    (("2020-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31")),
    (("2021-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")),
    (("2022-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31")),
]

ROLL_COST = 10          # use 10 bps as representative middle ground

print(f"\n{'═'*72}")
print(f"ROLLING WFO  (3yr train / 1yr test, {ROLL_COST} bps)")
print(f"{'═'*72}")

rolling_oos: list = []   # per-window OOS results
for (ts, te), (vs, ve) in ROLL_WINDOWS:
    wfo = walk_forward_validate(
        price_df, tickers,
        train_start=ts, train_end=te,
        test_start=vs,  test_end=ve,
        cost_bps=ROLL_COST,
    )
    print(f"\n  train {ts}→{te}  |  test {vs}→{ve}")
    # Print only test metrics compactly
    for key, v in sorted(wfo.get("summary", {}).items()):
        te_m = v["test"]
        if te_m["n"] == 0:
            continue
        print(f"    {key:<22}  n={te_m['n']:>4}  "
              f"sharpe={te_m['sharpe']:>5.2f}  "
              f"ann%={te_m['ann_ret']:>7.1f}  "
              f"maxdd%={te_m['max_dd']:>7.1f}")
    rolling_oos.append(wfo)

# ── Average OOS across rolling windows ───────────────────────────────────────
print(f"\n{'═'*72}")
print("AVERAGE OOS ACROSS ROLLING WINDOWS (10 bps)")
print(f"{'═'*72}")

from collections import defaultdict

avg_metrics: dict = defaultdict(lambda: {"n": 0, "ann_ret": 0.0, "sharpe": 0.0,
                                          "max_dd": 0.0, "win_rate": 0.0, "avg_bps": 0.0})

for wfo in rolling_oos:
    for key, v in wfo.get("summary", {}).items():
        m = v["test"]
        if m["n"] == 0:
            continue
        avg_metrics[key]["n"]        += m["n"]
        avg_metrics[key]["ann_ret"]  += m["ann_ret"]
        avg_metrics[key]["sharpe"]  += m["sharpe"]
        avg_metrics[key]["max_dd"]  += m["max_dd"]
        avg_metrics[key]["win_rate"]+= m["win_rate"]
        avg_metrics[key]["avg_bps"] += m["avg_bps"]
        avg_metrics[key]["count"]    = avg_metrics[key].get("count", 0) + 1

n_windows = len(rolling_oos)
print(f"{'Variant':<24} {'AvgN':>7} {'AvgAnn%':>9} {'AvgShp':>8} {'AvgMaxDD%':>11} {'AvgWR%':>8} {'AvgBps':>8}")
print("─" * 78)
for key in sorted(avg_metrics.keys()):
    m = avg_metrics[key]
    cnt = m.pop("count", 1)
    print(f"{key:<24} {m['n']//cnt:>7} "
          f"{m['ann_ret']/cnt:>9.1f} {m['sharpe']/cnt:>8.2f} "
          f"{m['max_dd']/cnt:>11.1f} {m['win_rate']/cnt:>8.1f} "
          f"{m['avg_bps']/cnt:>8.1f}")


# ════════════════════════════════════════════════════════════════════════════════
# 3.  SAVE
# ════════════════════════════════════════════════════════════════════════════════

out: dict = {
    "fixed_split": {str(k): v for k, v in fixed_split.items()},
    "rolling": rolling_oos,
    "rolling_avg_metrics": {k: {kk: vv for kk, vv in v.items() if kk != "count"}
                            for k, v in avg_metrics.items()},
    "rolling_n_windows": n_windows,
    "rolling_cost_bps": ROLL_COST,
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nJSON saved → {OUT_JSON}")


# ── Markdown summary ──────────────────────────────────────────────────────────
lines = [
    "# Engulfing Bar — Cost Sensitivity & Walk-Forward Validation",
    "",
    "## Fixed-Split WFO  |  Train: 2015–2020  |  Test: 2021–2025",
    "",
    "| Cost (bps) | Variant | TLook | Train Sharpe | Test Sharpe | Train Ann% | Test Ann% | Test MaxDD% | Test WR% |",
    "|---|---|---|---|---|---|---|---|---|",
]

for cost, wfo in fixed_split.items():
    for key, v in sorted(wfo.get("summary", {}).items()):
        tr, te = v["train"], v["test"]
        lines.append(
            f"| {cost} | {key} | {v['selected_tl']} | "
            f"{tr['sharpe']:.2f} | {te['sharpe']:.2f} | "
            f"{tr['ann_ret']:.1f} | {te['ann_ret']:.1f} | "
            f"{te['max_dd']:.1f} | {te['win_rate']:.1f} |"
        )
    lines.append("")

lines += [
    "## Rolling WFO  (3yr train / 1yr test, 10 bps) — Average OOS Metrics",
    "",
    "| Variant | Avg N | Avg Ann% | Avg Sharpe | Avg MaxDD% | Avg WR% | Avg Bps |",
    "|---|---|---|---|---|---|---|---|",
]
for key in sorted(avg_metrics.keys()):
    m = avg_metrics[key]
    cnt = m.pop("count", 1)
    lines.append(
        f"| {key} | {m['n']//cnt} | "
        f"{m['ann_ret']/cnt:.1f} | {m['sharpe']/cnt:.2f} | "
        f"{m['max_dd']/cnt:.1f} | {m['win_rate']/cnt:.1f} | "
        f"{m['avg_bps']/cnt:.1f} |"
    )

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"Markdown saved → {OUT_MD}")
