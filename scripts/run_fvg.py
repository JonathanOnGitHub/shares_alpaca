"""
FVG Strategy — Runner
Loads cached universe, runs all variants, saves results + markdown summary.
"""

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading.fvg import run_all_variants, print_results, save_results

CACHE   = ROOT / "data" / "engulfing_universe.pkl"
OUT_JSON = ROOT / "reports" / "fvg_results.json"
OUT_MD   = ROOT / "reports" / "fvg_summary.md"

START   = "2015-01-01"
END     = "2025-12-31"
COSTS   = [0, 10]    # no-cost baseline + realistic 10 bps

# ── Load cached universe ───────────────────────────────────────────────────────
print("Loading cached universe …")
with open(CACHE, "rb") as f:
    price_df = pickle.load(f)
tickers = list(price_df.keys())
print(f"  {len(tickers)} tickers, "
      f"{next(iter(price_df.values())).index[0].date()} → "
      f"{next(iter(price_df.values())).index[-1].date()}")


# ── Run ────────────────────────────────────────────────────────────────────────
for cost in COSTS:
    label = "no-cost" if cost == 0 else "10bps"
    print(f"\n{'═'*72}")
    print(f"  FVG  |  {START} – {END}  |  cost={cost} bps  [{label}]")
    print(f"{'═'*72}")

    results = run_all_variants(
        price_df, tickers,
        start_date=START, end_date=END,
        min_gap_bps=0.0,
        cost_bps=cost,
    )
    print_results(results)

    # Save JSON
    out_path = OUT_JSON.parent / f"fvg_results_{cost}bps.json"
    save_results(results, str(out_path))
    print(f"JSON → {out_path}")

    # Write markdown comparison table
    _write_md(results, OUT_MD, cost)
    print(f"Markdown → {OUT_MD}")


# ── Markdown writer ────────────────────────────────────────────────────────────
def _write_md(results: dict, path: Path, cost: float) -> None:
    lines = [
        f"# FVG Strategy — Backtest Results  ({cost} bps)",
        "",
        f"**Universe:** 60 US large-caps  |  **Period:** 2015–2025",
        "",
        "| Variant | N | Ann% | Sharpe | MaxDD% | WR% | AvgBps | Stop% | Tgt% |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    variants = [(k, v) for k, v in results.items() if k != "_meta"]
    variants.sort(key=lambda x: x[1].get("sharpe", -999), reverse=True)

    for name, r in variants:
        n = r.get("n", 0)
        if n == 0:
            continue
        lines.append(
            f"| {name} | {n} | "
            f"{r.get('ann_ret',0)*100:.1f} | "
            f"{r.get('sharpe',0):.2f} | "
            f"{r.get('max_dd',0):.1f} | "
            f"{r.get('win_rate',0):.1f} | "
            f"{r.get('avg_bps',0):.1f} | "
            f"{r.get('stop_rate',0):.1f} | "
            f"{r.get('target_rate',0):.1f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
