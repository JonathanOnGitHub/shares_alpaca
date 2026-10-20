"""Write engulfing results to report files."""
import sys
sys.path.insert(0, "src")

import pickle
from trading.engulfing import run_all_variants, save_results

with open("data/engulfing_universe.pkl", "rb") as f:
    store = pickle.load(f)

results = run_all_variants(
    store,
    list(store.keys()),
    start_date="2015-01-01",
    end_date="2025-12-31",
)

save_results(results, "reports/engulfing_results.json")
print("JSON saved")

# Markdown summary
lines = [
    "# Engulfing Bar — Backtest Summary\n",
    f"**Universe:** {len(store)} US large-caps  (2015-01-01 – 2025-12-31)\n\n",
    "| Variant | N | Ann Ret% | Sharpe | Max DD% | WR% | Avg Bps | Stop% | Target% |\n",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
]

variants = [(k, v) for k, v in results.items() if k != "_meta"]
variants.sort(key=lambda x: x[1].get("ann_ret", float("-inf")), reverse=True)

for name, r in variants:
    if r.get("n", 0) == 0:
        continue
    ann = r.get("ann_ret", 0.0) * 100
    shp = r.get("sharpe", 0.0)
    dd  = r.get("max_dd", 0.0)
    wr  = r.get("win_rate", 0.0)
    bps = r.get("avg_bps", 0.0)
    stp = r.get("stop_rate", 0.0)
    tgt = r.get("target_rate", 0.0)
    n   = r.get("n", 0)

    ann_str = f"{ann:.1f}" if ann == ann else "NaN"  # NaN check
    lines.append(
        f"| `{name}` | {n} | {ann_str} | {shp:.2f} | {dd:.1f} | "
        f"{wr:.1f} | {bps:.1f} | {stp:.1f} | {tgt:.1f} |\n"
    )

with open("reports/engulfing_summary.md", "w") as f:
    f.writelines(lines)

print("Markdown saved")
print(f"\nTop 5 by Ann Ret:")
for name, r in variants[:5]:
    if r.get("n", 0) == 0:
        continue
    ann = r.get("ann_ret", 0.0) * 100
    shp = r.get("sharpe", 0.0)
    n   = r.get("n", 0)
    print(f"  {name:<38}  ann={ann:7.1f}%  sharpe={shp:.2f}  n={n}")
