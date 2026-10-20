"""
FVG Strategy — Walk-Forward Validation Runner

Trains min_gap_bps (0, 5, 10, 15, 20, 25 bps) on 3yr rolling windows,
tests on the following 1yr.  8 windows covering 2015-2025.
"""

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading.fvg import walk_forward_validate, print_fvg_wfo_results

CACHE     = ROOT / "data" / "engulfing_universe.pkl"
OUT_DIR   = ROOT / "reports"
GAP_CANDS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
COSTS     = [0, 10]
N_WINDOWS = 8
TRAIN_YR  = 3
TEST_YR   = 1

# ── Load cached universe ───────────────────────────────────────────────────────
print("Loading cached universe …")
with open(CACHE, "rb") as f:
    price_df = pickle.load(f)
tickers = list(price_df.keys())
first_date = next(iter(price_df.values())).index[0].date()
last_date  = next(iter(price_df.values())).index[-1].date()
print(f"  {len(tickers)} tickers  |  {first_date} → {last_date}")


for cost in COSTS:
    label = "no-cost" if cost == 0 else "10bps"
    print(f"\n{'#'*72}")
    print(f"  FVG WFO  |  cost={cost} bps  [{label}]")
    print(f"{'#'*72}")

    wfo = walk_forward_validate(
        price_df,
        tickers,
        train_years=TRAIN_YR,
        test_years=TEST_YR,
        n_windows=N_WINDOWS,
        cost_bps=cost,
        gap_candidates=GAP_CANDS,
    )

    print_fvg_wfo_results(wfo)

    # Save JSON summary (strip window-level trade arrays — keep aggregated only)
    out_path = OUT_DIR / f"fvg_wfo_{cost}bps.json"
    out_data = {
        "aggregated": wfo["aggregated"],
        "params":     wfo["params"],
        # window_results without the full variant data — just test Sharpe per window
        "window_sharpes": {
            w["window"]: {
                k: w["variants"][k]["test"]["sharpe"]
                for k in w["variants"]
            }
            for w in wfo["window_results"]
        },
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nJSON → {out_path}")

print("\nDone.")
