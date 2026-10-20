#!/usr/bin/env python3
"""
run_engulfing.py
================
Download (or load from cache) daily OHLCV for a liquid US large-cap universe,
run the engulfing bar backtest across all variants, print results and save JSON.
"""

import sys
import os
import pickle
import json
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent.parent          # Shares_alpaca/
CACHE  = BASE / "data" / "engulfing_universe.pkl"
OUTDIR = BASE / "reports"
OUTDIR.mkdir(exist_ok=True)

# ── universe (60 liquid large-caps, ~10yr history) ─────────────────────────────
UNIVERSE = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","JPM","V","JNJ",
    "WMT","PG","XOM","UNH","HD","CVX","MA","BAC","DIS","COST",
    "ABBV","PFE","TMO","NKE","ADBE","CMCSA","KO","PEP","ABT","CRM",
    "CSCO","ACN","MCD","WFC","NFLX","DHR","LLY","TXN","NEE","PM",
    "T","VZ","INTC","AMD","QCOM","BMY","UNP","LOW","UPS","MS",
    "GS","BLK","SCHW","AXP","C","USB","PNC","TFC","AMGN","GILD",
]

START  = "2015-01-01"
END    = "2025-12-31"


# ── data fetch ─────────────────────────────────────────────────────────────────
def load_or_fetch() -> dict:
    if CACHE.exists():
        print(f"[cache hit] {CACHE}")
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)

    print(f"[cache miss] downloading {len(UNIVERSE)} tickers …")
    import yfinance as yf

    raw = yf.download(
        UNIVERSE,
        start=START,
        end=END,
        auto_adjust=True,
        progress=True,
        group_by=None,
    )

    # raw.columns is MultiIndex (ticker, field) from yfinance with group_by=None
    # Extract per-ticker DataFrames directly from the MultiIndex
    store = {}
    tickers_found = {c[0] for c in raw.columns if isinstance(c, tuple)}
    for ticker in UNIVERSE:
        if ticker not in tickers_found:
            print(f"  ✗ {ticker}: not in download — skipped")
            continue
        tdf = raw.xs(ticker, level=0, axis=1).dropna()
        if len(tdf) > 500:
            store[ticker] = tdf
            print(f"  ✓ {ticker}: {len(tdf)} rows")
        else:
            print(f"  ✗ {ticker}: only {len(tdf)} rows — skipped")

    CACHE.parent.mkdir(exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(store, fh)
    print(f"\n[done] {len(store)} tickers cached → {CACHE}")
    return store


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    sys.path.insert(0, str(BASE / "src"))
    from trading.engulfing import run_all_variants, print_results, save_results

    price_df = load_or_fetch()
    tickers   = list(price_df.keys())

    print(f"\nRunning engulfing backtest …")
    print(f"  tickers : {len(tickers)}")
    print(f"  start   : {START}")
    print(f"  end     : {END}")

    results = run_all_variants(
        price_df,
        tickers,
        start_date   = START,
        end_date     = END,
        min_body_ratio = 0.0,    # no body-ratio filter for initial run
    )

    print_results(results)

    # Save
    out_path = OUTDIR / "engulfing_results.json"
    save_results(results, str(out_path))
    print(f"\nResults saved → {out_path}")

    # Also save a tidy summary table as markdown
    md_path = OUTDIR / "engulfing_summary.md"
    _write_markdown(results, md_path)
    print(f"Summary saved → {md_path}")


def _write_markdown(results: dict, path: Path) -> None:
    variants = [(k, v) for k, v in results.items() if k != "_meta"]
    variants.sort(key=lambda x: x[1].get("sharpe", -999), reverse=True)

    lines = [
        "# Engulfing Bar — Backtest Summary\n",
        f"**Universe:** {len(results['_meta']['tickers'])} US large-caps  "
        f"({results['_meta']['start']} – {results['_meta']['end']})\n",
        "| Variant | N | Total Ret% | Ann Ret% | Sharpe | Max DD% | "
        "Win Rate% | Avg Bps | Stop% | Target% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, r in variants:
        n = r.get("n", 0)
        if n == 0:
            continue
        lines.append(
            f"| `{name}` | {n} | "
            f"{r.get('total_ret',0)*100:.1f} | "
            f"{r.get('ann_ret',0)*100:.1f} | "
            f"{r.get('sharpe',0):.2f} | "
            f"{r.get('max_dd',0):.1f} | "
            f"{r.get('win_rate',0):.1f} | "
            f"{r.get('avg_bps',0):.1f} | "
            f"{r.get('stop_rate',0):.1f} | "
            f"{r.get('target_rate',0):.1f} |"
        )

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
