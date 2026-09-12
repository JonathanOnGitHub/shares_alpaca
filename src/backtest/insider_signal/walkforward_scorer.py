"""
Walk-forward scorer for CEO purchase signals.

At each rebalance date:
  1. Aggregate trailing-12-month purchases per ticker (strict lookback — no future leakage)
  2. Score by composite = conviction × recency × size
  3. Quintile-rank all eligible tickers

Usage:
    python walkforward_scorer.py
"""

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
RAW_CSV    = Path(__file__).parent / "ceo_purchases_edgar_5yr.csv"
OUT_CSV    = Path(__file__).parent / "signal_history.csv"
REBALANCES = [
    "2021-10-01",   # Q4 2021
    "2022-01-01",   # Q1 2022
    "2022-04-01",   # Q2 2022
    "2022-07-01",   # Q3 2022
    "2022-10-01",   # Q4 2022
    "2023-01-01",   # Q1 2023
    "2023-04-01",   # Q2 2023
    "2023-07-01",   # Q3 2023
    "2023-10-01",   # Q4 2023
    "2024-01-01",   # Q1 2024
    "2024-04-01",   # Q2 2024
    "2024-07-01",   # Q3 2024
    "2024-10-01",   # Q4 2024
    "2025-01-01",   # Q1 2025
    "2025-04-01",   # Q2 2025
    "2025-07-01",   # Q3 2025
    "2025-10-01",   # Q4 2025
    "2026-01-01",   # Q1 2026
    "2026-04-01",   # Q2 2026
    "2026-07-01",   # Q3 2026 (live paper)
]
LOOKBACK_DAYS = 365       # trailing window for purchase aggregation
MIN_PURCHASES  = 1        # minimum CEO/CFO purchases to qualify
POSITION_FILTER = "Chief Executive Officer"   # only CEOs count as signal (no Directors/CFOs)
MIN_VALUE_USD  = 100_000   # minimum purchase value to count (filters noise)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("scorer")


def load_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["transaction_date"])
    df = df.sort_values("transaction_date")
    log.info("Loaded %d transactions from %s", len(df), path)
    return df


def extract_owner_type(owner_name: str) -> str:
    """Pull the role from 'NAME (Role)' string."""
    if "(" in owner_name:
        return owner_name.split("(")[-1].rstrip(")")
    return ""


def score_tickers_at_date(txns: pd.DataFrame, score_date: datetime,
                           lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Aggregate and score all tickers as of score_date using only
    transactions strictly before score_date.

    Returns DataFrame with columns:
      ticker, n_events, total_value, avg_price, latest_filing,
      recency_score, conviction_score, size_score, composite_score
    """
    cutoff = score_date - timedelta(days=lookback_days)
    window = txns[(txns["transaction_date"] >= cutoff) &
                  (txns["transaction_date"] < score_date)]

    if window.empty:
        log.warning("No transactions in window ending %s", score_date.date())
        return pd.DataFrame()

    # Owner-type filter: C-suite only (CEO/CFO/COO/President/GC/CTO)
    ceo_only = window[
        window["owner_role"].isin({
            "Chief Executive Officer",
            "Chief Financial Officer",
            "Chief Operating Officer",
            "President",
            "General Counsel",
            "Chief Technology Officer",
        }) &
        (window["total_value"] >= MIN_VALUE_USD)
    ]

    results = []
    all_tickers = window["ticker"].unique()

    for ticker in all_tickers:
        ticker_txns  = window[window["ticker"] == ticker]
        ticker_ceo   = ceo_only[ceo_only["ticker"] == ticker]

        n_events    = len(ticker_txns)
        n_ceo       = len(ticker_ceo)
        total_value = ticker_txns["total_value"].sum()
        latest_txn  = ticker_txns["transaction_date"].max()

        if n_ceo < MIN_PURCHASES:
            continue

        # ── Scores ──────────────────────────────────────────────────────────
        # Conviction: fraction of transactions that are CEO/CFO vs all
        conviction  = n_ceo / n_events if n_events > 0 else 0

        # Recency: days since latest transaction (0 = today, 1 = oldest)
        days_ago    = (score_date - latest_txn).days
        recency     = max(0.0, 1.0 - days_ago / lookback_days)

        # Size: log-scale purchase value (normalised per ticker)
        size        = np.log1p(total_value) / 20  # /20 keeps it in [0,1] range

        composite   = conviction * 0.5 + recency * 0.3 + size * 0.2

        results.append({
            "ticker":           ticker,
            "rebalance_date":   score_date.strftime("%Y-%m-%d"),
            "n_events":         n_events,
            "n_ceo_events":      n_ceo,
            "total_value_usd":  total_value,
            "avg_price_paid":   ticker_txns["total_value"].sum() / ticker_txns["shares"].sum()
                                 if ticker_txns["shares"].sum() > 0 else 0,
            "latest_filing":    latest_txn.strftime("%Y-%m-%d"),
            "recency_score":    round(recency, 4),
            "conviction_score": round(conviction, 4),
            "size_score":       round(size, 4),
            "composite_score":  round(composite, 4),
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("composite_score", ascending=False)

    # Quintile assignment
    n = len(df)
    df["quintile"] = pd.qcut(df["composite_score"], q=5,
                              labels=[5, 4, 3, 2, 1]).astype(int)

    log.info("  Scored %d tickers at %s — Q5(top)=%d, Q1(bot)=%d",
             n, score_date.date(),
             (df["quintile"] == 5).sum(),
             (df["quintile"] == 1).sum())
    return df


def main():
    txns = load_transactions(RAW_CSV)

    all_scores = []
    for rd in REBALANCES:
        score_date = pd.to_datetime(rd)
        scored = score_tickers_at_date(txns, score_date)
        if not scored.empty:
            all_scores.append(scored)

    if not all_scores:
        log.error("No scoring periods produced results.")
        return

    out = pd.concat(all_scores, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    log.info("Wrote %d scored ticker-period rows to %s", len(out), OUT_CSV)

    # Summary table per rebalance
    print("\n" + "=" * 70)
    print(" WALKFORD SCORER — SUMMARY")
    print("=" * 70)
    for rd in REBALANCES:
        period = out[out["rebalance_date"] == rd]
        if period.empty:
            print(f"\n  {rd}  →  no eligible tickers")
            continue
        print(f"\n  {rd}  ({len(period)} tickers)")
        print(f"    Q5 (top 20%)  : {period[period['quintile']==5]['ticker'].tolist()}")
        print(f"    Q4            : {period[period['quintile']==4]['ticker'].tolist()}")
        print(f"    Q3            : {period[period['quintile']==3]['ticker'].tolist()}")
        print(f"    Q2            : {period[period['quintile']==2]['ticker'].tolist()}")
        print(f"    Q1 (bot 20%)  : {period[period['quintile']==1]['ticker'].tolist()}")


if __name__ == "__main__":
    main()
