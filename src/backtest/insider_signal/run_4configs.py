"""
Run 4 scoring configs and write 4 signal_history files.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import logging

RAW_CSV = Path("ceo_purchases_edgar_5yr.csv")
log = logging.getLogger("scorer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

txns = pd.read_csv(RAW_CSV, parse_dates=["transaction_date"]).sort_values("transaction_date")
log.info("Loaded %d transactions", len(txns))

REBALANCES = [
    "2021-10-01","2022-01-01","2022-04-01","2022-07-01","2022-10-01",
    "2023-01-01","2023-04-01","2023-07-01","2023-10-01","2024-01-01",
    "2024-04-01","2024-07-01","2024-10-01","2025-01-01","2025-04-01",
    "2025-07-01","2025-10-01","2026-01-01","2026-04-01","2026-07-01",
]
MIN_PURCHASES = 1
MIN_VALUE_USD = 100_000
QUALIFYING_ROLES = {
    "Chief Executive Officer","Chief Financial Officer","Chief Operating Officer",
    "President","General Counsel","Chief Technology Officer",
}

def score_tickers(txns_df, score_date, lookback_days, w_size, w_recency, w_conviction):
    cutoff = score_date - timedelta(days=lookback_days)
    window = txns_df[
        (txns_df["transaction_date"] >= cutoff) &
        (txns_df["transaction_date"] < score_date)
    ]
    if window.empty:
        return pd.DataFrame()

    ceo_only = window[
        window["owner_role"].isin(QUALIFYING_ROLES) &
        (window["total_value"] >= MIN_VALUE_USD)
    ]

    results = []
    for ticker in window["ticker"].unique():
        t_txns = window[window["ticker"] == ticker]
        t_ceo  = ceo_only[ceo_only["ticker"] == ticker]
        n_events = len(t_txns)
        n_ceo    = len(t_ceo)
        total_val = t_txns["total_value"].sum()
        latest_txn = t_txns["transaction_date"].max()

        if n_ceo < MIN_PURCHASES:
            continue

        conviction = n_ceo / n_events if n_events > 0 else 0
        days_ago   = (score_date - latest_txn).days
        recency    = max(0.0, 1.0 - days_ago / lookback_days)
        size       = np.log1p(total_val) / 20
        composite  = conviction * w_conviction + recency * w_recency + size * w_size

        results.append({
            "ticker": ticker,
            "rebalance_date": score_date.strftime("%Y-%m-%d"),
            "n_events": n_events,
            "total_value_usd": total_val,
            "recency_score": round(recency, 4),
            "conviction_score": round(conviction, 4),
            "size_score": round(size, 4),
            "composite_score": round(composite, 4),
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("composite_score", ascending=False)
    n = len(df)
    df["quintile"] = pd.qcut(df["composite_score"], q=5, labels=[5,4,3,2,1]).astype(int)
    return df

def run_backtest(label, lookback_days, w_size, w_recency, w_conviction):
    all_scores = []
    for rd in REBALANCES:
        sd = pd.to_datetime(rd)
        scored = score_tickers(txns, sd, lookback_days, w_size, w_recency, w_conviction)
        if not scored.empty:
            all_scores.append(scored)

    out = pd.concat(all_scores, ignore_index=True)
    out.to_csv(f"signal_history_{label}.csv", index=False)
    log.info("[%s] Wrote %d rows", label, len(out))
    return out

configs = [
    ("A_baseline",  365, 0.2, 0.3, 0.5),
    ("B_valuewt",   365, 0.7, 0.1, 0.2),
    ("C_1qtr",       90, 0.2, 0.3, 0.5),
    ("D_both",       90, 0.7, 0.1, 0.2),
]

for label, lb, ws, wr, wc in configs:
    run_backtest(label, lb, ws, wr, wc)

print("\n All 4 signal_history files written.")
