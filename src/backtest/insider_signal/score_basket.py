"""
Score and rank a basket of S&P 500 stocks by CEO-insider-purchase signal strength.

Signal score components (each 0–1, equal weight → max 3.0):
  1. Recency score   — how recently did the CEO buy? (exponential decay, half-life 10d)
  2. Size score      — purchase value relative to CEO wealth / annual comp
  3. Conviction score — number of separate filings in the lookback window

Output: ranked CSV + console summary of top-N ideas.

Usage:
    python score_basket.py [purchases_csv] [output_csv]
"""

import sys
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
PURCHASES_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "ceo_purchases_raw.csv"
OUTPUT_CSV    = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "ceo_basket_ranked.csv"
TOP_N         = 25
LOOKBACK_DAYS = 365
DECAY_HALFLIFE = 10   # days for recency score exponential decay
MIN_TOTAL_VALUE = 50_000  # skip very small "keep whole family" purchases
# Rough CEO median compensation for S&P 500 execs (USD) — used for size normalisation
EST_CEO_COMP   = 12_000_000   # total annual comp (cash + equity)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("score")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_purchases(path: Path) -> pd.DataFrame:
    """Load raw Form-4 purchase CSV; parse dates; filter to purchases only."""
    df = pd.read_csv(path, parse_dates=["filing_date", "transaction_date"])
    df = df[df["code"] == "P"]
    df = df[df["total_value"] >= MIN_TOTAL_VALUE]
    today = pd.Timestamp.now("UTC").normalize()
    today = today.tz_localize(None)   # keep tz-naive to match parsed CSV dates
    df = df[df["filing_date"] >= (today - timedelta(days=LOOKBACK_DAYS))]
    log.info("Loaded %d qualifying purchase rows from %s", len(df), path)
    return df


def recency_score(filing_date, ref_date=None) -> float:
    """
    Exponential decay from 1.0 (today) → 0.5 (half_life days ago) → 0.
    Uses calendar days, anchored at ref_date (defaults to today UTC).
    """
    if ref_date is None:
        ref_date = pd.Timestamp.utcnow().normalize()
    days_ago = (ref_date - pd.to_datetime(filing_date)).days
    decay = np.exp(-np.log(2) / DECAY_HALFLIFE * days_ago)
    return min(1.0, decay)


def conviction_score(ticker: str, df: pd.DataFrame) -> float:
    """
    Number of distinct purchase filings for this ticker (normalised 0–1).
    More separate purchase events = higher conviction.
    """
    n = df[df["ticker"] == ticker]["filing_date"].nunique()
    return min(1.0, n / 5)  # cap at 5 events for max score


def size_score(total_value: float, est_comp: float = EST_CEO_COMP) -> float:
    """
    Fraction of estimated annual CEO wealth/comp deployed in this purchase.
    >10% of comp = score 1.0, linear below that.
    """
    return min(1.0, total_value / (0.10 * est_comp))


def fetch_pe_ratio(ticker: str) -> float | None:
    """Fetch trailing P/E from yfinance. Returns None on failure."""
    try:
        ticker_yf = yf.Ticker(ticker)
        info = ticker_yf.info
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe and pe > 0:
            return float(pe)
    except Exception:
        pass
    return None


def fetch_market_cap(ticker: str) -> float | None:
    try:
        ticker_yf = yf.Ticker(ticker)
        mc = ticker_yf.info.get("marketCap")
        if mc:
            return float(mc)
    except Exception:
        pass
    return None


def score_ticker(group: pd.DataFrame, ref_date=None) -> dict:
    """
    Aggregate all purchase events for one ticker into a composite signal score.
    Returns a dict with scores and metadata.
    """
    ticker       = group["ticker"].iloc[0]
    n_events     = len(group)
    total_value  = group["total_value"].sum()
    latest_filing = group["filing_date"].max()
    avg_price    = group.loc[group["price"] > 0, "price"].mean() if (group["price"] > 0).any() else 0

    s_recency  = recency_score(latest_filing, ref_date)
    s_conviction = conviction_score(ticker, group)
    s_size     = size_score(total_value)

    composite  = (s_recency + s_conviction + s_size) / 3  # 0–1

    return {
        "ticker":            ticker,
        "n_events":          n_events,
        "total_value_usd":   total_value,
        "avg_price_paid":    avg_price,
        "latest_filing":     latest_filing.strftime("%Y-%m-%d"),
        "recency_score":     round(s_recency, 3),
        "conviction_score":  round(s_conviction, 3),
        "size_score":        round(s_size, 3),
        "composite_score":   round(composite, 4),
    }


def enrich_with_fundamentals(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch trailing P/E and market cap for each ticker via yfinance.
    Adds: pe_ratio, market_cap_b, is_large_cap (>$10B).
    """
    results = []
    for ticker in scored_df["ticker"]:
        try:
            ticker_yf   = yf.Ticker(ticker)
            info        = ticker_yf.info
            # yfinance v0.7+ may return scalar or Series for these fields
            def _scalar(v):
                if v is None: return None
                if isinstance(v, pd.Series): return v.iloc[0] if not v.empty else None
                try: return float(v)
                except: return None
            pe = _scalar(info.get("trailingPE")) or _scalar(info.get("forwardPE"))
            mc = _scalar(info.get("marketCap"))
            results.append({
                "ticker":        ticker,
                "pe_ratio":      round(pe, 1) if pe and pe > 0 else None,
                "market_cap_b":  round(mc / 1e9, 1) if mc else None,
                "is_large_cap":  mc > 1e10 if mc else False,
            })
        except Exception:
            results.append({"ticker": ticker, "pe_ratio": None, "market_cap_b": None, "is_large_cap": False})
        import time; time.sleep(0.05)   # polite to yfinance
    meta = pd.DataFrame(results)
    return scored_df.merge(meta, on="ticker", how="left")


def summarise_top(scored_df: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Return top-N ideas with enriched display columns."""
    cols = [
        "ticker", "composite_score",
        "recency_score", "conviction_score", "size_score",
        "n_events", "total_value_usd", "avg_price_paid", "latest_filing",
        "pe_ratio", "market_cap_b",
    ]
    available = [c for c in cols if c in scored_df.columns]
    return scored_df.sort_values("composite_score", ascending=False).head(top_n)[available]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not PURCHASES_CSV.exists():
        log.error("Purchases CSV not found: %s  Run scrape_insider_purchases.py first.", PURCHASES_CSV)
        sys.exit(1)

    purchases = load_purchases(PURCHASES_CSV)

    if purchases.empty:
        log.warning("No qualifying purchases found in lookback window.")
        # Write empty output
        scored = pd.DataFrame(columns=[
            "ticker","n_events","total_value_usd","avg_price_paid","latest_filing",
            "recency_score","conviction_score","size_score","composite_score",
            "pe_ratio","market_cap_b","is_large_cap",
        ])
        scored.to_csv(OUTPUT_CSV, index=False)
        return

    # Score each ticker
    ref_date = pd.Timestamp.now("UTC").normalize()
    ref_date = ref_date.tz_localize(None)
    scored_rows = []
    for ticker, group in purchases.groupby("ticker"):
        scored_rows.append(score_ticker(group, ref_date))

    scored = pd.DataFrame(scored_rows)

    # Enrich with fundamentals
    log.info("Fetching fundamentals for %d tickers …", len(scored))
    scored = enrich_with_fundamentals(scored)
    scored = scored.sort_values("composite_score", ascending=False).reset_index(drop=True)

    scored.to_csv(OUTPUT_CSV, index=False)
    log.info("Wrote scored basket to %s", OUTPUT_CSV)

    # Print top-N summary
    top = summarise_top(scored, TOP_N)
    print("\n" + "═" * 90)
    print(f" TOP {len(top)} CEO-PURCHASE SIGNAL BASKET  (lookback {LOOKBACK_DAYS}d, min ${MIN_TOTAL_VALUE:,.0f})")
    print("═" * 90)
    print(f"{'Rank':<5} {'Ticker':<8} {'Score':<7} {'Recency':<9} {'Convict.':<9} {'Size':<7} "
          f"{'Events':<7} {'Total USD':<14} {'P/E':<7} {'Mkt Cap $B':<11}")
    print("─" * 90)
    for i, row in top.iterrows():
        pe_str  = f"{row['pe_ratio']:.1f}" if row.get('pe_ratio') else "N/A"
        mc_str   = f"{row['market_cap_b']:.1f}" if row.get('market_cap_b') else "N/A"
        print(f"{i+1:<5} {row['ticker']:<8} {row['composite_score']:<7.4f} "
              f"{row['recency_score']:<9.3f} {row['conviction_score']:<9.3f} "
              f"{row['size_score']:<7.3f} {row['n_events']:<7} "
              f"${row['total_value_usd']:>12,.0f}   {pe_str:<7} {mc_str:<11}")
    print("─" * 90)
    print(f"\nFull ranked list → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
