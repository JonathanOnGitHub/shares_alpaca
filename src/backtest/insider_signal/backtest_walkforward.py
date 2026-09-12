"""
Walk-forward backtest for CEO purchase signals.

At each quarterly rebalance:
  1. Load scores from walkforward_scorer output
  2. Fetch price history for the hold window
  3. Long top quintile (Q5), benchmark = SPY buy-and-hold
  4. Compute forward returns per ticker

Usage:
    python backtest_walkforward.py
"""

import sys
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

# ── Config ─────────────────────────────────────────────────────────────────────
SIGNAL_CSV    = Path(__file__).parent / "signal_history.csv"
RAW_TXN_CSV   = Path(__file__).parent / "ceo_purchases_raw.csv"
OUT_RESULTS   = Path(__file__).parent / "backtest_results_wf.csv"
OUT_SUMMARY   = Path(__file__).parent / "backtest_summary.txt"
HOLD_DAYS     = 63          # ~1 quarter (trading days)
INITIAL_CAPITAL = 100_000.0
RF_RATE       = 0.05        # risk-free rate for Sharpe

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("backtest_wf")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_signals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["rebalance_date", "latest_filing"])
    df = df.sort_values(["rebalance_date", "composite_score"], ascending=[True, False])
    log.info("Loaded %d signal rows from %s", len(df), path)
    return df


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close for all tickers + SPY."""
    tickers_str = " ".join(tickers + ["SPY"])
    log.info("Fetching prices for %d tickers …", len(tickers))
    data = yf.download(
        tickers_str, start=start, end=end,
        progress=False, auto_adjust=True,
    )
    if "Close" in data.columns.get_level_values(0):
        prices = data["Close"]
    else:
        prices = data
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices.droplevel(1, axis=1)
    return prices.ffill().dropna(how="all")


def compute_period_return(prices_df: pd.DataFrame, tickers: list[str],
                          start_idx: int, n_days: int) -> dict:
    """
    Compute equal-weight forward return for a basket of tickers
    over n_days starting at start_idx (index into prices_df).
    Returns per-ticker returns + basket mean.
    """
    end_idx = min(start_idx + n_days, len(prices_df))
    window  = prices_df.iloc[start_idx:end_idx]
    if window.empty or len(window) < 2:
        return {}

    results = {}
    for t in tickers:
        if t not in window.columns:
            continue
        series = window[t].dropna()
        if len(series) < 2:
            continue
        ret = (series.iloc[-1] / series.iloc[0]) - 1
        results[t] = ret

    return results


def period_metrics(pv_series: pd.Series, rf_rate: float = RF_RATE) -> dict:
    """Compute summary stats from a daily portfolio-value series."""
    if pv_series.empty or len(pv_series) < 2:
        return {}
    rets = pv_series.pct_change().dropna()
    total_ret = (pv_series.iloc[-1] / pv_series.iloc[0]) - 1
    ann_ret   = (1 + total_ret) ** (252 / len(rets)) - 1 if len(rets) > 0 else 0
    ann_vol   = rets.std() * np.sqrt(252) if len(rets) > 1 else 0
    sharpe    = float((ann_ret - rf_rate) / ann_vol) if ann_vol > 0 else 0.0
    cummax    = pv_series.cummax()
    drawdown  = (pv_series - cummax) / cummax
    return {
        "total_ret":    round(total_ret * 100, 2),
        "ann_ret":      round(ann_ret * 100, 2),
        "ann_vol":      round(ann_vol * 100, 2),
        "sharpe":       round(sharpe, 3),
        "max_dd":       round(drawdown.min() * 100, 2),
        "hit_rate":     round((rets > 0).mean() * 100, 1),
        "final_value":  round(pv_series.iloc[-1], 2),
        "n_days":        len(rets),
    }


def bootstrap_ci(returns: list[float], n_bootstrap: int = 9999,
                  ci: float = 0.95) -> tuple:
    """
    Non-parametric bootstrap on a list of per-ticker forward returns.
    Returns (mean, lower_ci, upper_ci).
    """
    if len(returns) < 2:
        return (np.mean(returns), np.nan, np.nan)
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(returns, size=len(returns), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(np.mean(returns)), float(lower), float(upper)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    if not SIGNAL_CSV.exists():
        log.error("signal_history.csv not found — run walkforward_scorer.py first")
        sys.exit(1)

    signals = load_signals(SIGNAL_CSV)
    rebalances = sorted(signals["rebalance_date"].unique())
    log.info("Rebalance periods: %s", [str(d.date()) for d in rebalances])

    # ── Fetch all price data in one shot ─────────────────────────────────────
    # Universe = all tickers that ever appear in signals + SPY
    all_tickers = signals["ticker"].unique().tolist()
    price_start = (signals["rebalance_date"].min() - timedelta(days=10)).strftime("%Y-%m-%d")
    price_end   = (signals["rebalance_date"].max() + timedelta(days=HOLD_DAYS + 10)).strftime("%Y-%m-%d")
    prices      = fetch_prices(all_tickers, price_start, price_end)

    if prices.empty:
        log.error("No price data fetched.")
        sys.exit(1)

    log.info("Price window: %s → %s (%d trading days)",
             price_start, price_end, len(prices))

    all_results  = []    # per-ticker per-period rows
    period_stats = []    # aggregate stats per period
    price_dates  = prices.index.normalize()

    for rebalance_date in rebalances:
        period_sigs = signals[signals["rebalance_date"] == rebalance_date].copy()
        q5 = period_sigs[period_sigs["quintile"] == 5]["ticker"].tolist()
        q1 = period_sigs[period_sigs["quintile"] == 1]["ticker"].tolist()

        log.info("\n── Period %s  Q5=%d  Q1=%d ──",
                 rebalance_date.date(), len(q5), len(q1))

        # Find entry index in price series
        valid = price_dates[price_dates >= rebalance_date]
        if valid.empty:
            log.warning("No price data after %s", rebalance_date.date())
            continue
        entry_idx = price_dates.get_loc(valid[0])

        # Compute per-ticker forward returns for Q5 and Q1
        q5_rets = compute_period_return(prices, q5, entry_idx, HOLD_DAYS)
        q1_rets = compute_period_return(prices, q1, entry_idx, HOLD_DAYS)

        if not q5_rets and not q1_rets:
            log.warning("No return data for period starting %s", rebalance_date.date())
            continue

        # SPY benchmark: buy-and-hold over same window
        spy_ret = compute_period_return(prices, ["SPY"], entry_idx, HOLD_DAYS)
        spy_ret_val = list(spy_ret.values())[0] if spy_ret else 0.0

        # Q5 basket stats
        q5_ret_list = list(q5_rets.values())
        q5_mean     = np.mean(q5_ret_list) if q5_ret_list else 0.0
        q5_median   = np.median(q5_ret_list) if q5_ret_list else 0.0
        q5_pos_rate = (np.sum(np.array(q5_ret_list) > 0) / len(q5_ret_list)) if q5_ret_list else 0.0

        # Q1 basket stats
        q1_ret_list = list(q1_rets.values())
        q1_mean     = np.mean(q1_ret_list) if q1_ret_list else 0.0

        # Quintile spread
        spread = q5_mean - q1_mean

        # Bootstrap CI on Q5 returns
        if len(q5_ret_list) >= 2:
            boot_mean, boot_lo, boot_hi = bootstrap_ci(q5_ret_list)
        else:
            boot_mean, boot_lo, boot_hi = q5_mean, np.nan, np.nan

        period_label = rebalance_date.strftime("%Y-%m-%d")
        period_stats.append({
            "period":          period_label,
            "n_q5":            len(q5_rets),
            "n_q1":            len(q1_rets),
            "q5_mean_ret":     round(q5_mean * 100, 2),
            "q5_median_ret":   round(q5_median * 100, 2),
            "q5_pos_rate":     round(q5_pos_rate * 100, 1),
            "q1_mean_ret":     round(q1_mean * 100, 2),
            "q5_q1_spread":    round(spread * 100, 2),
            "spy_ret":         round(spy_ret_val * 100, 2),
            "alpha_vs_spy":    round((q5_mean - spy_ret_val) * 100, 2),
            "boot_mean_pct":   round(boot_mean * 100, 2),
            "boot_ci_lo":      round(boot_lo * 100, 2),
            "boot_ci_hi":      round(boot_hi * 100, 2),
        })

        # Per-ticker rows for export
        for t, r in q5_rets.items():
            all_results.append({"period": period_label, "quintile": "Q5", "ticker": t,
                                "forward_return": round(r * 100, 4)})
        for t, r in q1_rets.items():
            all_results.append({"period": period_label, "quintile": "Q1", "ticker": t,
                                "forward_return": round(r * 100, 4)})

        # Log
        log.info("  Q5 mean: %+.2f%%  median: %+.2f%%  pos: %.0f%%  spread(Q5-Q1): %+.2f%%",
                 q5_mean*100, q5_median*100, q5_pos_rate*100, spread*100)
        log.info("  SPY buy-hold: %+.2f%%  alpha vs SPY: %+.2f%%",
                 spy_ret_val*100, (q5_mean - spy_ret_val)*100)
        if len(q5_ret_list) >= 2:
            log.info("  Q5 bootstrap 95%% CI: [%+.2f %%,  %+.2f%%]",
                     boot_lo*100, boot_hi*100)

    if not period_stats:
        log.error("No periods computed.")
        sys.exit(1)

    # ── Aggregate across all periods ──────────────────────────────────────────
    stats_df = pd.DataFrame(period_stats)

    # Portfolio equity curve: equal-weight Q5 rebalanced each period
    portfolio_values = []
    capital = INITIAL_CAPITAL
    for s in period_stats:
        q5_ret = s["q5_mean_ret"] / 100.0
        capital = capital * (1 + q5_ret)
        portfolio_values.append({
            "period":     s["period"],
            "portfolio":  round(capital, 2),
            "q5_ret":     s["q5_mean_ret"],
        })

    pv_df = pd.DataFrame(portfolio_values)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(" CEO-PURCHASE SIGNAL — WALK-FORWARD BACKTEST RESULTS")
    print("=" * 72)
    print(f"  Rebalance periods : {[s['period'] for s in period_stats]}")
    print(f"  Hold window      : {HOLD_DAYS} trading days (~1 quarter)")
    print(f"  Initial capital  : ${INITIAL_CAPITAL:,.0f}")
    print(f"  Risk-free rate   : {RF_RATE*100:.1f}%")
    print("=" * 72)

    print(f"\n  {'Period':<12}  {'Q5(n)':>7}  {'Q5 ret':>8}  {'Q1 ret':>8}  "
          f"{'Spread':>8}  {'SPY':>8}  {'Alpha':>8}  {'Pos%':>6}")
    print("  " + "-" * 68)
    for s in period_stats:
        print(f"  {s['period']:<12}  {s['n_q5']:>5}({s['n_q1']:>2})  "
              f"{s['q5_mean_ret']:>+7.2f}%  {s['q1_mean_ret']:>+7.2f}%  "
              f"{s['q5_q1_spread']:>+7.2f}%  {s['spy_ret']:>+7.2f}%  "
              f"{s['alpha_vs_spy']:>+7.2f}%  {s['q5_pos_rate']:>5.0f}%")

    print("  " + "-" * 68)
    # Cross-period means
    avg_q5    = stats_df["q5_mean_ret"].mean()
    avg_q1    = stats_df["q1_mean_ret"].mean()
    avg_spread= stats_df["q5_q1_spread"].mean()
    avg_spy   = stats_df["spy_ret"].mean()
    avg_alpha = stats_df["alpha_vs_spy"].mean()
    print(f"  {'AVG (n=%d)' % len(period_stats):<12}  {'':<7}  "
          f"{avg_q5:>+7.2f}%  {avg_q1:>+7.2f}%  {avg_spread:>+7.2f}%  "
          f"{avg_spy:>+7.2f}%  {avg_alpha:>+7.2f}%")

    print("\n" + "=" * 72)
    print(" BOOTSTRAP 95%% CI ON Q5 RETURNS (per period, non-parametric n=%d)"
          % stats_df["n_q5"].iloc[0])
    print("=" * 72)
    for s in period_stats:
        lo = s["boot_ci_lo"]
        hi = s["boot_ci_hi"]
        lo_str = f"{lo:>+6.2f}%" if not np.isnan(lo) else "     N/A"
        hi_str = f"{hi:>+6.2f}%" if not np.isnan(hi) else "     N/A"
        print(f"  {s['period']:<12}  mean={s['boot_mean_pct']:>+6.2f}%  "
              f"95%% CI=[{lo_str},  {hi_str}]")

    print("\n" + "=" * 72)
    print(" PORTFOLIO EQUITY CURVE (equal-weight Q5, rebalanced quarterly)")
    print("=" * 72)
    print(f"  {'Period':<12}  {'Portfolio $':>12}  {'Q5 Return':>10}")
    print("  " + "-" * 38)
    for pv in portfolio_values:
        print(f"  {pv['period']:<12}  ${pv['portfolio']:>11,.0f}  {pv['q5_ret']:>+9.2f}%")
    print("  " + "-" * 38)
    total_return = (portfolio_values[-1]["portfolio"] / INITIAL_CAPITAL - 1) * 100
    print(f"  {'Total':<12}  ${portfolio_values[-1]['portfolio']:>11,.0f}  "
          f"{total_return:>+9.2f}%")

    # ── Honest interpretation ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(" HONEST INTERPRETATION")
    print("=" * 72)
    n_periods = len(period_stats)
    if n_periods < 3:
        print(f"\n  ⚠  Only {n_periods} rebalancing period(s) — Sharpe, MaxDD, and")
        print(f"     annualised volatility are NOT meaningful with this sample size.")
        print(f"     Treat the per-period returns as directional signal indicators only.")
    else:
        # Sharpe approximation across periods
        period_rets = [s["q5_mean_ret"] / 100 for s in period_stats]
        mean_ret    = np.mean(period_rets)
        std_ret     = np.std(period_rets, ddof=1)
        if std_ret > 0:
            ann_ret_approx = mean_ret * 4   # 4 quarters/year
            ann_vol_approx = std_ret * 2     # sqrt(4) = 2
            sharpe_approx  = (ann_ret_approx - RF_RATE) / ann_vol_approx
            print(f"\n  Approximate annualised (4-quarter annualisation):")
            print(f"    Ann. return ≈ {ann_ret_approx*100:+.2f}%")
            print(f"    Ann. vol    ≈ {ann_vol_approx*100:.2f}%")
            print(f"    Sharpe      ≈ {sharpe_approx:.2f}  (rough, N={n_periods})")
        print(f"\n  Max drawdown and full Sharpe require daily data across the")
        print(f"  full window — not available with only {n_periods} rebalance points.")

    print(f"\n  Key statistic: AVG QUINTILE SPREAD = {avg_spread*100:+.2f}% per quarter.")
    if avg_spread > 0:
        print(f"  → Signal direction is correct: top-quintile CEOs outperformed bottom.")
    else:
        print(f"  → Signal direction is WRONG: top-quintile CEOs underperformed bottom.")
    print(f"\n  Note: transaction costs, slippage, and tax NOT modelled.")
    print(f"  Realised returns will be lower by an estimated 0.1–0.3%% per trade.")

    # ── Save outputs ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_RESULTS, index=False)
    log.info("Per-ticker results saved to %s", OUT_RESULTS)

    # Save summary as text
    import io, sys
    buf = io.StringIO()
    stdout_backup = sys.stdout
    sys.stdout = buf
    print(f"WALK-FORWARD BACKTEST SUMMARY  |  {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"Periods: {[s['period'] for s in period_stats]}")
    print(f"Hold: {HOLD_DAYS} trading days | Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"AVG Q5 return: {avg_q5:+.4f}%  |  AVG Q1 return: {avg_q1:+.4f}%  |  AVG Spread: {avg_spread:+.4f}%")
    print(f"AVG SPY: {avg_spy:+.4f}%  |  AVG Alpha: {avg_alpha:+.4f}%")
    sys.stdout = stdout_backup
    with open(OUT_SUMMARY, "w") as f:
        f.write(buf.getvalue())
    log.info("Summary saved to %s", OUT_SUMMARY)


if __name__ == "__main__":
    main()
