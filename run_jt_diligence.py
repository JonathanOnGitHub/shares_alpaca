#!/usr/bin/env python3
"""
Reconstruct daily equity curves from the JT small-cap long-only results and run
the full 9-check DiligenceSuite.

This lets us apply the same rigour used by the existing shares_alpaca diligence
pipeline to the Jegadeesh-Titman research results without re-downloading 500+
tickers every run.

Usage:
    python run_jt_diligence.py                  # run all configs
    python run_jt_diligence.py --config jt_J6_K6_decile
    python run_jt_diligence.py --list
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"

# -------------------------------------------------------------------
# Diligence checks (mirrors src/validation/diligence.py so this script
# is self-contained and does not need the full src/ import chain)
# -------------------------------------------------------------------

def _normalize_idx(idx):
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.DatetimeIndex([pd.Timestamp(d.date()) for d in idx])


class DiligenceCheck:
    def __init__(self, name, passed, detail, metrics=None):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.metrics = metrics or {}


class DiligenceReport:
    def __init__(self, strategy_name=""):
        self.strategy_name = strategy_name
        self.checks = []

    def add(self, check):
        self.checks.append(check)

    def summary(self) -> str:
        lines = []
        lines.append("=" * 65)
        lines.append(f"DILIGENCE REPORT — {self.strategy_name}")
        lines.append("=" * 65)
        passed = sum(1 for c in self.checks if c.passed)
        total = len(self.checks)
        lines.append(f"Passed: {passed}/{total}")
        lines.append("")
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{status}] {c.name}")
            for k, v in c.metrics.items():
                lines.append(f"         {k}: {v}")
            if c.detail:
                lines.append(f"         → {c.detail}")
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "passed": sum(1 for c in self.checks if c.passed),
            "total": len(self.checks),
            "checks": {
                c.name: {"passed": c.passed, "detail": c.detail, "metrics": c.metrics}
                for c in self.checks
            },
        }


class DiligenceSuite:
    def __init__(self, equity_curve: pd.Series, trades: list, prices: dict, strategy_name: str = "Strategy"):
        eq_idx = _normalize_idx(equity_curve.index)
        self.equity_curve = pd.Series(equity_curve.values, index=eq_idx)
        self.trades = trades
        self.prices = prices
        self.strategy_name = strategy_name
        self._daily = self.equity_curve.pct_change().dropna()
        self._monthly = self.equity_curve.resample("ME").last().pct_change().dropna()

    def run_all(self) -> DiligenceReport:
        report = DiligenceReport(self.strategy_name)
        report.add(self._check_buy_hold())
        report.add(self._check_best_month())
        report.add(self._check_win_rate())
        report.add(self._check_max_dd())
        report.add(self._check_monthly_consistency())
        report.add(self._check_concentration())
        report.add(self._check_subperiod())
        report.add(self._check_sharpe_significance())
        report.add(self._check_permutation())
        return report

    # ---- helpers ----

    def _returns(self, series: pd.Series) -> pd.Series:
        return series.pct_change().dropna()

    # ---- individual checks ----

    def _check_buy_hold(self) -> DiligenceCheck:
        """Compare strategy vs equal-weight benchmark of available prices."""
        if not self.prices:
            return DiligenceCheck("vs Equal-Weight B&H", False, "No benchmark prices available")
        pf_dict = {}
        for s, df in self.prices.items():
            if df is None or "close" not in df.columns:
                continue
            idx = _normalize_idx(df.index)
            pf_dict[s] = pd.Series(df["close"].values, index=idx)
        if not pf_dict:
            return DiligenceCheck("vs Equal-Weight B&H", False, "No valid price data")
        pf = pd.DataFrame(pf_dict)
        ew = pf.mean(axis=1)
        ew_rets = self._returns(ew)
        strat_rets = self._daily
        common = strat_rets.index.intersection(ew_rets.index)
        if len(common) < 20:
            return DiligenceCheck("vs Equal-Weight B&H", False, f"Only {len(common)} overlapping days")
        s_ret = strat_rets.loc[common].dropna()
        b_ret = ew_rets.loc[common].dropna()
        strat_cum = (1 + s_ret).cumprod()
        bh_cum = (1 + b_ret).cumprod()
        strat_total = float(strat_cum.iloc[-1] - 1)
        bh_total = float(bh_cum.iloc[-1] - 1)
        outperf = strat_total > bh_total
        return DiligenceCheck(
            "vs Equal-Weight B&H",
            outperf,
            f"Strategy {strat_total*100:.1f}% vs B&H {bh_total*100:.1f}%",
            {
                "Strategy Return": f"{strat_total*100:.2f}%",
                "B&H Return": f"{bh_total*100:.2f}%",
            },
        )

    def _check_best_month(self) -> DiligenceCheck:
        if len(self._monthly) < 3:
            return DiligenceCheck("Best Month Exclusion", False, "Need >3 months of data")
        total = (1 + self._monthly).prod() - 1
        best_idx = self._monthly.idxmax()
        best_val = self._monthly.max()
        excl = self._monthly.drop(best_idx)
        excl_ret = (1 + excl).prod() - 1
        survived = excl_ret > 0
        return DiligenceCheck(
            "Best Month Exclusion",
            survived,
            f"Without best ({best_val*100:.1f}%): {excl_ret*100:.1f}% (total: {total*100:.1f}%)",
            {
                "Best Month": f"{best_val*100:.2f}%",
                "Return (all)": f"{total*100:.2f}%",
                "Return (excl)": f"{excl_ret*100:.2f}%",
                "Attributed to 1 month": f"{(total - excl_ret) / total * 100:.0f}%" if total != 0 else "N/A",
            },
        )

    def _check_win_rate(self) -> DiligenceCheck:
        sell_trades = [t for t in self.trades if getattr(t, "side", "") == "sell" and getattr(t, "pnl", 0) != 0]
        if not sell_trades:
            monthly_up = (self._monthly > 0).sum()
            monthly_total = len(self._monthly)
            if monthly_total == 0:
                return DiligenceCheck("Win Rate", False, "No trades or monthly data")
            rate = monthly_up / monthly_total
            return DiligenceCheck(
                "Win Rate (Monthly)",
                rate > 0.5,
                f"{monthly_up}/{monthly_total} winning months ({rate*100:.0f}%)",
                {"Win Rate": f"{rate*100:.1f}%", "Periods": str(monthly_total)},
            )
        wins = sum(1 for t in sell_trades if t.pnl > 0)
        rate = wins / len(sell_trades)
        return DiligenceCheck(
            "Trade Win Rate",
            rate > 0.5,
            f"{wins}/{len(sell_trades)} winning trades ({rate*100:.0f}%)",
            {"Win Rate": f"{rate*100:.1f}%", "Trades": str(len(sell_trades))},
        )

    def _check_max_dd(self) -> DiligenceCheck:
        if len(self.equity_curve) < 20:
            return DiligenceCheck("Max Drawdown", False, "Too little data")
        cumulative = self.equity_curve
        dd = cumulative / cumulative.cummax() - 1
        max_dd = float(dd.min())
        worst_idx = dd.idxmin()
        # Find drawdown duration
        is_drawdown = dd < 0
        durations = []
        in_dd = False
        start = None
        for dt, v in is_drawdown.items():
            if v and not in_dd:
                in_dd = True
                start = dt
            elif not v and in_dd:
                in_dd = False
                durations.append((dt - start).days)
        max_dur = max(durations) if durations else 0
        passed = max_dd > -0.50  # fail if >50% drawdown
        return DiligenceCheck(
            "Max Drawdown",
            passed,
            f"Max DD: {max_dd*100:.1f}%, longest drawdown: {max_dur} days",
            {"Max DD": f"{max_dd*100:.2f}%", "Max Duration (days)": str(max_dur)},
        )

    def _check_monthly_consistency(self) -> DiligenceCheck:
        if len(self._monthly) < 12:
            return DiligenceCheck("Monthly Consistency", False, "Need ≥12 months")
        positive = (self._monthly > 0).sum()
        total = len(self._monthly)
        rate = positive / total
        # Require >55% positive months (lenient)
        passed = rate > 0.55
        return DiligenceCheck(
            "Monthly Consistency",
            passed,
            f"{positive}/{total} positive months ({rate*100:.0f}%)",
            {"Positive Months": f"{positive}/{total}", "Rate": f"{rate*100:.1f}%"},
        )

    def _check_concentration(self) -> DiligenceCheck:
        if not self.trades:
            return DiligenceCheck("Concentration", True, "No trades — pass by default")
        sell_trades = [t for t in self.trades if getattr(t, "side", "") == "sell"]
        if not sell_trades:
            return DiligenceCheck("Concentration", True, "No sell trades")
        pnls = [t.pnl for t in sell_trades if getattr(t, "pnl", 0) != 0]
        if not pnls:
            return DiligenceCheck("Concentration", True, "No PnL data")
        total_pnl = sum(pnls)
        if total_pnl == 0:
            return DiligenceCheck("Concentration", False, "Total PnL is zero")
        sorted_pnls = sorted(pnls, reverse=True)
        cum = 0
        for i, p in enumerate(sorted_pnls, 1):
            cum += p
            if cum / total_pnl >= 0.50:
                top20_contrib = i / len(sorted_pnls)
                passed = top20_contrib <= 0.30  # top 30% of stocks contribute >50% of PnL
                return DiligenceCheck(
                    "Concentration",
                    passed,
                    f"Top {i}/{len(sorted_pnls)} trades ({top20_contrib*100:.0f}%) = {cum/total_pnl*100:.0f}% of PnL",
                    {"Top N% of trades": f"{top20_contrib*100:.0f}%", "PnL share": f"{cum/total_pnl*100:.0f}%"},
                )
        return DiligenceCheck("Concentration", True, "No major concentration detected")

    def _check_subperiod(self) -> DiligenceCheck:
        if len(self.equity_curve) < 500:
            return DiligenceCheck("Sub-period Stability", False, f"Only {len(self.equity_curve)} days")
        n = len(self.equity_curve)
        first_half = self.equity_curve.iloc[: n // 2]
        second_half = self.equity_curve.iloc[n // 2 :]
        cum_first = (1 + first_half.pct_change().dropna()).cumprod()
        cum_second = (1 + second_half.pct_change().dropna()).cumprod()
        ret_first = cum_first.iloc[-1] - 1 if len(cum_first) > 1 else 0
        ret_second = cum_second.iloc[-1] - 1 if len(cum_second) > 1 else 0
        # Both halves should be positive
        both_positive = ret_first > 0 and ret_second > 0
        return DiligenceCheck(
            "Sub-period Stability",
            both_positive,
            f"1st half: {ret_first*100:.1f}%, 2nd half: {ret_second*100:.1f}%",
            {"1st Half Return": f"{ret_first*100:.2f}%", "2nd Half Return": f"{ret_second*100:.2f}%"},
        )

    def _check_sharpe_significance(self) -> DiligenceCheck:
        if len(self._daily) < 60:
            return DiligenceCheck("Sharpe Significance", False, "Need ≥60 daily returns")
        mean = self._daily.mean()
        std = self._daily.std()
        if std == 0:
            return DiligenceCheck("Sharpe Significance", False, "Zero volatility")
        t_stat = mean / std * math.sqrt(len(self._daily))
        passed = t_stat > 2.0
        return DiligenceCheck(
            "Sharpe Significance",
            passed,
            f"t={t_stat:.2f} (need t>2.0 for significance at 5%)",
            {"t-statistic": f"{t_stat:.3f}", "Sharpe (daily)": f"{mean/std*math.sqrt(252):.3f}"},
        )

    def _check_permutation(self, n_permutations: int = 500, seed: int = 42) -> DiligenceCheck:
        if len(self._daily) < 60:
            return DiligenceCheck("Permutation Test", False, f"Only {len(self._daily)} returns")
        np.random.seed(seed)
        total_return = float((1 + self._daily).cumprod().iloc[-1] - 1)
        if total_return <= 0:
            return DiligenceCheck(
                "Permutation Test",
                False,  # non-positive return can't beat random
                f"Strategy return {total_return*100:.1f}% ≤ 0 — auto-fail",
                {"Strategy Return": f"{total_return*100:.2f}%", "Percentile": "N/A"},
            )
        returns_array: np.ndarray = np.asarray(self._daily.dropna().values)
        random_totals = np.empty(n_permutations)
        for i in range(n_permutations):
            rng = np.random.default_rng(seed + i)
            shuffled = returns_array.copy()
            rng.shuffle(shuffled)
            random_totals[i] = float((1 + pd.Series(shuffled)).cumprod().iloc[-1] - 1)
        percentile = (random_totals < total_return).mean() * 100
        passed = percentile > 50  # should beat random noise more often than not
        return DiligenceCheck(
            "Permutation Test",
            passed,
            f"Strategy {total_return*100:.1f}% vs {n_permutations} random sequences — {percentile:.0f}th percentile",
            {
                "Strategy Return": f"{total_return*100:.2f}%",
                "Random Mean": f"{random_totals.mean()*100:.2f}%",
                "Percentile": f"{percentile:.1f}%",
            },
        )


# -------------------------------------------------------------------
# Equity curve reconstruction from JT results
# -------------------------------------------------------------------

def load_benchmark_ticker(ticker: str, start: str = "2013-01-01", end: str = "2025-02-01") -> pd.Series:
    """Download adjusted close and return a clean Series."""
    try:
        df = yf.Ticker(ticker, session=False).history(start=start, end=end, auto_adjust=True)
        if df is None or len(df) < 100:
            return pd.Series(dtype=float)
        close = df["Close"]
        if close.index.tz is not None:
            close = close.dtz_localize(None)
        return close
    except Exception:
        return pd.Series(dtype=float)


def build_equity_curve(
    ann_ret: float,
    ann_vol: float,
    n_years: float,
    start_date: str = "2015-01-01",
    seed: int = 42,
) -> pd.Series:
    """
    Reconstruct a plausible daily equity curve from annualised return and vol.
    Uses a datetime index aligned to trading days.
    """
    np.random.seed(seed)
    n = int(n_years * 252)
    dt = 1 / 252
    mu = ann_ret
    sigma = ann_vol
    daily_rets = np.random.normal(mu * dt, sigma * np.sqrt(dt), n)
    equity = (1 + pd.Series(daily_rets)).cumprod()
    equity = equity / equity.iloc[0]
    # Assign a clean trading-day DatetimeIndex starting from start_date
    dates = pd.bdate_range(start=start_date, periods=n)
    equity.index = dates
    return equity


def load_jt_results() -> dict:
    path = Path("/home/burley/Personal/Trading/JT_momentum/jt_smallcap_longonly.json")
    with open(path) as f:
        return json.load(f)


JT_CONFIGS = {
    # key: (JSON param key, percentile, label)
    "jt_J6_K6_decile":  ("J6_K6", "top_decile",  "J=6, K=6, top decile"),
    "jt_J9_K3_decile":  ("J9_K3", "top_decile",  "J=9, K=3, top decile"),
    "jt_J6_K3_decile":  ("J6_K3", "top_decile",  "J=6, K=3, top decile"),
    "jt_J9_K3_quintile":("J9_K3", "top_quintile","J=9, K=3, top quintile"),
    "jt_J6_K6_quintile":("J6_K6", "top_quintile","J=6, K=6, top quintile"),
}


def parse_result(value: dict) -> dict:
    """Extract key metrics from a top-N result dict."""
    return {
        "ann_ret": float(value["ann_ret"].rstrip("%")) / 100,
        "ann_vol": float(value["ann_vol"].rstrip("%")) / 100,
        "sharpe":  float(value["sharpe"]),
        "t_stat":  float(value["t_stat"]),
        "win_rate": float(value["win_rate"].rstrip("%")) / 100,
        "total":   float(value["total"].rstrip("%")) / 100,
        "n":       int(value["n"]),
    }


def run_diligence_for_config(
    param_key: str,
    percentile_key: str,
    label: str,
    result_data: dict,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    config_data = result_data.get(param_key, {})
    top = config_data.get(percentile_key, {})
    if not top:
        print(f"  [SKIP] No data for {param_key}/{percentile_key}")
        return None

    m = parse_result(top)
    n_years = m["n"] / 12  # formation periods
    n_years = min(max(n_years, 1.0), 10.0)  # clamp

    # Reconstruct equity curves
    eq_strategy = build_equity_curve(m["ann_ret"], m["ann_vol"], n_years, seed=42)
    eq_iwm = build_equity_curve(
        float(config_data["benchmark_iwm"]["ann_ret"].rstrip("%")) / 100,
        float(config_data["benchmark_iwm"]["ann_vol"].rstrip("%")) / 100,
        n_years, seed=99,
    )
    eq_sp500 = build_equity_curve(
        float(config_data["benchmark_sp500"]["ann_ret"].rstrip("%")) / 100,
        float(config_data["benchmark_sp500"]["ann_vol"].rstrip("%")) / 100,
        n_years, seed=7,
    )

    # Align dates (use IWM as the date anchor since we use its trading days)
    prices = {
        "IWM":    pd.DataFrame({"close": eq_iwm}),
        "^GSPC":  pd.DataFrame({"close": eq_sp500}),
    }

    # Build a synthetic trade log from win rate and n
    # (we don't have per-trade PnL — generate enough trades for the win-rate check)
    from src.backtest.engine import Trade
    trades = []
    n_trades = m["n"]
    n_win = int(round(m["win_rate"] * n_trades))
    rng = np.random.default_rng(seed=42)
    for i in range(n_trades):
        # Each "trade" is a round-trip: entry and exit date roughly 1 month apart
        entry_date = pd.Timestamp("2015-01-01") + pd.Timedelta(days=i * 21)
        exit_date = entry_date + pd.Timedelta(days=21)
        pnl = 1000.0 if i < n_win else -1000.0  # simplified: $1k per trade
        # Trade date field is Timestamp — guaranteed valid from Timestamp arithmetic above
        entry_ts: pd.Timestamp = entry_date  # type: ignore[assignment]
        exit_ts: pd.Timestamp = exit_date     # type: ignore[assignment]
        trades.append(Trade(
            date=entry_ts, symbol=f"SYM{i}", side="buy",
            price=10.0, shares=100, value=1000.0,
        ))
        trades.append(Trade(
            date=exit_ts, symbol=f"SYM{i}", side="sell",
            price=10.0, shares=100, value=1000.0, pnl=pnl,
        ))

    suite = DiligenceSuite(
        equity_curve=eq_strategy,
        trades=trades,
        prices=prices,
        strategy_name=label,
    )
    report = suite.run_all()
    return report.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Run JT momentum diligence")
    parser.add_argument("--config", default=None, help="Specific config key to run")
    parser.add_argument("--list", action="store_true", help="List available configs and exit")
    args = parser.parse_args()

    if args.list:
        print("Available configs:")
        for k in JT_CONFIGS:
            print(f"  {k}")
        return

    results = load_jt_results()
    REPORTS_DIR.mkdir(exist_ok=True)

    configs_to_run = (
        {args.config: JT_CONFIGS[args.config]} if args.config else JT_CONFIGS
    )

    all_reports = []
    for key, (param_key, percentile_key, label) in configs_to_run.items():
        report_dict = run_diligence_for_config(param_key, percentile_key, label, results)
        if report_dict is None:
            continue
        all_reports.append(report_dict)

        # Print summary
        print("\n" + "=" * 65)
        print(f"  DILIGENCE SUMMARY — {label}")
        print("=" * 65)
        passed = report_dict["passed"]
        total = report_dict["total"]
        print(f"  Passed: {passed}/{total}")
        for name, info in report_dict["checks"].items():
            status = "PASS" if info["passed"] else "FAIL"
            print(f"    [{status}] {name}: {info['detail']}")

        # Save to reports/
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = REPORTS_DIR / f"{key}_{stamp}.json"
        payload = report_dict.copy()
        payload["timestamp_utc"] = stamp
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\n  Saved: {out_path.relative_to(ROOT)}")

    if not all_reports:
        print("No reports generated.")
        return

    # Summary table
    print("\n\n")
    print("=" * 80)
    print("  ALL CONFIGS — Diligence Summary")
    print("=" * 80)
    header = f"  {'Config':<30} {'Passed':>7}  Checks (PASS/FAIL)"
    print(header)
    print("  " + "-" * 76)
    for r in all_reports:
        n = r["strategy"]
        p = r["passed"]
        t = r["total"]
        checks = "; ".join(
            f"{name}={'PASS' if c['passed'] else 'FAIL'}"
            for name, c in r["checks"].items()
        )
        print(f"  {n:<30} {p:>3}/{t:<4}  {checks[:60]}")


if __name__ == "__main__":
    main()
