"""Reusable diligence/validation suite for any strategy backtest.

Usage:
    from src.validation.diligence import DiligenceSuite
    suite = DiligenceSuite(equity_curve, trades, prices)
    report = suite.run_all()
    print(report.summary())
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class DiligenceCheck:
    name: str
    passed: bool
    detail: str
    metrics: dict = field(default_factory=dict)


@dataclass
class DiligenceReport:
    strategy_name: str = ""
    checks: list[DiligenceCheck] = field(default_factory=list)

    def add(self, check: DiligenceCheck) -> None:
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
            "checks": {c.name: {"passed": c.passed, "detail": c.detail, "metrics": c.metrics}
                       for c in self.checks},
        }


class DiligenceSuite:
    """Run standard diligence checks on any backtest result."""

    def __init__(
        self,
        equity_curve: pd.Series,
        trades: list,
        prices: dict[str, pd.DataFrame],
        config: Optional[dict] = None,
        strategy_name: str = "Strategy",
    ):
        self.equity_curve = equity_curve
        self.trades = trades
        self.prices = prices
        self.config = config or {}
        self.strategy_name = strategy_name
        self._daily_returns = equity_curve.pct_change().dropna()
        self._monthly_returns = equity_curve.resample("ME").last().pct_change().dropna()

    def run_all(self) -> DiligenceReport:
        report = DiligenceReport(strategy_name=self.strategy_name)

        report.add(self._check_buy_hold())
        report.add(self._check_best_month())
        report.add(self._check_win_rate())
        report.add(self._check_max_dd_duration())
        report.add(self._check_monthly_consistency())
        report.add(self._check_concentration())
        report.add(self._check_subperiod())
        report.add(self._check_sharpe_significance())

        return report

    # ---- Individual checks ----

    def _check_buy_hold(self) -> DiligenceCheck:
        n = len(self.prices)
        if n == 0:
            return DiligenceCheck("vs Equal-Weight B&H", False, "No price data")

        pf = pd.DataFrame({s: self.prices[s]["close"] for s in self.prices})
        ew = pf.mean(axis=1)
        ew_rets = ew.pct_change().dropna()

        strat_rets = self._daily_returns
        common = strat_rets.index.intersection(ew_rets.index)
        s = strat_rets.loc[common].dropna()
        b = ew_rets.loc[common].dropna()

        if len(s) < 20:
            return DiligenceCheck("vs Equal-Weight B&H", False, "Too few overlapping days")

        strat_cum = (1 + s).cumprod()
        bh_cum = (1 + b).cumprod()
        strat_ret = strat_cum.iloc[-1] - 1
        bh_ret = bh_cum.iloc[-1] - 1

        # Annualized Sharpe comparison
        strat_sharpe = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
        bh_sharpe = b.mean() / b.std() * np.sqrt(252) if b.std() > 0 else 0

        outperf = strat_ret > bh_ret
        return DiligenceCheck(
            "vs Equal-Weight B&H",
            outperf,
            f"Strategy {strat_ret*100:.1f}% vs B&H {bh_ret*100:.1f}%",
            {"Strategy Return": f"{strat_ret*100:.2f}%",
             "B&H Return": f"{bh_ret*100:.2f}%",
             "Strategy Sharpe": f"{strat_sharpe:.2f}",
             "B&H Sharpe": f"{bh_sharpe:.2f}"},
        )

    def _check_best_month(self) -> DiligenceCheck:
        if len(self._monthly_returns) < 3:
            return DiligenceCheck("Best Month Exclusion", False, "Need >3 months of data")

        total = (1 + self._monthly_returns).prod() - 1
        best_idx = self._monthly_returns.idxmax()
        best_val = self._monthly_returns.max()
        excl = self._monthly_returns.drop(best_idx)
        excl_ret = (1 + excl).prod() - 1

        best_pct = (total - excl_ret) / total if total != 0 else 0
        survived = excl_ret > 0

        return DiligenceCheck(
            "Best Month Exclusion",
            survived,
            f"Without best month ({best_val*100:.1f}%): {excl_ret*100:.1f}% (total: {total*100:.1f}%)",
            {"Best Month": f"{best_val*100:.2f}%",
             "Return (all)": f"{total*100:.2f}%",
             "Return (excl)": f"{excl_ret*100:.2f}%",
             "Attributed to 1 month": f"{best_pct*100:.0f}%"},
        )

    def _check_win_rate(self) -> DiligenceCheck:
        sell_trades = [t for t in self.trades if getattr(t, "side", "") == "sell" and getattr(t, "pnl", 0) != 0]
        if not sell_trades and self.trades:
            sell_trades = [t for t in self.trades if getattr(t, "pnl", 0) != 0]
        if not sell_trades:
            monthly_up = (self._monthly_returns > 0).sum()
            monthly_total = len(self._monthly_returns)
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

    def _check_max_dd_duration(self) -> DiligenceCheck:
        if len(self.equity_curve) < 20:
            return DiligenceCheck("Max DD Duration", False, "Too little data")

        cum = self.equity_curve / self.equity_curve.cummax() - 1
        dd_series = cum[cum < 0]
        if len(dd_series) == 0:
            return DiligenceCheck("Max DD Duration", True, "No drawdowns")

        max_dd = dd_series.min()

        # Find longest drawdown period
        in_dd = cum < 0
        dd_streaks = []
        current_len = 0
        for val in in_dd:
            if val:
                current_len += 1
            elif current_len > 0:
                dd_streaks.append(current_len)
                current_len = 0
        if current_len > 0:
            dd_streaks.append(current_len)

        longest_dd = max(dd_streaks) if dd_streaks else 0

        # Pass if max DD is manageable and recovery is reasonable
        passed = max_dd > -0.5 and longest_dd < 252 * 2  # 50% DD, 2 years recovery
        return DiligenceCheck(
            "Max Drawdown Analysis",
            passed,
            f"Max DD: {max_dd*100:.1f}%, longest streak: {longest_dd} days",
            {"Max DD": f"{max_dd*100:.2f}%",
             "Longest DD Days": str(longest_dd)},
        )

    def _check_monthly_consistency(self) -> DiligenceCheck:
        if len(self._monthly_returns) < 6:
            return DiligenceCheck("Monthly Consistency", False, "Need >6 months")

        # Rolling 6-month return: what fraction are positive?
        rolling = self._monthly_returns.rolling(6).apply(lambda x: (1 + x).prod() - 1).dropna()
        pos_windows = (rolling > 0).sum()
        total_windows = len(rolling)
        consistency = pos_windows / total_windows if total_windows > 0 else 0

        # Also check monthly Sharpe
        monthly_sharpe = self._monthly_returns.mean() / self._monthly_returns.std() if self._monthly_returns.std() > 0 else 0

        passed = consistency > 0.5 and monthly_sharpe > 0
        return DiligenceCheck(
            "Monthly Consistency",
            passed,
            f"{pos_windows}/{total_windows} rolling 6m windows positive, monthly Sharpe: {monthly_sharpe:.2f}",
            {"Positive 6m Windows": f"{pos_windows}/{total_windows} ({consistency*100:.0f}%)",
             "Monthly Sharpe": f"{monthly_sharpe:.2f}"},
        )

    def _check_concentration(self) -> DiligenceCheck:
        sell_trades = [t for t in self.trades if getattr(t, "side", "") == "sell" and getattr(t, "pnl", 0) != 0]
        if not sell_trades and self.trades:
            sell_trades = [t for t in self.trades if getattr(t, "pnl", 0) != 0]
        if len(sell_trades) < 5:
            return DiligenceCheck("Trade Concentration", True, f"Only {len(sell_trades)} trades — insufficient for analysis")

        pnls = np.array([abs(t.pnl) for t in sell_trades if hasattr(t, "pnl")])
        if len(pnls) == 0:
            return DiligenceCheck("Trade Concentration", True, "No PnL data")

        top1_pct = pnls.max() / pnls.sum() * 100
        top3_pct = sum(sorted(pnls, reverse=True)[:3]) / pnls.sum() * 100 if len(pnls) >= 3 else 100

        passed = top1_pct < 50  # No single trade should dominate
        return DiligenceCheck(
            "Trade Concentration",
            passed,
            f"Top 1 trade: {top1_pct:.0f}% of total PnL, Top 3: {top3_pct:.0f}%",
            {"Top Trade Share": f"{top1_pct:.1f}%",
             "Top 3 Share": f"{top3_pct:.1f}%",
             "Total Trades": str(len(sell_trades))},
        )

    def _check_subperiod(self) -> DiligenceCheck:
        if len(self._daily_returns) < 100:
            return DiligenceCheck("Sub-Period Consistency", False, "Too little data")

        mid = len(self._daily_returns) // 2
        first = (1 + self._daily_returns.iloc[:mid]).prod() - 1
        second = (1 + self._daily_returns.iloc[mid:]).prod() - 1

        # Pass if both halves are positive or the second half isn't dramatically worse
        both_positive = first > 0 and second > 0
        second_not_terrible = second > first * 0.5  # second half at least 50% as good
        passed = both_positive or (first > 0 and second_not_terrible)

        return DiligenceCheck(
            "Sub-Period Consistency",
            passed,
            f"First half: {first*100:.1f}%, Second half: {second*100:.1f}%",
            {"First Half Return": f"{first*100:.2f}%",
             "Second Half Return": f"{second*100:.2f}%"},
        )

    def _check_sharpe_significance(self) -> DiligenceCheck:
        if len(self._monthly_returns) < 12:
            return DiligenceCheck("Sharpe Significance", False, "Need >12 months")

        sharpe = self._monthly_returns.mean() / self._monthly_returns.std() * np.sqrt(12) if self._monthly_returns.std() > 0 else 0
        n_months = len(self._monthly_returns)

        # Simple significance: Sharpe > 2/sqrt(N) is ~ statistically significant
        threshold = 2 / np.sqrt(n_months)
        passed = sharpe > threshold

        return DiligenceCheck(
            "Sharpe Significance",
            passed,
            f"Sharpe {sharpe:.2f} vs threshold {threshold:.2f} for N={n_months} months",
            {"Annual Sharpe": f"{sharpe:.2f}",
             "Significance Threshold": f"{threshold:.2f}"},
        )
